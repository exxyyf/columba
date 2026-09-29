"""Этап 7: HTTP-сервис инференса (FastAPI).

Запуск:

    uv run uvicorn columba.service:app --host 0.0.0.0 --port 8000

(или через `Dockerfile` — см. README, раздел «Сборка и запуск в Docker»).
Сервис — тонкая
HTTP-обёртка над уже существующим `inference.run_inference`, никакой новой
логики инференса здесь нет: те же чекеры, тот же агрегатор, тот же формат
сабмита (`submission.SUBMISSION_COLUMNS`, см. README «Формат входных и
выходных данных»). Seed
зафиксирован (`config.SEED`), внешних сетевых вызовов при инференсе нет —
сервис только принимает файлы по HTTP и отвечает, ничего не запрашивает
вовне.

Эндпоинты:

* `GET /` — редирект на `/docs`. Веб-интерфейс один — просмотрщик
  `interface/` (галерея, загрузка, контроль качества, разбор снимка);
  он ходит сюда в `/predict` и `/visualize`.
* `POST /predict` — список DICOM-файлов. Без `paths` — плоский список
  (сторона бедра методом `content_only`, парное разведение недоступно, т.к.
  структура исследований не передаётся); с `paths` (задача 9.6, по одному
  относительному пути на каждый файл в том же порядке — папочная загрузка
  веб-интерфейса) — структура восстанавливается на диске, парное разведение
  стороны бедра внутри исследования доступно, как у `/predict/zip`.
* `POST /predict/zip` — ZIP-архив каталога, устроенного как закрытый тест
  (плоский или с папками исследований) — сохраняет структуру, значит и
  парное определение стороны бедра. Предпочтительный способ для реального
  батча без браузера (`curl`/скрипт).
* `POST /visualize` — один DICOM-файл -> оверлей ориентиров + результаты
  чекеров, тем же путём региона/стороны, что `/predict` (этап 8, бонус 1 +
  этап 9, п. 9.6, `visualize.py` — никакой новой логики детекции, только
  отрисовка уже посчитанного). `?format=json` — те же результаты как
  структурированные данные вместо PNG.

`/predict`/`/predict/zip` возвращают CSV (`text/csv`, ровно формат
`submission.save_submission`) по умолчанию; `?format=json` — тот же сабмит
построчно в JSON.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response

from . import config as cfg
from .dicom_io import STATUS_SUCCESS, normalize, read_dicom
from .hip_cnn import load_hip_positioning_predictor
from .inference import describe_inputs, run_inference
from .inventory import DICOM_SUFFIX
from .region_cnn import default_device
from .visualize import figure_to_png_bytes, render_violation_overlay

app = FastAPI(
    title="columba",
    description="Определение качества исследования денситограммы (формат — README, «Формат входных и выходных данных»).",
    version="0.1.0",
)


@app.get("/")
def index() -> RedirectResponse:
    """Своего веб-интерфейса у API нет — единый интерфейс живёт в `interface/`."""
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    """Статус сервиса: устройство инференса и наличие весов CNN-веток.

    Все ветки опциональны (см. WEIGHTS.md/README) — сервис работает и без
    них (честная деградация каждого чекера/предиктора), но `hip_positioning`
    без CNN даёт заметно более слабый сигнал (`stage_4_decisions.md`).
    """
    weights = {
        "region_cnn": cfg.REGION_CNN_WEIGHTS.exists(),
        "spine_keypoint_unet": cfg.SPINE_UNET_WEIGHTS.exists(),
        "spine_objects_cnn": cfg.SPINE_OBJECTS_CNN_WEIGHTS.exists(),
        "hip_positioning_cnn": cfg.HIP_POSITIONING_CNN_WEIGHTS.exists(),
    }
    return {
        "status": "ok",
        "device": str(default_device()),
        "seed": cfg.SEED,
        "weights_loaded": weights,
    }


def _is_safe_zip_member(name: str) -> bool:
    """Защита от zip-slip: без абсолютных путей и `..` в частях пути."""
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def _dedupe_filename(name: str, seen: dict[str, int]) -> str:
    """Разрешить коллизию имён без потери файла (задача 9.3).

    Первое появление имени сохраняет его как есть, повторные получают
    суффикс `__N` перед последним расширением, СОХРАНЯЯ любой каталожный
    префикс (`study_001/CR000000.dcm` -> `study_001/CR000000__1.dcm`, не
    `CR000000__1.dcm` без папки — задача 9.6, структурная загрузка папки).
    `PurePosixPath`, не `Path`: `name` здесь всегда логический posix-путь
    (ключ словаря/значение из `webkitRelativePath`), а не путь файловой
    системы текущей ОС — на Windows-сервере обычный `Path` превратил бы
    разделители в бэкслеши и сломал сравнение со строками, которые прислал
    браузер. Тот же алгоритм в порядке добавления файлов может повторить
    любой клиент (раньше — `static/app.js`, теперь пачки шлёт
    `interface/src/quality.ts` с уникальными `relative_path`) — клиент и сервер видят файлы в одном
    порядке (FastAPI сохраняет порядок частей формы с одинаковым именем
    поля), поэтому независимо вычисленные уникальные имена совпадают без
    обмена дополнительными данными, и таблица результатов (`row.file_name`)
    остаётся кликабельной в интерфейсе для обоих режимов загрузки (плоского
    и папочного).
    """
    count = seen.get(name, 0)
    seen[name] = count + 1
    if count == 0:
        return name
    path = PurePosixPath(name)
    return str(path.with_stem(f"{path.stem}__{count}"))


def _save_uploads_flat(files: list[UploadFile], target_dir: Path) -> None:
    """Сохранить плоский список загрузок без перезаписи одноимённых файлов.

    Проверено экспериментально: браузер шлёт `UploadFile.filename` как
    голое имя файла (без директорий) для обычного `<input type=file
    multiple>` — в этом интерфейсе `webkitdirectory` не используется, так
    что `webkitRelativePath` всегда пуст и не несёт дополнительной
    информации. Коллизия базового имени (например, два файла `CR000000.dcm`
    из разных папок закрытого теста, выбранные в одном диалоге) раньше молча
    затирала файл на диске ДО инференса — теперь второй и последующие
    файлы получают уникализированное имя (`_dedupe_filename`), ни один
    файл не теряется.
    """
    seen: dict[str, int] = {}
    for index, upload in enumerate(files):
        name = Path(upload.filename or f"file_{index}.dcm").name  # без вложенных путей — плоско
        name = _dedupe_filename(name, seen)
        with (target_dir / name).open("wb") as out:
            shutil.copyfileobj(upload.file, out)


def _save_uploads_structured(files: list[UploadFile], relpaths: list[str], target_dir: Path) -> None:
    """Сохранить пачку С сохранением относительной структуры папок (задача 9.6).

    `relpaths[i]` — путь файла `files[i]` относительно ВЫБРАННОЙ пользователем
    папки, БЕЗ имени самой этой папки (браузерный клиент отрезает
    первый сегмент `File.webkitRelativePath` сам, тем же способом, каким
    `/predict/zip` не видит имя корневого каталога архива). Значит
    `study_folder`, который из этой структуры посчитает
    `inference._study_key`, — те же подпапки, что реально выбрал
    пользователь, и парное определение стороны бедра внутри исследования
    работает как у `/predict/zip`, не как у плоского `/predict`.

    Та же защита от zip-slip, что у `_extract_zip` (`_is_safe_zip_member`) —
    `relpaths` пришли от клиента, доверять им нельзя. Коллизии — тем же
    `_dedupe_filename`, что и у плоской загрузки, но работающим на ПОЛНОМ
    относительном пути (сохраняет папку в имени коллизии).
    """
    if len(files) != len(relpaths):
        raise HTTPException(400, "files и paths не совпадают по количеству")
    seen: dict[str, int] = {}
    for upload, relpath in zip(files, relpaths):
        relpath = relpath.replace("\\", "/") or Path(upload.filename or "upload.dcm").name
        if not _is_safe_zip_member(relpath):
            raise HTTPException(400, f"недопустимый относительный путь: {relpath}")
        relpath = _dedupe_filename(relpath, seen)
        dest = target_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            shutil.copyfileobj(upload.file, out)


def _extract_zip(data: bytes, target_dir: Path) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = [m for m in archive.infolist() if not m.is_dir() and _is_safe_zip_member(m.filename)]
            archive.extractall(target_dir, members=members)
    except zipfile.BadZipFile as exc:
        raise HTTPException(400, "не удалось распаковать zip") from exc


def _run_and_respond(input_dir: Path, fmt: Literal["csv", "json"]) -> JSONResponse | PlainTextResponse:
    if not any(input_dir.rglob(f"*{DICOM_SUFFIX}")):
        raise HTTPException(400, f"во входных данных нет файлов {DICOM_SUFFIX}")
    started = time.perf_counter()
    try:
        submission = run_inference(input_dir)
    except ValueError as exc:  # ValueError покрывает и «нет файлов», и SubmissionError (её подкласс)
        raise HTTPException(400, str(exc)) from exc
    elapsed = time.perf_counter() - started

    if fmt == "json":
        return JSONResponse(
            {
                "rows": submission.to_dict(orient="records"),
                "n_files": len(submission),
                "elapsed_seconds": round(elapsed, 3),
            }
        )
    return PlainTextResponse(submission.to_csv(index=False), media_type="text/csv")


@app.post("/predict", response_model=None)
async def predict(
    files: list[UploadFile] = File(...),
    paths: list[str] | None = Form(None),
    fmt: Literal["csv", "json"] = Query("csv", alias="format"),
) -> JSONResponse | PlainTextResponse:
    """Плоский список файлов, либо (задача 9.6) папка с относительной
    структурой, если клиент прислал `paths` (по одному на каждый `files[i]`,
    в том же порядке — так шлёт пачки `interface/src/quality.ts`). С `paths` доступно
    парное определение стороны бедра внутри исследования, как у `/predict/zip`
    (`_save_uploads_structured`); без него — плоский режим без изменений
    (`_save_uploads_flat`, метод `content_only`).
    """
    if not files:
        raise HTTPException(400, "нет загруженных файлов")
    with tempfile.TemporaryDirectory(prefix="columba_predict_") as tmp:
        tmp_path = Path(tmp)
        if paths:
            _save_uploads_structured(files, paths, tmp_path)
        else:
            _save_uploads_flat(files, tmp_path)
        return _run_and_respond(tmp_path, fmt)


@app.post("/predict/zip", response_model=None)
async def predict_zip(
    file: UploadFile = File(...),
    fmt: Literal["csv", "json"] = Query("csv", alias="format"),
) -> JSONResponse | PlainTextResponse:
    data = await file.read()
    with tempfile.TemporaryDirectory(prefix="columba_predict_zip_") as tmp:
        tmp_path = Path(tmp)
        _extract_zip(data, tmp_path)
        return _run_and_respond(tmp_path, fmt)


def _json_safe(value):
    """Рекурсивно заменить NaN на `None` — `json.dumps` по умолчанию не
    принимает NaN как валидный JSON (в отличие от `signals`-сайдкаров
    `inference.py`, которые пишут NaN на диск в собственном формате)."""
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _checker_result_to_json(result) -> dict:
    return _json_safe(result.to_dict())


@app.post("/visualize", response_model=None)
async def visualize(
    file: UploadFile = File(...),
    fmt: Literal["png", "json"] = Query("png", alias="format"),
) -> Response | JSONResponse:
    """Один DICOM-файл -> оверлей ориентиров + результаты чекеров.

    Регион/сторона определяются ТЕМ ЖЕ путём, что `/predict`
    (`inference.describe_inputs` — эвристика этапа 1 + CNN-арбитраж, если
    веса на месте), не отдельной упрощённой эвристикой внутри `visualize.py`
    — иначе на кадре, где эвристика и CNN расходятся, `/visualize` мог бы
    показать другой регион/сторону, чем `/predict` для того же файла
    (задача 9.6, «тест консистентности» — `test_visualize_matches_predict_region_and_side`
    в `test_service.py`). Как и у `/predict` без ZIP (одиночный файл, без
    структуры исследования), сторона бедра не может участвовать в парном
    разведении между снимками одного пациента — то же ограничение метода
    `content_only`, что уже описано для `/predict`.

    `?format=json` — тот же регион/сторона/результаты чекеров, что рисуются
    в PNG, но как структурированные данные, не как текст заголовка внутри
    картинки (задача 9.6, WCAG — текст должен быть доступен как HTML, не
    впечатан в пиксели). `hip_cnn.load_hip_positioning_predictor()` кэширует
    предиктор по пути весов (`hip_cnn._PREDICTOR_CACHE`) — повторные вызовы
    не перезагружают модель с диска.
    """
    data = await file.read()
    with tempfile.TemporaryDirectory(prefix="columba_visualize_") as tmp:
        tmp_path = Path(tmp)
        upload_path = tmp_path / (Path(file.filename or "upload.dcm").name)
        upload_path.write_bytes(data)
        manifest = describe_inputs(tmp_path, spine_checkers=False, hip_checkers=False)
        if manifest.empty or manifest.iloc[0]["read_status"] != STATUS_SUCCESS:
            error = manifest.iloc[0]["read_error"] if not manifest.empty else "файл не распознан"
            raise HTTPException(400, f"не удалось прочитать DICOM: {error}")
        row = manifest.iloc[0]
        result = read_dicom(row["abs_path"])
        pixels = normalize(result.pixels, result.tags)
        predictor = load_hip_positioning_predictor()
        fig, info = render_violation_overlay(
            pixels,
            region=row["region_final"],
            side=row["hip_side_final"] if pd.notna(row["hip_side_final"]) else None,
            hip_cnn_predictor=predictor,
        )
        if fmt == "json":
            import matplotlib.pyplot as plt

            plt.close(fig)
            return JSONResponse(
                {
                    "region": info["region"],
                    "side": info.get("side"),
                    "checkers": {key: _checker_result_to_json(value) for key, value in info["checkers"].items()},
                }
            )
        png = figure_to_png_bytes(fig)
    return Response(content=png, media_type="image/png")
