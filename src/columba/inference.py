"""Прогон по выборке, устроенной как закрытый тест.

Организаторы подтвердили: закрытый тест снят на тех же аппаратах, типы
исследований те же, но ни дополнительных тегов, ни меток зоны, ни суффиксов
региона в именах файлов не будет. Поэтому здесь НЕ используется ничего, кроме
пиксельных данных: регион — по ширине кадра, сторона бедра — по содержимому,
CNN этапа 1 (если веса на месте) страхует эвристику через арбитраж.

Этап 3 подключает агрегатор чекеров позвоночника (`aggregate.py`) как
предиктор по умолчанию: `run_inference(input_dir)` без явного `predictor`
теперь отдаёт настоящий сабмит (позвоночник — реальные предсказания из
чекеров этапа 2, бедро — приор `config.HIP_PRIOR_QUALITY_PROB` до чекеров
бедра этапов 4-5). Явный `predictor=None` по-прежнему принудительно даёт
«нулевой» сабмит формата раздела 1.2 — это используется в тестах формата и
для отладки, когда предсказания намеренно не нужны.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import config as cfg
from .dedup import assign_dedup_groups, pixel_hash
from .dicom_io import STATUS_SUCCESS, normalize, read_dicom
from .inventory import DICOM_SUFFIX
from .regions import (
    SIDE_LEFT,
    SIDE_RIGHT,
    classify_region_from_pixels,
    hip_side,
    resolve_sides_within_study,
)
from .submission import build_submission, validate_submission

if TYPE_CHECKING:
    from .region_cnn import RegionCnnPredictor

Predictor = Callable[[pd.DataFrame], pd.DataFrame]

# Сентинел «загрузить CNN, если веса на месте» (ветка по умолчанию).
AUTO = "auto"


def list_input_files(input_dir: Path | str) -> list[Path]:
    """Все .dcm во входном каталоге, плоском или вложенном."""
    return sorted(p for p in Path(input_dir).rglob(f"*{DICOM_SUFFIX}") if p.is_file())


def describe_inputs(
    input_dir: Path | str,
    *,
    region_predictor: "RegionCnnPredictor | str | None" = AUTO,
    spine_checkers: str | bool = AUTO,
    hip_checkers: str | bool = AUTO,
) -> pd.DataFrame:
    """Манифест инференса: строка на файл, признаки только из пикселей.

    `region_predictor`: AUTO — подхватить CNN этапа 1, если веса лежат в
    `config.REGION_CNN_WEIGHTS` (иначе чистая эвристика); None — принудительно
    ветка «только эвристика»; экземпляр `RegionCnnPredictor` — использовать его.
    Итог арбитража — в колонках `region_final`, `hip_side_final`,
    `region_method`, `region_disagreement`; сырые ответы веток сохраняются.

    `spine_checkers`: AUTO — прогнать чекеры этапа 2 по кадрам с
    `region_final == spine` (CNN предметов подхватывается при наличии весов,
    иначе rule-based); False/None — не прогонять. Выход чекеров — колонки
    `<checker>_score` / `<checker>_flag` / `<checker>_status` и JSON-сайдкар
    `spine_signals`; для бедра значения нейтрально пусты. Дубликаты
    обсчитываются один раз через кэш по хэшу пикселей.

    `hip_checkers`: AUTO — прогнать чекер(ы) бедра этапа 4 по кадрам с
    `region_final == hip`, по образцу `spine_checkers`: колонки
    `hip_positioning_{score,flag,status}` + JSON-сайдкар `hip_signals`;
    сторона берётся из `hip_side_final`. Если модуль ориентиров бедра
    (`columba.hip_landmarks`) ещё не готов (`ImportError`), колонки честно
    получают статус `not_evaluated` с причиной, манифест не падает.
    False/None — не прогонять.
    """
    input_dir = Path(input_dir)
    predictor = _resolve_region_predictor(region_predictor)
    records = []
    cnn_cache: dict[str, object] = {}  # dedup-группы обсчитываются один раз
    for index, path in enumerate(list_input_files(input_dir)):
        result = read_dicom(path)
        # `file_name` = `relative_path` (posix-разделители) — единственный
        # уникальный ключ строки сабмита и в плоском, и в папочном входе
        # (см. stages/stage_9.md, раздел 9.3): голое имя файла коллидирует,
        # когда организаторы снимают суффиксы региона в закрытом тесте (уже
        # набор «Для теста» даёт коллизию после снятия суффиксов).
        relative_path = path.relative_to(input_dir).as_posix()
        record = {
            "file_id": f"f{index:04d}",
            "file_name": relative_path,
            "relative_path": relative_path,
            "abs_path": str(path),
            # Группировка для разведения сторон: папка, если она есть,
            # иначе весь каталог считается одной группой.
            "study_folder": _study_key(path, input_dir),
            "read_status": result.status,
            "read_error": result.error,
        }
        if result.ok:
            pixels = result.pixels
            record["pixel_hash"] = pixel_hash(pixels)
            record["rows"] = int(pixels.shape[0])
            record["cols"] = int(pixels.shape[1])
            record["region"] = classify_region_from_pixels(pixels)
            side = hip_side(pixels) if record["region"] == cfg.REGION_HIP else None
            record["hip_side_score"] = side.score if side else np.nan
            record["hip_side_confident"] = bool(side.confident) if side else False
            record["cnn_label"] = _cnn_label_cached(predictor, record["pixel_hash"], pixels, result.tags, cnn_cache)
            # Дополнительная колонка сабмита (не входит в SUBMISSION_COLUMNS,
            # раздел 1.2): читается напрямую из тега, ни на что в пайплайне
            # не влияет — только для разбора полётов и сверки с организаторами.
            record["sop_instance_uid"] = result.tags.get("SOPInstanceUID")
        else:
            record.update(
                {
                    "pixel_hash": None,
                    "rows": pd.NA,
                    "cols": pd.NA,
                    "region": cfg.REGION_UNKNOWN,
                    "hip_side_score": np.nan,
                    "hip_side_confident": False,
                    "cnn_label": None,
                    "sop_instance_uid": None,
                }
            )
        records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame = assign_dedup_groups(frame)
    frame = _assign_sides(frame)
    frame = _arbitrate(frame, cnn_cache)
    if spine_checkers in (AUTO, True):
        frame = _attach_spine_checkers(frame)
    if hip_checkers in (AUTO, True):
        frame = _attach_hip_checkers(frame)
    return frame


def submission_from_manifest(
    manifest: pd.DataFrame,
    predictor: Predictor | str | None = AUTO,
) -> pd.DataFrame:
    """Собрать и провалидировать сабмит из уже посчитанного манифеста.

    Вынесено из `run_inference`, чтобы вызывающий код (например,
    `stage3_eval`) мог получить сабмит из ОДНОГО прогона `describe_inputs`,
    не пересчитывая чекеры второй раз ради собственного манифеста с
    `pixel_hash` и т. п.

    `predictor`: AUTO (по умолчанию) — подключить агрегатор этапа 3
    (`aggregate.aggregate_predictions`): позвоночник получает настоящие
    `quality_class`/`quality_prob`/`violation_type` из чекеров этапа 2, бедро —
    приор `config.HIP_PRIOR_QUALITY_PROB`. Явный `predictor=None` отключает
    предсказания принудительно — сабмит «нулевой» (все `quality_class=0`),
    но по-прежнему корректный по формату; любой другой `Predictor` (функция
    `manifest -> predictions` с колонками `quality_class`/`quality_prob`/
    `violation_type` и `file_id`/`dedup_group_id`) подставляется как есть.
    """
    resolved_predictor = _resolve_predictor(predictor)
    predictions = resolved_predictor(manifest) if resolved_predictor is not None else None
    # В сабмит идёт итог арбитража; сырая эвристика остаётся в манифесте.
    submission = build_submission(manifest.assign(region=manifest["region_final"]), predictions)
    validate_submission(submission, expected_rows=len(manifest))
    return submission


def run_inference(
    input_dir: Path | str,
    predictor: Predictor | str | None = AUTO,
    *,
    output_csv: Path | str | None = None,
    region_predictor: "RegionCnnPredictor | str | None" = AUTO,
    spine_checkers: str | bool = AUTO,
) -> pd.DataFrame:
    """Построить и провалидировать сабмит по каталогу закрытого теста.

    `predictor` — см. `submission_from_manifest`.
    """
    manifest = describe_inputs(
        input_dir, region_predictor=region_predictor, spine_checkers=spine_checkers
    )
    if manifest.empty:
        raise ValueError(f"во входном каталоге нет файлов {DICOM_SUFFIX}: {input_dir}")

    submission = submission_from_manifest(manifest, predictor)
    if output_csv is not None:
        from .submission import save_submission

        save_submission(submission, output_csv)
    return submission


def _study_key(path: Path, input_dir: Path) -> str:
    relative = path.relative_to(input_dir)
    return relative.parts[0] if len(relative.parts) > 1 else "."


# --------------------------------------------------------------------------- #
# CNN-ветка и арбитраж (этап 1)
# --------------------------------------------------------------------------- #


def _resolve_predictor(predictor):
    """AUTO -> агрегатор этапа 3; None остаётся None (принудительно нулевой сабмит)."""
    if predictor is None:
        return None
    if isinstance(predictor, str) and predictor == AUTO:
        from .aggregate import aggregate_predictions

        return aggregate_predictions
    return predictor


def _resolve_region_predictor(region_predictor):
    if region_predictor is None:
        return None
    if region_predictor == AUTO:
        from .region_cnn import load_region_predictor

        return load_region_predictor()
    return region_predictor


def _cnn_label_cached(predictor, pixel_hash_value, pixels, tags, cache) -> str | None:
    """CNN-предсказание с кэшем по хэшу пикселей: дубликаты не обсчитываются."""
    if predictor is None:
        return None
    if pixel_hash_value not in cache:
        cache[pixel_hash_value] = predictor.predict_pixels(pixels, tags)
    return cache[pixel_hash_value].label


def _arbitrate(frame: pd.DataFrame, cnn_cache: dict) -> pd.DataFrame:
    """Свести эвристику и CNN в итог (`region_final`, `hip_side_final`)."""
    from .region_cnn import arbitrate_region

    frame = frame.copy()
    finals, sides, methods, disagreements = [], [], [], []
    for row in frame.itertuples():
        cnn = cnn_cache.get(row.pixel_hash) if row.pixel_hash else None
        width = int(row.cols) if pd.notna(row.cols) else None
        heuristic_side = row.hip_side if pd.notna(row.hip_side) else None
        verdict = arbitrate_region(width, row.region, heuristic_side, cnn)
        finals.append(verdict.region)
        sides.append(verdict.side)
        methods.append(verdict.method)
        disagreements.append(verdict.disagreement)
        if verdict.disagreement:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: эвристика и CNN разошлись на {row.relative_path}: "
                f"эвристика {row.region}/{heuristic_side}, CNN {verdict.cnn_label}; "
                f"принято {verdict.region} ({verdict.method})",
                flush=True,
            )
    frame["region_final"] = finals
    frame["hip_side_final"] = sides
    frame["region_method"] = methods
    frame["region_disagreement"] = disagreements
    return frame


# --------------------------------------------------------------------------- #
# Чекеры позвоночника (этап 2)
# --------------------------------------------------------------------------- #

SPINE_CHECKER_KEYS = ("spine_axis", "spine_positioning", "spine_objects")


def _attach_spine_checkers(frame: pd.DataFrame) -> pd.DataFrame:
    """Прогнать чекеры этапа 2 по кадрам ПОП и разложить выходы в колонки.

    Дубликаты обсчитываются один раз (кэш по хэшу пикселей). Для бедра и
    нечитаемых файлов колонки нейтрально пусты: решение, что делать с ними
    в сабмите, принимает агрегация этапа 3. Исключение чекера на КОНКРЕТНОМ
    файле (вырожденный кадр, edge-кейс детектора и т. п.) не должно ронять
    весь батч (раздел 1.2: строка на каждый пришедший файл) — такой файл
    честно получает `status=not_evaluated` по всем трём чекерам, причина
    падения пишется в `spine_signals` и в лог (`print(..., flush=True)`),
    чтобы баг не маскировался молча.
    """
    import json as _json

    from .spine_checkers import run_spine_checkers
    from .spine_keypoints import STATUS_NOT_EVALUATED
    from .spine_objects_cnn import load_objects_predictor

    objects_predictor = load_objects_predictor()
    cache: dict[str, dict] = {}
    frame = frame.copy()
    for key in SPINE_CHECKER_KEYS:
        frame[f"{key}_score"] = np.nan
        frame[f"{key}_flag"] = pd.NA
        frame[f"{key}_status"] = pd.NA
    frame["spine_signals"] = pd.NA

    spine_rows = frame[(frame["region_final"] == cfg.REGION_SPINE) & frame["pixel_hash"].notna()]
    for row in spine_rows.itertuples():
        if row.pixel_hash not in cache:
            result = read_dicom(row.abs_path)
            if not result.ok:
                continue
            pixels = normalize(result.pixels, result.tags)
            try:
                checkers = run_spine_checkers(pixels, objects_predictor=objects_predictor)
                cache[row.pixel_hash] = {
                    "columns": {
                        f"{key}_{field}": getattr(checkers[key], field)
                        for key in SPINE_CHECKER_KEYS
                        for field in ("score", "flag", "status")
                    },
                    "signals": _json.dumps(
                        {key: checkers[key].to_dict() for key in SPINE_CHECKER_KEYS},
                        ensure_ascii=False,
                    ),
                }
            except Exception as exc:  # noqa: BLE001 — один упавший файл не должен ронять батч
                reason = f"чекер позвоночника упал: {type(exc).__name__}: {exc}"
                print(
                    f"ПРЕДУПРЕЖДЕНИЕ: чекеры позвоночника упали на {row.relative_path}: {reason}",
                    flush=True,
                )
                cache[row.pixel_hash] = {
                    "columns": {
                        **{f"{key}_score": float("nan") for key in SPINE_CHECKER_KEYS},
                        **{f"{key}_flag": False for key in SPINE_CHECKER_KEYS},
                        **{f"{key}_status": STATUS_NOT_EVALUATED for key in SPINE_CHECKER_KEYS},
                    },
                    "signals": _json.dumps(
                        {
                            key: {
                                "checker": key,
                                "status": STATUS_NOT_EVALUATED,
                                "score": None,
                                "flag": False,
                                "reason": reason,
                                "signals": {},
                            }
                            for key in SPINE_CHECKER_KEYS
                        },
                        ensure_ascii=False,
                    ),
                }
        cached = cache[row.pixel_hash]
        for column, value in cached["columns"].items():
            frame.loc[row.Index, column] = value
        frame.loc[row.Index, "spine_signals"] = cached["signals"]
    return frame


# --------------------------------------------------------------------------- #
# Чекер(ы) бедра (этап 4)
# --------------------------------------------------------------------------- #


def _attach_hip_checkers(frame: pd.DataFrame) -> pd.DataFrame:
    """Прогнать чекер(ы) бедра этапа 4 по кадрам бедра и разложить в колонки.

    По образцу `_attach_spine_checkers`: дубликаты обсчитываются один раз
    (кэш по хэшу пикселей), для позвоночника и нечитаемых файлов колонки
    нейтрально пусты. Любое исключение чекера на КОНКРЕТНОМ файле — модуль
    ориентиров бедра (`columba.hip_landmarks`) ещё не существует
    (`ImportError`), вырожденные кейпоинты, пустой/однородный кадр и т. п. —
    честно сводится к `not_evaluated` на все дубли этой группы, без падения
    всего манифеста; причина падения пишется и в `hip_signals`, и в лог
    (`print(..., flush=True)`), чтобы баг не маскировался молча.
    """
    import json as _json

    from .hip_checkers import HIP_CHECKER_KEYS, run_hip_checkers
    from .hip_cnn import load_hip_positioning_predictor
    from .spine_keypoints import STATUS_NOT_EVALUATED

    hip_positioning_predictor = load_hip_positioning_predictor()
    cache: dict[str, dict] = {}
    frame = frame.copy()
    for key in HIP_CHECKER_KEYS:
        frame[f"{key}_score"] = np.nan
        frame[f"{key}_flag"] = pd.NA
        frame[f"{key}_status"] = pd.NA
    frame["hip_signals"] = pd.NA

    hip_rows = frame[(frame["region_final"] == cfg.REGION_HIP) & frame["pixel_hash"].notna()]
    for row in hip_rows.itertuples():
        if row.pixel_hash not in cache:
            result = read_dicom(row.abs_path)
            if not result.ok:
                continue
            pixels = normalize(result.pixels, result.tags)
            side = getattr(row, "hip_side_final", None)
            side_confident = getattr(row, "hip_side_confident", None)
            try:
                checkers = run_hip_checkers(
                    pixels, side=side, cnn_predictor=hip_positioning_predictor, side_confident=side_confident
                )
                columns = {
                    f"{key}_{field}": getattr(checkers[key], field)
                    for key in HIP_CHECKER_KEYS
                    for field in ("score", "flag", "status")
                }
                signals_json = _json.dumps(
                    {key: checkers[key].to_dict() for key in HIP_CHECKER_KEYS}, ensure_ascii=False
                )
            except Exception as exc:  # noqa: BLE001 — один упавший файл не должен ронять батч
                if isinstance(exc, ImportError):
                    reason = f"модуль ориентиров бедра недоступен: {exc}"
                else:
                    reason = f"чекер бедра упал: {type(exc).__name__}: {exc}"
                print(
                    f"ПРЕДУПРЕЖДЕНИЕ: чекеры бедра упали на {row.relative_path}: {reason}",
                    flush=True,
                )
                columns = {}
                for key in HIP_CHECKER_KEYS:
                    columns[f"{key}_score"] = float("nan")
                    columns[f"{key}_flag"] = False
                    columns[f"{key}_status"] = STATUS_NOT_EVALUATED
                signals_json = _json.dumps(
                    {
                        key: {
                            "checker": key,
                            "status": STATUS_NOT_EVALUATED,
                            "score": None,
                            "flag": False,
                            "reason": reason,
                            "signals": {},
                        }
                        for key in HIP_CHECKER_KEYS
                    },
                    ensure_ascii=False,
                )
            cache[row.pixel_hash] = {"columns": columns, "signals": signals_json}
        cached = cache[row.pixel_hash]
        for column, value in cached["columns"].items():
            frame.loc[row.Index, column] = value
        frame.loc[row.Index, "hip_signals"] = cached["signals"]
    return frame


HIP_SIDE_METHOD_PAIRED = "paired"  # две проекции в папке исследования
HIP_SIDE_METHOD_ABSOLUTE = "absolute"  # папка исследования, но проекций не две
HIP_SIDE_METHOD_CONTENT_ONLY = "content_only"  # плоский каталог: пары не видно

FLAT_STUDY_KEY = "."


def _assign_sides(frame: pd.DataFrame) -> pd.DataFrame:
    """Сторона бедра.

    В сабмите сторона не публикуется (раздел 1.2: «сторона бедра в выходе не
    различается»), но это НЕ значит, что ошибка здесь безобидна: `hip_side`/
    `hip_side_final` управляет зеркалированием кадра для CNN бедра
    (`hip_export.preprocess_for_cnn`), а значит влияет на `quality_prob`/
    `quality_class` бедра в сабмите — просто сама колонка стороны наружу не
    идёт. Парное разведение (снимок с большим score — правое бедро)
    допустимо только внутри папки исследования: там два бедра гарантированно
    принадлежат одному пациенту. В плоском каталоге (`study_folder == "."`)
    два снимка бёдер могут быть от разных пациентов, поэтому пары не
    строятся — работает абсолютный знак признака по каждому снимку, а метод
    честно пишется как `content_only`.
    """
    frame = frame.copy()
    frame["hip_side"] = pd.NA
    frame["hip_side_method"] = pd.NA
    hips = frame[(frame["region"] == cfg.REGION_HIP) & frame["dedup_group_id"].notna()]
    for study, chunk in hips.groupby("study_folder", sort=True):
        groups = chunk.groupby("dedup_group_id", sort=True)["hip_side_score"].first().sort_index()
        scores = list(groups.values)
        if study == FLAT_STUDY_KEY:
            sides = [_absolute_side(score) for score in scores]
            method = HIP_SIDE_METHOD_CONTENT_ONLY
        else:
            sides = resolve_sides_within_study(scores)
            method = HIP_SIDE_METHOD_PAIRED if len(groups) == 2 else HIP_SIDE_METHOD_ABSOLUTE
        mapping = dict(zip(groups.index, sides))
        mask = (frame["study_folder"] == study) & frame["dedup_group_id"].isin(mapping)
        frame.loc[mask, "hip_side"] = frame.loc[mask, "dedup_group_id"].map(mapping)
        frame.loc[mask, "hip_side_method"] = method
    return frame


def _absolute_side(score: float) -> str | None:
    if score != score:  # NaN
        return None
    return SIDE_RIGHT if score > 0 else SIDE_LEFT
