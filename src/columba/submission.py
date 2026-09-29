"""Выходная таблица (раздел 1.2) и её валидация.

Инвариант этапа 0: строка на КАЖДЫЙ пришедший файл, включая попиксельные
дубликаты и нечитаемые файлы. Дедупликация нужна для обучения и сплита,
но не для инференса.

Конвенция направления меток (раздел 1.4) зафиксирована в `config`:
`quality_class = 1` и `quality_prob` — это НАРУШЕНИЕ, а не качество.
Перепутанная полярность обнуляет обе метрики, поэтому здесь она проверяется
явно, а строки словаря сверяются посимвольно.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as cfg
from .dicom_io import STATUS_SUCCESS

SUBMISSION_COLUMNS: tuple[str, ...] = (
    "file_name",
    "anatomical_region",
    "quality_class",
    "quality_prob",
    "violation_type",
)

# Колонки сверх формата автопроверке не мешают и нужны для разбора полётов.
# `sop_instance_uid` — задача 9.3: дополнительный идентификатор из тега
# DICOM (не входит в SUBMISSION_COLUMNS), присутствует только когда манифест
# его считал (`inference.describe_inputs`); отсутствие колонки в манифесте
# не ошибка — `build_submission` фильтрует DEBUG_COLUMNS по наличию.
DEBUG_COLUMNS: tuple[str, ...] = (
    "relative_path",
    "study_folder",
    "read_status",
    "dedup_group_id",
    "sop_instance_uid",
)


class SubmissionError(ValueError):
    """Сабмит нарушает формат раздела 1.2 или конвенцию раздела 1.4."""


def region_to_output(region: str) -> str:
    """Внутренний регион -> посимвольная строка из словаря организаторов.

    В выходе допустимы ровно две строки, поэтому `unknown` уходит в
    `config.REGION_SUBMISSION_FALLBACK`. В манифесте `unknown` при этом
    сохраняется — это честный сигнал для разбора.
    """
    if region not in cfg.REGION_OUTPUT_NAMES:
        region = cfg.REGION_SUBMISSION_FALLBACK
    return cfg.REGION_OUTPUT_NAMES[region]


def build_submission(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    *,
    include_debug_columns: bool = True,
) -> pd.DataFrame:
    """Собрать выходную таблицу.

    `manifest` — файловый уровень (по строке на файл). `predictions` — таблица
    с `dedup_group_id` (или `file_id`) и колонками `quality_prob`,
    `quality_class`, `violation_type`. Без предсказаний возвращается
    корректный по формату «нулевой» сабмит: он нужен, чтобы проверить формат
    и полярность до появления модели.
    """
    frame = manifest.copy()
    frame["anatomical_region"] = frame["region"].map(region_to_output)

    frame["quality_class"] = cfg.QUALITY_CLASS_OK
    frame["quality_prob"] = 0.0
    frame["violation_type"] = ""

    failures = frame["read_status"] != STATUS_SUCCESS
    frame.loc[failures, "quality_class"] = cfg.FAILURE_ROW_QUALITY_CLASS
    frame.loc[failures, "quality_prob"] = cfg.FAILURE_ROW_QUALITY_PROB
    frame.loc[failures, "violation_type"] = ""

    if predictions is not None:
        key = "file_id" if "file_id" in predictions.columns else "dedup_group_id"
        merged = frame.merge(
            predictions[[key, "quality_class", "quality_prob", "violation_type"]],
            on=key,
            how="left",
            suffixes=("", "_pred"),
        )
        for column in ("quality_class", "quality_prob", "violation_type"):
            predicted = merged[f"{column}_pred"]
            merged[column] = predicted.where(predicted.notna(), merged[column])
            merged.drop(columns=[f"{column}_pred"], inplace=True)
        frame = merged

    frame["quality_class"] = frame["quality_class"].astype(int)
    frame["quality_prob"] = frame["quality_prob"].astype(float)
    frame["violation_type"] = frame["violation_type"].fillna("").astype(str)

    columns = list(SUBMISSION_COLUMNS)
    if include_debug_columns:
        columns += [c for c in DEBUG_COLUMNS if c in frame.columns]
    return frame.reindex(columns=columns)


def validate_submission(submission: pd.DataFrame, *, expected_rows: int | None = None) -> None:
    """Проверить формат и конвенцию. Бросает `SubmissionError` при нарушении."""
    missing = [c for c in SUBMISSION_COLUMNS if c not in submission.columns]
    if missing:
        raise SubmissionError(f"нет обязательных колонок: {missing}")

    if expected_rows is not None and len(submission) != expected_rows:
        raise SubmissionError(
            f"строк {len(submission)}, а файлов на входе {expected_rows}: "
            "нужна строка на каждый файл, включая попиксельные дубликаты"
        )

    if submission["anatomical_region"].isna().any():
        raise SubmissionError("anatomical_region не заполнен — допустимы ровно две строки словаря")
    regions = set(submission["anatomical_region"].unique())
    allowed_regions = set(cfg.REGION_OUTPUT_NAMES.values())
    if not regions <= allowed_regions:
        raise SubmissionError(f"anatomical_region вне словаря: {sorted(regions - allowed_regions)}")

    classes = set(submission["quality_class"].unique())
    if not classes <= {cfg.QUALITY_CLASS_OK, cfg.QUALITY_CLASS_VIOLATION}:
        raise SubmissionError(f"quality_class должен быть 0/1, получено {sorted(classes)}")

    probs = submission["quality_prob"].astype(float)
    if probs.isna().any() or float(probs.min()) < 0.0 or float(probs.max()) > 1.0:
        raise SubmissionError("quality_prob должен лежать в [0; 1] без пропусков")

    for row in submission.itertuples():
        text = "" if pd.isna(row.violation_type) else str(row.violation_type)
        types = [t for t in text.split(cfg.VIOLATION_TYPE_SEPARATOR) if t != ""]

        if types and row.quality_class != cfg.QUALITY_CLASS_VIOLATION:
            raise SubmissionError(
                f"{row.file_name}: перечислены нарушения при quality_class="
                f"{row.quality_class} — нарушена конвенция 1 = нарушение"
            )
        if not types and row.quality_class != cfg.QUALITY_CLASS_OK:
            raise SubmissionError(
                f"{row.file_name}: quality_class={row.quality_class}, но поле violation_type пустое"
            )

        allowed = _allowed_types(row.anatomical_region)
        unknown = [t for t in types if t not in allowed]
        if unknown:
            raise SubmissionError(
                f"{row.file_name}: типы вне словаря региона «{row.anatomical_region}»: {unknown}"
            )
        if len(set(types)) != len(types):
            raise SubmissionError(f"{row.file_name}: повторяющиеся типы нарушения")

    # Построчная сверка конвенции 1.4: `quality_class` обязан быть порогом
    # 0.5 по `quality_prob` (аггрегатор этапа 3 именно так и строит пару —
    # см. `aggregate.py`, комментарий у `_spine_row_prediction`), иначе
    # ROC-AUC по вероятности и метрика по классу измеряют РАЗНЫЕ решения.
    # Проверяется последней: остальные проверки этой функции диагностируют
    # свою причину точнее (например, пустой `violation_type` при
    # `quality_class=1`), эта — общая подстраховка после них.
    mismatched = (submission["quality_class"] == cfg.QUALITY_CLASS_VIOLATION) != (probs >= 0.5)
    if mismatched.any():
        bad_files = submission.loc[mismatched, "file_name"].astype(str).tolist()
        preview = bad_files[:5]
        suffix = "…" if len(bad_files) > 5 else ""
        raise SubmissionError(
            "quality_class не соответствует порогу 0.5 по quality_prob "
            f"(конвенция 1.4) на {len(bad_files)} строк(и): {preview}{suffix}"
        )


def _allowed_types(anatomical_region) -> tuple[str, ...]:
    for region, name in cfg.REGION_OUTPUT_NAMES.items():
        if name == anatomical_region:
            return cfg.VIOLATION_TYPES_BY_REGION[region]
    return ()


def save_submission(submission: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False, encoding="utf-8")
    return path
