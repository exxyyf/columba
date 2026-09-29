"""Чтение таблицы «Калибровка» (шаг 5) — только 7 критериев и служебные поля.

Колонки блока «Итог» и агрегаты «Общий» читаются ОТДЕЛЬНО и только для
логирования расхождений: таргеты из них не строятся (шаг 6).
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd

from .config import (
    CRITERIA,
    MARKUP_COMMENT_COLUMN,
    MARKUP_HEADER_ROWS,
    MARKUP_INDEX_COLUMN,
    MARKUP_SHEET,
    MARKUP_STUDY_COLUMN,
    MARKUP_XLSX,
)

# Колонки «Итог» по зонам — нужны только для аудита расхождений.
LEGACY_TOTAL_COLUMNS = {"legacy_total_spine": 9, "legacy_total_hip_right": 10, "legacy_total_hip_left": 11}


def load_markup(xlsx_path: Path | str = MARKUP_XLSX, sheet: str = MARKUP_SHEET) -> pd.DataFrame:
    """Вернуть таблицу разметки: одна строка на исследование."""
    workbook = openpyxl.load_workbook(Path(xlsx_path), data_only=True, read_only=True)
    worksheet = workbook[sheet]
    raw_rows = list(worksheet.iter_rows(min_row=MARKUP_HEADER_ROWS + 1, values_only=True))
    workbook.close()

    records = []
    for offset, row in enumerate(raw_rows):
        study = row[MARKUP_STUDY_COLUMN] if len(row) > MARKUP_STUDY_COLUMN else None
        if study is None or not str(study).strip():
            continue
        record = {
            "markup_row": MARKUP_HEADER_ROWS + 1 + offset,
            "markup_index": row[MARKUP_INDEX_COLUMN],
            "study_folder": str(study).strip(),
            "comment": _clean(row[MARKUP_COMMENT_COLUMN] if len(row) > MARKUP_COMMENT_COLUMN else None),
        }
        for criterion in CRITERIA:
            record[criterion.key] = _binary(row[criterion.column] if len(row) > criterion.column else None)
        for name, column in LEGACY_TOTAL_COLUMNS.items():
            record[name] = _binary(row[column] if len(row) > column else None)
        records.append(record)

    frame = pd.DataFrame(records)
    for criterion in CRITERIA:
        frame[criterion.key] = frame[criterion.key].astype("Int64")
    for name in LEGACY_TOTAL_COLUMNS:
        frame[name] = frame[name].astype("Int64")
    return frame


def _binary(value) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return 1 if number != 0 else 0


def _clean(value) -> str:
    return "" if value is None else str(value).strip()
