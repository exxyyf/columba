"""Бонус 3 (этап 8, приоритет 3): сопутствующие патологии — по ключевым
словам из врачебных комментариев таблицы «Калибровка» (`markup.py`, шаг 5).

**Офлайн EDA-отчёт, НЕ подключён к инференсу/`/predict`.** Комментарии
врача существуют только для размеченной train-выборки
(`markup.load_markup()`); на закрытом тесте такого столбца нет и не будет
(вход закрытого теста — только DICOM-файлы) — реализовать это как живой
чекер, работающий на закрытом тесте, физически нечем.

Классификация — сопоставление по регулярным выражениям
(`CATEGORY_PATTERNS`), подобранным под 10 реально встретившихся уникальных
строк комментариев (см. тест `test_comorbidity.py`), НЕ NLP/ML-модель:
обучать классификатор категорий патологий не на чем — непустой комментарий
есть только у 23 исследований из 100 в таблице «Калибровка», отдельной
разметки категорий (сколиоз/перелом/эндопротез/...) не существует.

Запуск: `uv run python -m columba.comorbidity` -> `artifacts/eda/comorbidities.md`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import config as cfg
from .inventory import load_manifest

# Подобраны под реально встретившиеся в `markup.load_markup()` строки —
# не общий NLP-парсер медицинских заключений. Один комментарий может
# получить несколько категорий (например, "Сколиоз, не верня разметка" ->
# сколиоз + пометка о некорректной разметке).
CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "сколиоз": re.compile(r"сколиоз", re.IGNORECASE),
    "перелом": re.compile(r"перелом", re.IGNORECASE),
    "эндопротезирование": re.compile(r"эндопротез", re.IGNORECASE),
    "анатомический вариант (люмбализация)": re.compile(r"люмбализ", re.IGNORECASE),
    "отклонение оси (заметка)": re.compile(r"отклонени\w*\s+ос", re.IGNORECASE),
    "ротация (заметка)": re.compile(r"ротаци", re.IGNORECASE),
    "отсутствие малого вертела (заметка)": re.compile(r"нет\s+малых?\s+вертел", re.IGNORECASE),
    "пометка о некорректной разметке": re.compile(r"не\s*\w*\s+разметк", re.IGNORECASE),
    "требует внимания (без уточнения)": re.compile(r"требует\s+внимани", re.IGNORECASE),
    "учебный пример": re.compile(r"хороший\s+пример", re.IGNORECASE),
}


def classify_comment(comment: str | None) -> list[str]:
    """Комментарий врача -> список совпавших категорий (пусто = без категории)."""
    if comment is None or not str(comment).strip():
        return []
    text = str(comment)
    return [name for name, pattern in CATEGORY_PATTERNS.items() if pattern.search(text)]


def build_comorbidity_table(manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Одна строка на dedup-группу (уникальное изображение) с непустым
    комментарием врача, плюс совпавшие категории (`classify_comment`).

    `manifest["markup_comment"]` уже присоединён по `study_folder`
    (`targets.build_targets`/`pipeline.py`, шаг 6) — здесь второй раз этот
    джойн не делается, только фильтрация и классификация.
    """
    if manifest is None:
        manifest = load_manifest()
    representatives = manifest[manifest["is_group_representative"].fillna(False)].copy()
    comment = representatives["markup_comment"]
    has_comment = comment.notna() & (comment.astype(str).str.strip() != "")
    rows = representatives.loc[has_comment].copy()
    rows["categories"] = rows["markup_comment"].map(classify_comment)
    return rows[["dedup_group_id", "study_folder", "region", "quality_class", "markup_comment", "categories"]].reset_index(
        drop=True
    )


def write_comorbidity_report(
    manifest: pd.DataFrame | None = None,
    output: Path | str = cfg.EDA_DIR / "comorbidities.md",
) -> Path:
    if manifest is None:
        manifest = load_manifest()
    table = build_comorbidity_table(manifest)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # `reset_index`: `explode` duplicates the original row index for a
    # multi-category comment (e.g. "сколиоз" + "пометка о некорректной
    # разметке" из одной строки) — `pd.crosstab` ниже не принимает индекс с
    # повторами при выравнивании.
    exploded = table.explode("categories").reset_index(drop=True)
    n_representatives = int(manifest["is_group_representative"].fillna(False).sum())

    lines = [
        "# Бонус 3 — сопутствующие патологии по комментариям врача",
        "",
        "Офлайн-отчёт (не часть инференса/`/predict`: вход закрытого теста —",
        "только DICOM-файлы, без врачебных комментариев). Категории — по",
        "ключевым словам (`comorbidity.CATEGORY_PATTERNS`) на 10 реально",
        "встретившихся уникальных строк комментариев, не NLP/ML-модель.",
        "",
        f"- уникальных изображений с комментарием: **{len(table)}** из "
        f"**{n_representatives}** представителей dedup-групп",
        f"- исследований с комментарием: **{table['study_folder'].nunique()}**",
        "",
        "## Категории",
        "",
        "| категория | изображений |",
        "| --- | --- |",
    ]
    category_counts = exploded["categories"].value_counts()
    lines += [f"| {cat} | {int(n)} |" for cat, n in category_counts.items()]

    lines += ["", "## Категория × quality_class", ""]
    crosstab = pd.crosstab(exploded["categories"], exploded["quality_class"].fillna(-1).astype(int))
    class_labels = {-1: "н/д", 0: "0 (норма)", 1: "1 (нарушение)"}
    header_cols = [class_labels.get(c, str(c)) for c in crosstab.columns]
    lines.append("| категория | " + " | ".join(header_cols) + " |")
    lines.append("| " + " --- |" * (len(crosstab.columns) + 1))
    for cat, row in crosstab.iterrows():
        lines.append(f"| {cat} | " + " | ".join(str(int(v)) for v in row) + " |")

    lines += [
        "",
        "## Все строки",
        "",
        "| dedup_group_id | регион | quality_class | комментарий | категории |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in table.itertuples():
        categories = ", ".join(row.categories) if row.categories else "—"
        quality_class = "н/д" if pd.isna(row.quality_class) else str(int(row.quality_class))
        lines.append(f"| {row.dedup_group_id} | {row.region} | {quality_class} | {row.markup_comment} | {categories} |")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> int:
    path = write_comorbidity_report()
    print(f"отчёт: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
