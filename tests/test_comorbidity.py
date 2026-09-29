"""Бонус 3: классификация врачебных комментариев по ключевым словам.

Никакой связи с инференсом/`/predict` — тестируется только офлайн-отчёт.
"""

from __future__ import annotations

import pytest

from columba.comorbidity import classify_comment, build_comorbidity_table, write_comorbidity_report

from conftest import requires_data

# Все 10 уникальных строк комментариев, реально встретившихся в
# `markup.load_markup()` на train-выгрузке (см. stages/stage_9.md,
# «Бонусы 3–5») — регрессия на изменение регулярных выражений.
KNOWN_COMMENTS = [
    ("L6, люмбализация", {"анатомический вариант (люмбализация)"}),
    (
        "Отклонение оси, нет малых вертелов",
        {"отклонение оси (заметка)", "отсутствие малого вертела (заметка)"},
    ),
    ("Сколиоз, не верня разметка", {"сколиоз", "пометка о некорректной разметке"}),
    ("Требует внимание", {"требует внимания (без уточнения)"}),
    (
        "Хороший пример отклонения оси при КТ п-ка",
        {"отклонение оси (заметка)", "учебный пример"},
    ),
    ("Хороший пример ротации", {"ротация (заметка)", "учебный пример"}),
    ("перелом", {"перелом"}),
    ("правое бедро эндопротезирования", {"эндопротезирование"}),
    ("сколиоз", {"сколиоз"}),
    ("эндопротезирования ТБС", {"эндопротезирование"}),
]


@pytest.mark.parametrize("comment,expected", KNOWN_COMMENTS)
def test_classify_comment_matches_expected_categories(comment, expected):
    assert set(classify_comment(comment)) == expected


@pytest.mark.parametrize("comment", [None, "", "   ", "снимок без особенностей"])
def test_classify_comment_empty_or_unmatched_returns_empty_list(comment):
    assert classify_comment(comment) == []


@requires_data
def test_build_comorbidity_table_has_one_row_per_commented_image(manifest):
    table = build_comorbidity_table(manifest)
    assert len(table) > 0
    # Каждая строка — представитель dedup-группы с непустым комментарием.
    assert (table["markup_comment"].str.strip() != "").all()
    # Категории посчитаны для каждой строки (список, возможно пустой, но не NaN).
    assert table["categories"].apply(lambda c: isinstance(c, list)).all()
    # Ни одна строка с известным комментарием не осталась без категории —
    # полное покрытие CATEGORY_PATTERNS на реальных данных.
    uncategorized = table[table["categories"].apply(len) == 0]
    assert len(uncategorized) == 0, uncategorized["markup_comment"].tolist()


@requires_data
def test_write_comorbidity_report_creates_readable_markdown(manifest, tmp_path):
    output = write_comorbidity_report(manifest, output=tmp_path / "comorbidities.md")
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "сопутствующие патологии" in text.lower()
    assert "сколиоз" in text
    assert "категория" in text.lower()
