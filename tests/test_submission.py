"""Выходной формат сабмита и конвенция полярности (1 = нарушение).

Строки словаря здесь выписаны ВТОРОЙ раз, вручную из ТЗ: смысл теста — поймать
опечатку в `config`, а не сверить конфиг сам с собой.
"""

from __future__ import annotations

import pandas as pd
import pytest

from columba import config as cfg
from columba.submission import (
    SUBMISSION_COLUMNS,
    SubmissionError,
    build_submission,
    region_to_output,
    validate_submission,
)
from columba.targets import violation_string

from conftest import requires_data

REGION_SPINE_TEXT = "Поясничный отдел позвоночника"
REGION_HIP_TEXT = "Проксимальный отдел бедра"
TYPE_POSITIONING = "Некорректная укладка"
TYPE_AXIS = "Не выровнена ось позвоночника"
TYPE_OBJECTS = "Присутствуют посторонние предметы"
TYPE_ROI = "Некорректная область интереса"


def test_region_strings_char_for_char():
    assert cfg.REGION_OUTPUT_NAMES[cfg.REGION_SPINE] == REGION_SPINE_TEXT
    assert cfg.REGION_OUTPUT_NAMES[cfg.REGION_HIP] == REGION_HIP_TEXT
    assert len(cfg.REGION_OUTPUT_NAMES) == 2
    assert region_to_output(cfg.REGION_SPINE) == REGION_SPINE_TEXT
    # Регион вне эвристики обязан получить одну из двух строк, а не пропуск.
    assert region_to_output(cfg.REGION_UNKNOWN) in {REGION_SPINE_TEXT, REGION_HIP_TEXT}


def test_violation_strings_char_for_char():
    assert cfg.VIOLATION_TYPES_BY_REGION[cfg.REGION_SPINE] == (
        TYPE_POSITIONING,
        TYPE_AXIS,
        TYPE_OBJECTS,
    )
    assert cfg.VIOLATION_TYPES_BY_REGION[cfg.REGION_HIP] == (TYPE_POSITIONING, TYPE_ROI)
    assert set(cfg.VIOLATION_TYPES) == {TYPE_POSITIONING, TYPE_AXIS, TYPE_OBJECTS, TYPE_ROI}
    assert cfg.VIOLATION_TYPE_SEPARATOR == ";"


def test_label_polarity_convention():
    """1 = нарушение, 0 = качественное исследование. Перепутать нельзя."""
    assert cfg.QUALITY_CLASS_VIOLATION == 1
    assert cfg.QUALITY_CLASS_OK == 0


def test_violation_string_joins_and_deduplicates():
    assert violation_string([]) == ""
    assert violation_string(["spine_axis"]) == TYPE_AXIS
    assert violation_string(["spine_positioning", "spine_objects"]) == f"{TYPE_POSITIONING};{TYPE_OBJECTS}"
    # Укладка позвоночника и укладка бедра — одна строка, дублировать нельзя.
    assert violation_string(["spine_positioning", "hip_positioning"]) == TYPE_POSITIONING


def _row(**overrides) -> pd.DataFrame:
    base = {
        "file_name": "CR000000.dcm",
        "anatomical_region": REGION_HIP_TEXT,
        "quality_class": 0,
        "quality_prob": 0.1,
        "violation_type": "",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_validator_accepts_a_correct_row():
    validate_submission(_row(quality_class=1, quality_prob=0.9, violation_type=TYPE_ROI))


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"quality_class": 1, "violation_type": ""}, "пустое"),
        ({"quality_class": 0, "violation_type": TYPE_ROI}, "конвенция"),
        ({"quality_class": 2, "violation_type": ""}, "0/1"),
        ({"quality_prob": 1.4}, "[0; 1]"),
        ({"anatomical_region": "бедро"}, "вне словаря"),
        ({"anatomical_region": None}, "не заполнен"),
        ({"quality_class": 1, "violation_type": TYPE_AXIS}, "вне словаря региона"),
        ({"quality_class": 1, "violation_type": f"{TYPE_ROI};{TYPE_ROI}"}, "повторяющиеся"),
    ],
)
def test_validator_rejects_broken_rows(overrides, fragment):
    with pytest.raises(SubmissionError) as error:
        validate_submission(_row(**overrides))
    assert fragment in str(error.value)


def test_validator_requires_a_row_per_file():
    with pytest.raises(SubmissionError, match="дубликаты"):
        validate_submission(_row(), expected_rows=2)


def test_validator_rejects_violation_class_below_threshold():
    """quality_class=1 обязан сопровождаться quality_prob >= 0.5 (задача 9.3):
    иначе ROC-AUC по вероятности и Macro-F1 по классу измеряют разные решения."""
    bad = _row(quality_class=1, quality_prob=0.2, violation_type=TYPE_ROI)
    with pytest.raises(SubmissionError, match="0.5") as error:
        validate_submission(bad)
    assert "CR000000.dcm" in str(error.value)


def test_validator_rejects_ok_class_above_threshold():
    bad = _row(quality_class=0, quality_prob=0.9, violation_type="")
    with pytest.raises(SubmissionError, match="0.5"):
        validate_submission(bad)


def test_validator_accepts_threshold_boundary_consistent_with_class():
    validate_submission(_row(quality_class=1, quality_prob=0.5, violation_type=TYPE_ROI))
    validate_submission(_row(quality_class=0, quality_prob=0.4999, violation_type=""))


@requires_data
def test_reference_submission_has_one_row_per_file(stage0, manifest):
    submission = stage0["submission_reference"]
    assert len(submission) == len(manifest) == cfg.EXPECTED_TRAIN_FILES
    assert list(submission.columns)[: len(SUBMISSION_COLUMNS)] == list(SUBMISSION_COLUMNS)
    validate_submission(submission, expected_rows=len(manifest))


@requires_data
def test_reference_submission_keeps_duplicates(stage0, manifest):
    """Дубликаты дедуплицируются для обучения, но не для вывода."""
    duplicated = manifest[manifest["dedup_group_size"] > 1]
    assert len(duplicated) > 0
    submission = stage0["submission_reference"]
    assert int(submission["dedup_group_id"].isin(duplicated["dedup_group_id"]).sum()) == len(duplicated)


@requires_data
def test_reference_submission_matches_targets(stage0, targets):
    """Полярность и типы в сабмите совпадают с таргетами уровня изображения."""
    submission = stage0["submission_reference"].set_index("dedup_group_id")
    annotated = targets[targets["has_target"]]
    for row in annotated.itertuples():
        rows = submission.loc[[row.dedup_group_id]]
        assert set(rows["quality_class"]) == {int(row.quality_class)}
        expected = "" if pd.isna(row.violation_type) else row.violation_type
        assert set(rows["violation_type"]) == {expected}


@requires_data
def test_every_submission_region_is_one_of_two(stage0):
    regions = set(stage0["submission_reference"]["anatomical_region"])
    assert regions == {REGION_SPINE_TEXT, REGION_HIP_TEXT}
