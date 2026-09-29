"""Шаг 6: перегенерация таргетов из критериев."""

from __future__ import annotations

import pandas as pd
import pytest

from columba import config as cfg
from columba.targets import (
    CRITERION_KEYS,
    LABEL_COLUMNS,
    LABEL_PREFIX,
    build_targets,
    check_no_violation_without_criteria,
)

from conftest import requires_data


@requires_data
def test_no_quality_class_without_positive_criterion(targets):
    """Главная проверка шага 6: 1 не появляется из воздуха."""
    assert len(check_no_violation_without_criteria(targets)) == 0


@requires_data
def test_quality_class_is_or_over_own_zone(targets):
    annotated = targets[targets["has_target"]]
    for row in annotated.itertuples():
        zone_criteria = cfg.CRITERIA_BY_ZONE[row.zone_key]
        values = [getattr(row, c.key) for c in zone_criteria]
        present = [int(v) for v in values if pd.notna(v)]
        assert present, f"{row.dedup_group_id}: зона помечена размеченной, но критериев нет"
        assert int(row.quality_class) == int(any(present))


@requires_data
def test_missing_criteria_never_become_zeros(targets, markup):
    """Пропуск = зоны нет у пациента. Такая ячейка не участвует в OR."""
    unannotated = targets[~targets["has_target"]]
    assert len(unannotated) == 3  # 2 кадра эндопротеза + правое бедро эндопротезированного
    for row in unannotated.itertuples():
        assert pd.isna(row.quality_class)
        for key in CRITERION_KEYS:
            assert pd.isna(getattr(row, key))
        for column in LABEL_COLUMNS:
            assert pd.isna(getattr(row, column))

    # Число размеченных зон совпадает с числом заполненных ячеек в таблице.
    annotated = targets[targets["has_target"]]
    for zone in cfg.ZONES:
        expected_studies = {
            row.study_folder
            for row in markup.itertuples()
            if any(pd.notna(getattr(row, c.key)) for c in cfg.CRITERIA_BY_ZONE[zone])
        }
        got_studies = set(annotated.loc[annotated["zone_key"] == zone, "study_folder"])
        assert got_studies == expected_studies, zone


@requires_data
def test_итог_column_is_not_used(targets, markup):
    """«Итог» игнорируется: расхождения существуют и это ожидаемо."""
    legacy = {
        cfg.ZONE_SPINE: "legacy_total_spine",
        cfg.ZONE_HIP_RIGHT: "legacy_total_hip_right",
        cfg.ZONE_HIP_LEFT: "legacy_total_hip_left",
    }
    lookup = markup.set_index("study_folder")
    mismatches = set()
    for row in targets[targets["has_target"]].itertuples():
        value = lookup.loc[row.study_folder, legacy[row.zone_key]]
        if pd.notna(value) and int(value) != int(row.quality_class):
            mismatches.add((row.study_folder, row.zone_key))
    # Известные помарки: сколиоз в строках 6 и 11, перелом в строке 35.
    assert mismatches == {
        ("2.25.339027809107632348165171236474821772940", cfg.ZONE_SPINE),
        ("2.25.52532476139575422857225834315571570912", cfg.ZONE_SPINE),
        ("2.25.127126130998341190348103034690069487890", cfg.ZONE_SPINE),
    }


@requires_data
def test_multilabel_vector_is_consistent_with_quality_class(targets):
    annotated = targets[targets["has_target"]]
    label_any = annotated[list(LABEL_COLUMNS)].fillna(0).max(axis=1)
    assert (label_any == annotated["quality_class"]).all()


@requires_data
def test_multilabel_has_five_dictionary_rows(targets):
    assert len(cfg.OUTPUT_LABELS) == 5
    assert set(LABEL_COLUMNS) <= set(targets.columns)
    annotated = targets[targets["has_target"]]
    for key in cfg.OUTPUT_LABEL_KEYS:
        column = f"{LABEL_PREFIX}{key}"
        assert annotated[column].notna().all()
        assert set(annotated[column].unique()) <= {0, 1}


@requires_data
def test_labels_are_region_appropriate(targets):
    """Строки словаря чужого региона всегда нули."""
    annotated = targets[targets["has_target"]]
    for region in (cfg.REGION_SPINE, cfg.REGION_HIP):
        subset = annotated[annotated["region"] == region]
        foreign = [
            label.key for label in cfg.OUTPUT_LABELS if label.region != region
        ]
        for key in foreign:
            assert (subset[f"{LABEL_PREFIX}{key}"] == 0).all(), (region, key)


@requires_data
def test_violation_type_field_matches_labels(targets):
    """Поле сабмита собирается ровно из активных строк словаря."""
    annotated = targets[targets["has_target"]]
    for row in annotated.itertuples():
        active = [
            key for key in cfg.OUTPUT_LABEL_KEYS if getattr(row, f"{LABEL_PREFIX}{key}") == 1
        ]
        text = "" if pd.isna(row.violation_type) else str(row.violation_type)
        parts = [t for t in text.split(cfg.VIOLATION_TYPE_SEPARATOR) if t]
        assert parts == list(dict.fromkeys(cfg.OUTPUT_LABELS_BY_KEY[k].violation_type for k in active))
        assert bool(parts) == (int(row.quality_class) == cfg.QUALITY_CLASS_VIOLATION)


def test_hip_rotation_and_spine_positioning_share_one_dictionary_row():
    """Организаторы объединили ротацию бедра с «Некорректной укладкой»."""
    spine = cfg.OUTPUT_LABELS_BY_KEY["spine_positioning"].violation_type
    hip = cfg.OUTPUT_LABELS_BY_KEY["hip_positioning"].violation_type
    assert spine == hip == cfg.VIOLATION_POSITIONING
    # 7 критериев -> 5 строк словаря -> 4 уникальные строки.
    assert len(cfg.CRITERIA) == 7
    assert len(cfg.OUTPUT_LABELS) == 5
    assert len({label.violation_type for label in cfg.OUTPUT_LABELS}) == 4


@requires_data
def test_targets_are_one_row_per_unique_image(targets):
    assert len(targets) == cfg.EXPECTED_DEDUP_GROUPS
    assert targets["dedup_group_id"].is_unique


# --------------------------------------------------------------------------- #
# Синтетика: dedup-группа на два исследования (этап 9, п. 9.1)
# --------------------------------------------------------------------------- #


def _synthetic_manifest_markup():
    """Одна dedup-группа с файлами из двух РАЗНЫХ исследований, оба с зоной
    позвоночника — зона однозначна, спорна только принадлежность study."""
    manifest = pd.DataFrame(
        {
            "dedup_group_id": ["g0001", "g0001"],
            "study_folder": ["study_a", "study_b"],
            "zone_key": [cfg.ZONE_SPINE, cfg.ZONE_SPINE],
            "region": [cfg.REGION_SPINE, cfg.REGION_SPINE],
            "hip_side": [pd.NA, pd.NA],
            "rows": [300, 300],
            "cols": [300, 300],
        }
    )
    # study_a размечен как нарушение, study_b — как норма: если бы таргет
    # брался из studies[0] («study_a»), quality_class ошибочно оказался бы 1.
    markup = pd.DataFrame(
        {
            "study_folder": ["study_a", "study_b"],
            "spine_positioning": [1, 0],
            "spine_axis": [0, 0],
            "spine_artifacts": [0, 0],
            "hip_right_rotation": [pd.NA, pd.NA],
            "hip_right_roi": [pd.NA, pd.NA],
            "hip_left_rotation": [pd.NA, pd.NA],
            "hip_left_roi": [pd.NA, pd.NA],
            "comment": ["", ""],
        }
    )
    return manifest, markup


def test_dedup_group_spanning_studies_gets_no_target():
    """Группа на два исследования не наследует таргет studies[0] —
    уходит в «без таргета», как и dedup_group_ambiguous_zone."""
    manifest, markup = _synthetic_manifest_markup()
    targets, anomalies = build_targets(manifest, markup)

    assert len(targets) == 1
    row = targets.iloc[0]
    assert row["has_target"] == False  # noqa: E712 — явно bool, не truthy-проверка
    assert pd.isna(row["quality_class"])
    assert pd.isna(row["zone_key"])
    for key in CRITERION_KEYS:
        assert pd.isna(row[key])
    for column in LABEL_COLUMNS:
        assert pd.isna(row[column])

    kinds = [a["kind"] for a in anomalies]
    assert "dedup_group_spans_studies" in kinds
