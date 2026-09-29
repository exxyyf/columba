"""Тесты метрик этапа 3 (шаг 8): на синтетике + sanity на реальной разметке."""

from __future__ import annotations

import pandas as pd
import pytest

from columba import config as cfg
from columba import stage3_eval as ev
from conftest import requires_data


def _linked(rows: list[dict]) -> pd.DataFrame:
    """Собрать синтетический `linked`-фрейм (колонки ev.LINKED_COLUMNS)."""
    defaults = {
        "split": "train",
        "region": cfg.REGION_SPINE,
        "quality_class_true": 0,
        "quality_class_pred": 0,
        "quality_prob_pred": 0.0,
        "violation_type_true": "",
        "violation_type_pred": "",
    }
    full_rows = []
    for i, row in enumerate(rows):
        merged = dict(defaults)
        merged["dedup_group_id"] = f"g{i:04d}"
        merged["study_folder"] = f"s{i:04d}"
        merged.update(row)
        full_rows.append(merged)
    return pd.DataFrame(full_rows, columns=list(ev.LINKED_COLUMNS))


# --------------------------------------------------------------------------- #
# Метрики на синтетике
# --------------------------------------------------------------------------- #


def test_binary_f1_and_auc_perfect_prediction():
    frame = _linked(
        [
            {"quality_class_true": 1, "quality_class_pred": 1, "quality_prob_pred": 0.9},
            {"quality_class_true": 1, "quality_class_pred": 1, "quality_prob_pred": 0.8},
            {"quality_class_true": 0, "quality_class_pred": 0, "quality_prob_pred": 0.1},
            {"quality_class_true": 0, "quality_class_pred": 0, "quality_prob_pred": 0.2},
        ]
    )
    metrics = ev.compute_metrics(frame, bootstrap_n=50)
    assert metrics["binary_f1"]["value"] == 1.0
    assert metrics["binary_auc"]["value"] == 1.0


def test_auc_ties_gives_half_credit():
    # Скор совпадает у позитива и негатива -> ранговая формула даёт 0.5 на паре.
    frame = _linked(
        [
            {"quality_class_true": 1, "quality_class_pred": 1, "quality_prob_pred": 0.5},
            {"quality_class_true": 0, "quality_class_pred": 0, "quality_prob_pred": 0.5},
        ]
    )
    metrics = ev.compute_metrics(frame, bootstrap_n=50)
    assert metrics["binary_auc"]["value"] == 0.5


def test_binary_f1_nan_without_true_positives():
    frame = _linked(
        [
            {"quality_class_true": 0, "quality_class_pred": 0, "quality_prob_pred": 0.1},
            {"quality_class_true": 0, "quality_class_pred": 1, "quality_prob_pred": 0.6},
        ]
    )
    metrics = ev.compute_metrics(frame, bootstrap_n=50)
    assert metrics["binary_f1"]["value"] != metrics["binary_f1"]["value"]  # NaN


def test_type_f1_nan_when_type_has_no_positives_in_slice():
    # Ни у кого нет "Присутствуют посторонние предметы" -> F1 этого типа NaN.
    frame = _linked(
        [
            {
                "quality_class_true": 1,
                "quality_class_pred": 1,
                "violation_type_true": cfg.VIOLATION_POSITIONING,
                "violation_type_pred": cfg.VIOLATION_POSITIONING,
            },
            {"quality_class_true": 0, "quality_class_pred": 0},
        ]
    )
    metrics = ev.compute_metrics(frame, bootstrap_n=50)
    foreign = metrics["by_type"][cfg.VIOLATION_FOREIGN_OBJECTS]["value"]
    assert foreign != foreign  # NaN
    positioning = metrics["by_type"][cfg.VIOLATION_POSITIONING]["value"]
    assert positioning == 1.0


def test_macro_f1_averages_only_defined_types():
    # Ровно два типа из 4 имеют истинные позитивы в срезе (оба предсказаны верно
    # -> F1=1.0); остальные два — NaN и не должны понижать Macro-F1.
    frame = _linked(
        [
            {
                "quality_class_true": 1,
                "quality_class_pred": 1,
                "violation_type_true": f"{cfg.VIOLATION_POSITIONING}{cfg.VIOLATION_TYPE_SEPARATOR}{cfg.VIOLATION_SPINE_AXIS}",
                "violation_type_pred": f"{cfg.VIOLATION_POSITIONING}{cfg.VIOLATION_TYPE_SEPARATOR}{cfg.VIOLATION_SPINE_AXIS}",
            },
            {"quality_class_true": 0, "quality_class_pred": 0},
        ]
    )
    metrics = ev.compute_metrics(frame, bootstrap_n=50)
    assert metrics["macro_f1"]["value"] == 1.0
    assert metrics["by_type"][cfg.VIOLATION_FOREIGN_OBJECTS]["value"] != metrics["by_type"][cfg.VIOLATION_FOREIGN_OBJECTS]["value"]
    assert metrics["by_type"][cfg.VIOLATION_ROI]["value"] != metrics["by_type"][cfg.VIOLATION_ROI]["value"]


def test_type_f1_nan_note_when_false_positives_without_true_positives():
    # Тип "Присутствуют посторонние предметы" предсказан (1 ложное срабатывание),
    # но истинных позитивов в срезе нет -> F1 NaN, но predicted_positives=1 и note не пусто.
    frame = _linked(
        [
            {
                "quality_class_true": 1,
                "quality_class_pred": 1,
                "violation_type_true": cfg.VIOLATION_POSITIONING,
                "violation_type_pred": f"{cfg.VIOLATION_POSITIONING}{cfg.VIOLATION_TYPE_SEPARATOR}{cfg.VIOLATION_FOREIGN_OBJECTS}",
            },
            {"quality_class_true": 0, "quality_class_pred": 0},
        ]
    )
    metrics = ev.compute_metrics(frame, bootstrap_n=50)
    foreign = metrics["by_type"][cfg.VIOLATION_FOREIGN_OBJECTS]
    assert foreign["value"] != foreign["value"]  # NaN
    assert foreign["predicted_positives"] == 1
    assert foreign["note"] != ""
    positioning = metrics["by_type"][cfg.VIOLATION_POSITIONING]
    assert positioning["note"] == ""


def test_macro_f1_helper_nan_when_all_undefined():
    assert ev.macro_f1([float("nan"), float("nan")]) != ev.macro_f1([float("nan"), float("nan")])


def test_bootstrap_ci_reproducible_by_seed():
    frame = _linked(
        [
            {"quality_class_true": 1, "quality_class_pred": 1, "quality_prob_pred": 0.9, "study_folder": "s1"},
            {"quality_class_true": 1, "quality_class_pred": 0, "quality_prob_pred": 0.4, "study_folder": "s2"},
            {"quality_class_true": 0, "quality_class_pred": 0, "quality_prob_pred": 0.2, "study_folder": "s3"},
            {"quality_class_true": 0, "quality_class_pred": 1, "quality_prob_pred": 0.7, "study_folder": "s4"},
        ]
    )
    def _ci_equal(a, b):
        return all(
            (x != x and y != y) or x == y for x, y in zip(a, b)  # NaN считается равным NaN
        )

    first = ev.compute_metrics(frame, bootstrap_n=200)
    second = ev.compute_metrics(frame, bootstrap_n=200)
    assert _ci_equal(first["binary_f1"]["ci95"], second["binary_f1"]["ci95"])
    assert _ci_equal(first["binary_auc"]["ci95"], second["binary_auc"]["ci95"])
    assert _ci_equal(first["macro_f1"]["ci95"], second["macro_f1"]["ci95"])


def test_empty_slice_gives_nan_not_crash():
    frame = _linked([])
    metrics = ev.compute_metrics(frame, bootstrap_n=50)
    assert metrics["n"] == 0
    assert metrics["binary_f1"]["value"] != metrics["binary_f1"]["value"]
    assert metrics["binary_f1"]["ci95"] == (float("nan"), float("nan")) or all(
        v != v for v in metrics["binary_f1"]["ci95"]
    )


# --------------------------------------------------------------------------- #
# Связывание
# --------------------------------------------------------------------------- #


def _targets_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dedup_group_id": ["g0000", "g0001", "g0002"],
            "study_folder": ["study_a", "study_b", "study_c"],
            "region": [cfg.REGION_SPINE, cfg.REGION_HIP, cfg.REGION_SPINE],
            "split": ["train", "val", "train"],
            "has_target": [True, True, False],
            "quality_class": [1, 0, 0],
            "violation_type": [cfg.VIOLATION_SPINE_AXIS, "", ""],
        }
    )


def test_link_via_dedup_group_id_filters_has_target_and_dedups():
    targets = _targets_frame()
    submission = pd.DataFrame(
        {
            "dedup_group_id": ["g0000", "g0000", "g0001", "g0002"],  # g0000 продублирован
            "quality_class": [1, 1, 0, 0],
            "quality_prob": [0.9, 0.9, 0.1, 0.05],
            "violation_type": [cfg.VIOLATION_SPINE_AXIS, cfg.VIOLATION_SPINE_AXIS, "", ""],
        }
    )
    linked = ev.link_via_dedup_group_id(submission, targets)
    # g0002 не размечен (has_target=False) -> исключён; g0000 -> одна строка.
    assert sorted(linked["dedup_group_id"]) == ["g0000", "g0001"]
    row0 = linked[linked["dedup_group_id"] == "g0000"].iloc[0]
    assert row0["quality_class_true"] == 1
    assert row0["quality_class_pred"] == 1
    assert row0["violation_type_true"] == cfg.VIOLATION_SPINE_AXIS


def test_attach_pixel_hash_checks_relative_path_and_copies_by_position():
    manifest = pd.DataFrame(
        {
            "relative_path": ["a/1.dcm", "b/2.dcm", "c/3.dcm"],
            "pixel_hash": ["hashA", "hashB", "hashC"],
        }
    )
    submission = pd.DataFrame(
        {
            "relative_path": ["a/1.dcm", "b/2.dcm", "c/3.dcm"],
            "quality_class": [1, 0, 0],
            "quality_prob": [0.9, 0.1, 0.05],
            "violation_type": [cfg.VIOLATION_SPINE_AXIS, "", ""],
        }
    )
    linked = ev.attach_pixel_hash(submission, manifest)
    assert list(linked["pixel_hash"]) == ["hashA", "hashB", "hashC"]


def test_attach_pixel_hash_raises_on_relative_path_mismatch():
    manifest = pd.DataFrame({"relative_path": ["a/1.dcm", "b/2.dcm"], "pixel_hash": ["hashA", "hashB"]})
    submission = pd.DataFrame({"relative_path": ["a/1.dcm", "X/other.dcm"], "quality_class": [0, 0]})
    with pytest.raises(ValueError):
        ev.attach_pixel_hash(submission, manifest)


def test_link_via_pixel_hash_joins_through_manifest_reference():
    targets = _targets_frame()
    # Сабмит уже с приклеенным pixel_hash (attach_pixel_hash) — свои id дублей.
    submission = pd.DataFrame(
        {
            "relative_path": ["a/1.dcm", "b/2.dcm", "c/3.dcm"],
            "pixel_hash": ["hashA", "hashB", "hashC"],
            "quality_class": [1, 0, 0],
            "quality_prob": [0.9, 0.1, 0.05],
            "violation_type": [cfg.VIOLATION_SPINE_AXIS, "", ""],
        }
    )
    # manifest.parquet этапа 0: pixel_hash -> dedup_group_id, ЧУЖИЕ по значению
    # (совпадают с targets, но не с id, которые присвоил бы describe_inputs).
    manifest_reference = pd.DataFrame(
        {
            "pixel_hash": ["hashA", "hashA", "hashB", "hashC"],  # hashA задублирован
            "dedup_group_id": ["g0000", "g0000", "g0001", "g0002"],
        }
    )
    linked = ev.link_via_pixel_hash(submission, manifest_reference, targets)
    assert sorted(linked["dedup_group_id"]) == ["g0000", "g0001"]
    row0 = linked[linked["dedup_group_id"] == "g0000"].iloc[0]
    assert row0["quality_class_true"] == 1
    assert row0["quality_class_pred"] == 1


# --------------------------------------------------------------------------- #
# Sanity: submission_reference.csv против targets.csv -> 1.0
# --------------------------------------------------------------------------- #


@requires_data
def test_sanity_reference_submission_gives_perfect_metrics():
    if not cfg.SUBMISSION_REFERENCE_CSV.exists() or not cfg.TARGETS_CSV.exists():
        pytest.skip("нет artifacts/submission_reference.csv или targets.csv — прогнать пайплайн этапа 0")
    metrics = ev.sanity_check(bootstrap_n=50)
    assert metrics["n"] > 0
    assert metrics["binary_f1"]["value"] == 1.0
    assert metrics["binary_auc"]["value"] == 1.0
    assert metrics["macro_f1"]["value"] == 1.0
    for vtype in cfg.VIOLATION_TYPES:
        value = metrics["by_type"][vtype]["value"]
        assert value != value or value == 1.0  # NaN (нет позитивов типа) или 1.0
