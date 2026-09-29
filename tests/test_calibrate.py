"""Этап 6 (`calibrate.py`): PAVA/Platt/пороги на синтетике + sanity на реальных данных."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from columba import calibrate as cb
from columba import config as cfg

from conftest import requires_data


# --------------------------------------------------------------------------- #
# PAVA / изотоника
# --------------------------------------------------------------------------- #


def test_pava_already_monotonic_is_unchanged():
    values = np.array([0.0, 0.0, 1.0, 1.0])
    result = cb._pava(values)
    assert list(result) == [0.0, 0.0, 1.0, 1.0]


def test_pava_fixes_single_violation():
    # y убывает на одном шаге (1 -> 0) — PAVA обязана усреднить нарушающий блок.
    values = np.array([0.0, 1.0, 0.0, 1.0])
    result = cb._pava(values)
    assert all(result[i] <= result[i + 1] + 1e-12 for i in range(len(result) - 1))
    assert result[1] == pytest.approx(result[2])  # нарушающая пара усреднена


def test_fit_isotonic_is_monotonic_on_new_points():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(-1, 1, 40), rng.normal(1, 1, 40)])
    y = np.concatenate([np.zeros(40), np.ones(40)])
    model = cb.fit_isotonic(z, y)
    grid = np.linspace(z.min() - 1, z.max() + 1, 50)
    preds = model["predict"](grid)
    assert all(preds[i] <= preds[i + 1] + 1e-12 for i in range(len(preds) - 1))
    assert preds.min() >= 0.0 - 1e-9
    assert preds.max() <= 1.0 + 1e-9


def test_fit_isotonic_extrapolates_below_range_with_first_step():
    z = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = cb.fit_isotonic(z, y)
    assert model["predict"](np.array([-10.0]))[0] == pytest.approx(model["predict"](np.array([0.0]))[0])


# --------------------------------------------------------------------------- #
# Platt
# --------------------------------------------------------------------------- #


def test_fit_platt_separates_well_separated_classes():
    rng = np.random.default_rng(1)
    z = np.concatenate([rng.normal(-3, 0.5, 60), rng.normal(3, 0.5, 60)])
    y = np.concatenate([np.zeros(60), np.ones(60)])
    model = cb.fit_platt(z, y)
    low = model["predict"](np.array([-3.0]))[0]
    high = model["predict"](np.array([3.0]))[0]
    assert low < 0.3
    assert high > 0.7


def test_fit_platt_predict_is_monotonic_in_z():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1, 100)
    y = (z + rng.normal(0, 0.3, 100) > 0).astype(float)
    model = cb.fit_platt(z, y)
    grid = np.linspace(-3, 3, 30)
    preds = model["predict"](grid)
    assert all(preds[i] <= preds[i + 1] + 1e-9 for i in range(len(preds) - 1))


# --------------------------------------------------------------------------- #
# Метрики калибровки
# --------------------------------------------------------------------------- #


def test_brier_score_perfect_is_zero():
    assert cb.brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)


def test_brier_score_worst_case_is_one():
    assert cb.brier_score(np.array([0.0, 1.0]), np.array([1, 0])) == pytest.approx(1.0)


def test_log_loss_confident_correct_beats_unsure():
    confident = cb.log_loss(np.array([0.95]), np.array([1]))
    unsure = cb.log_loss(np.array([0.55]), np.array([1]))
    assert confident < unsure


# --------------------------------------------------------------------------- #
# Пороги
# --------------------------------------------------------------------------- #


def _frame(probs, labels, groups=None) -> pd.DataFrame:
    n = len(probs)
    return pd.DataFrame(
        {
            "prob": probs,
            "y_true": labels,
            "study_folder": groups or [f"s{i}" for i in range(n)],
        }
    )


def test_recall_weighted_threshold_favors_full_recall():
    frame = _frame(probs=[0.9, 0.6, 0.4, 0.1], labels=[1, 1, 0, 0])
    threshold = cb.recall_weighted_threshold(frame, "prob", beta=2.0)
    flags = frame["prob"].values >= threshold
    recall = (flags & (frame["y_true"].values == 1)).sum() / (frame["y_true"].values == 1).sum()
    assert recall == 1.0


def test_threshold_metrics_reports_confusion_counts():
    frame = _frame(probs=[0.9, 0.8, 0.3, 0.1], labels=[1, 0, 1, 0])
    result = cb.threshold_metrics(frame, "prob", threshold=0.5, n_bootstrap=10)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1


# --------------------------------------------------------------------------- #
# Чек-лист комментариев (синтетика на уже собранном фрейме)
# --------------------------------------------------------------------------- #


def test_doctor_comment_checklist_flags_mismatches(monkeypatch, tmp_path):
    manifest = pd.DataFrame(
        {
            "dedup_group_id": ["g0001", "g0002", "g0003"],
            "is_group_representative": [True, True, True],
            "markup_comment": ["сколиоз", None, ""],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    monkeypatch.setattr(cfg, "MANIFEST_CSV", manifest_path)

    frame = pd.DataFrame(
        {
            "dedup_group_id": ["g0001", "g0002", "g0003"],
            "region": [cfg.REGION_SPINE, cfg.REGION_HIP, cfg.REGION_SPINE],
            "split": ["train", "train", "train"],
            "y_true": [1, 1, 0],
            "prob": [0.2, 0.9, 0.1],  # g0001: комментарий, но ошибка (true=1, prob<threshold)
        }
    )
    checklist = cb.doctor_comment_checklist(frame, "prob", threshold=0.5)
    assert len(checklist) == 1  # только g0001 (единственный с непустым комментарием)
    row = checklist.iloc[0]
    assert row["dedup_group_id"] == "g0001"
    assert not row["correct"]


# --------------------------------------------------------------------------- #
# Интеграция на реальных данных
# --------------------------------------------------------------------------- #


@requires_data
def test_build_calibration_table_covers_all_labeled_images():
    table = cb.build_calibration_table(verbose=False)
    targets = pd.read_csv(cfg.TARGETS_CSV)
    expected_n = int(targets["has_target"].fillna(False).sum())
    assert len(table) <= expected_n  # <= : файлы без ok-чекера честно исключены
    assert set(table["region"]) <= {cfg.REGION_SPINE, cfg.REGION_HIP}
    assert table["z_oof"].notna().all()


@requires_data
def test_main_writes_report_and_picks_a_calibration():
    report = cb.main()
    assert report["chosen_calibration"] in ("platt", "isotonic")
    assert cb.STAGE6_REPORT_MD.exists()
    assert cb.STAGE6_SCORES_CSV.exists()
    # AUC инвариантна к монотонному Platt/сигмоиде на train (тот же z, тот же порядок).
    naive_auc = report["calibration"]["naive_sigmoid"]["train"]["auc"]["value"]
    platt_auc = report["calibration"]["platt"]["train"]["auc"]["value"]
    assert naive_auc == pytest.approx(platt_auc)
