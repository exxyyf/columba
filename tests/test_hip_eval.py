"""Тесты `hip_eval.py` (этап 4, шаг 7) — только на синтетике.

`build_signal_table` не тестируется здесь: она читает реальные DICOM и
вызывает `hip_landmarks.detect_hip_landmarks`/`compute_hip_signals` —
модуль ориентиров разрабатывался параллельно и может отсутствовать (импорт
внутри функции — намеренно, чтобы модуль оставался импортируемым без него).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from columba import config as cfg
from columba import hip_eval


def _make_table(n_studies: int = 8, per_study: int = 1, seed: int = 0) -> pd.DataFrame:
    """Синтетическая таблица сигналов: одна строка на исследование по
    умолчанию, чтобы группировка по `study_folder` была однозначной."""
    rng = np.random.default_rng(seed)
    rows = []
    versions = ["18.41.005", "18.50.082"]
    for i in range(n_studies):
        for j in range(per_study):
            label = int(i % 2 == 0)  # ровно половина исследований — позитивы
            rows.append(
                {
                    "dedup_group_id": f"g{i:04d}{j}",
                    "study_folder": f"study_{i:03d}",
                    "split": "train",
                    "software_version": versions[i % 2],
                    "y_positioning": label,
                    "y_roi": int(rng.random() < 0.1),
                    "sig_a": 10.0 if label else 0.0,
                    "sig_b": rng.normal(0.0, 1.0),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# AUC сигнала, идеально разделяющего классы
# --------------------------------------------------------------------------- #


def test_signal_auc_perfect_separator_is_one():
    table = _make_table()
    table["split"] = "train"
    result = hip_eval.signal_auc_table(table, label_col="y_positioning")
    row = result[(result["signal"] == "sig_a") & (result["transform"] == "raw") & (result["fold"] == "train")]
    assert len(row) == 1
    assert row.iloc[0]["auc"] == 1.0


def test_signal_auc_handles_val_split():
    table = _make_table()
    table.loc[table.index[:2], "split"] = "val"
    result = hip_eval.signal_auc_table(table, label_col="y_positioning")
    assert "val" in set(result["fold"])


# --------------------------------------------------------------------------- #
# Вложенная CV: нет утечки исследований между fit и hold
# --------------------------------------------------------------------------- #


def test_study_cv_folds_no_group_leakage():
    table = _make_table(n_studies=12, per_study=3)
    n_folds = 4
    cv_fold = hip_eval._study_cv_folds(table, n_folds)
    for fold_id in range(n_folds):
        fit_studies = set(table.loc[(cv_fold.notna()) & (cv_fold != fold_id), "study_folder"])
        hold_studies = set(table.loc[cv_fold == fold_id, "study_folder"])
        assert fit_studies.isdisjoint(hold_studies)
        assert len(hold_studies) > 0


def test_nested_group_cv_respects_group_leakage_via_folds_log():
    table = _make_table(n_studies=12, per_study=3)
    result = hip_eval.nested_group_cv(table, candidates=["sig_a", "abs:sig_b"], label_col="y_positioning")
    seen_studies: set[str] = set()
    for fold in result["folds"]:
        holdout = set(fold["holdout_studies"])
        assert seen_studies.isdisjoint(holdout)
        seen_studies |= holdout


# --------------------------------------------------------------------------- #
# Воспроизводимость по seed
# --------------------------------------------------------------------------- #


def test_nested_group_cv_reproducible_with_logreg_candidate():
    table = _make_table(n_studies=12, per_study=2)
    candidates = ["sig_a", "abs:sig_b", ("sig_a", "sig_b")]
    first = hip_eval.nested_group_cv(table, candidates, label_col="y_positioning", verbose=False)
    second = hip_eval.nested_group_cv(table, candidates, label_col="y_positioning", verbose=False)
    pd.testing.assert_series_equal(first["oof_score"], second["oof_score"])
    pd.testing.assert_series_equal(first["oof_flag"], second["oof_flag"])
    assert first["folds"] == second["folds"]


# --------------------------------------------------------------------------- #
# NaN-сигналы не роняют пайплайн
# --------------------------------------------------------------------------- #


def test_nan_signals_do_not_crash():
    table = _make_table(n_studies=12, per_study=2)
    table.loc[table.index[::3], "sig_a"] = np.nan
    table["sig_all_nan"] = np.nan

    auc_table = hip_eval.signal_auc_table(table, label_col="y_positioning")
    assert len(auc_table) > 0
    assert np.isfinite(auc_table.loc[auc_table["signal"] == "sig_a", "auc"].dropna()).all()

    result = hip_eval.nested_group_cv(
        table, candidates=["sig_a", "sig_all_nan", ("sig_a", "sig_b", "sig_all_nan")], label_col="y_positioning"
    )
    assert not result["oof_score"].isna().all()
    auc_value = result["summary"]["auc"]["value"]
    assert np.isfinite(auc_value) or auc_value != auc_value  # число либо честный NaN, без исключений


# --------------------------------------------------------------------------- #
# Красный флаг версии софта
# --------------------------------------------------------------------------- #


def test_version_redflag_triggers_when_score_tracks_version():
    rng = np.random.default_rng(1)
    n = 40
    versions = np.array(["18.41.005"] * (n // 2) + ["18.50.082"] * (n // 2))
    table = pd.DataFrame(
        {
            "dedup_group_id": [f"g{i:04d}" for i in range(n)],
            "study_folder": [f"study_{i:03d}" for i in range(n)],
            "split": "train",
            "software_version": versions,
            "y_positioning": rng.integers(0, 2, size=n),
        }
    )
    # скор идеально совпадает с версией и не зависит от метки
    scores = (versions == "18.50.082").astype(float)
    result = hip_eval.version_breakdown(table, scores, label_col="y_positioning")
    assert result["auc_vs_version"]["value"] == 1.0
    assert result["red_flag"] is True


def test_version_redflag_does_not_trigger_when_score_tracks_label():
    rng = np.random.default_rng(2)
    n = 40
    versions = np.array(["18.41.005"] * (n // 2) + ["18.50.082"] * (n // 2))
    rng.shuffle(versions)
    labels = rng.integers(0, 2, size=n)
    table = pd.DataFrame(
        {
            "dedup_group_id": [f"g{i:04d}" for i in range(n)],
            "study_folder": [f"study_{i:03d}" for i in range(n)],
            "split": "train",
            "software_version": versions,
            "y_positioning": labels,
        }
    )
    scores = labels.astype(float)  # скор == метка, версия несвязана
    result = hip_eval.version_breakdown(table, scores, label_col="y_positioning")
    assert result["red_flag"] is False


# --------------------------------------------------------------------------- #
# Рабочая точка рантайма
# --------------------------------------------------------------------------- #


def test_runtime_operating_point_train_val_metrics():
    table = _make_table(n_studies=10, per_study=1)
    table.loc[table.index[-3:], "split"] = "val"

    def score_fn(frame: pd.DataFrame) -> np.ndarray:
        return frame["sig_a"].astype(float).values

    result = hip_eval.runtime_operating_point(table, score_fn, threshold=5.0, label_col="y_positioning")
    assert set(result) == {"train", "val"}
    train_entry = result["train"]
    assert train_entry["tp"] + train_entry["fn"] == int(
        table.loc[table["split"] == "train", "y_positioning"].sum()
    )
    assert train_entry["fp"] == 0  # порог 5.0 разделяет sig_a=10/0 идеально


def test_seed_constant_used_for_logreg():
    assert cfg.SEED == 42


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
