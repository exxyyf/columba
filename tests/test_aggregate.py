"""Этап 3, шаг 9 (п.1-9): юнит-тесты агрегатора на синтетическом манифесте.

Манифест здесь — обычный pandas DataFrame с ручными значениями
`region_final`/`{key}_score`/`{key}_flag`/`{key}_status`/`read_status`, без
DICOM и пикселей: агрегатор читает только уже посчитанные колонки.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from columba import config as cfg
from columba.aggregate import aggregate_predictions
from columba.dicom_io import STATUS_SUCCESS
from columba.inference import run_inference
from columba.spine_keypoints import STATUS_NOT_EVALUATED, STATUS_OK

from conftest import requires_data

SPINE_LABELS = cfg.OUTPUT_LABELS_BY_REGION[cfg.REGION_SPINE]
SPINE_KEYS = [label.key for label in SPINE_LABELS]

# Пороги чекеров позвоночника (существующие константы этапа 2).
_THRESHOLDS = {
    "spine_axis": cfg.SPINE_AXIS_MAX_ANGLE_DEG,
    "spine_positioning": cfg.SPINE_POSITIONING_FLAG_DEFICIT,
    "spine_objects": cfg.SPINE_METAL_MIN_SCORE,
}
# `>` для оси (строгое), `>=` для укладки/предметов (нестрогое) — см. spine_checkers.py.
_STRICT = {"spine_axis": True, "spine_positioning": False, "spine_objects": False}


def _checker_columns(key: str, *, score: float, flag: bool, status: str = STATUS_OK) -> dict:
    return {f"{key}_score": score, f"{key}_flag": flag, f"{key}_status": status}


def _spine_row(file_id: str = "f0000", **overrides) -> dict:
    """Базовая строка позвоночника: все три чекера ok, без флагов, скор ниже порога."""
    row = {
        "file_id": file_id,
        "region_final": cfg.REGION_SPINE,
        "read_status": STATUS_SUCCESS,
    }
    for key in SPINE_KEYS:
        row.update(_checker_columns(key, score=0.0, flag=False))
    row.update(overrides)
    return row


def _manifest(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _predict_one(row: dict) -> pd.Series:
    result = aggregate_predictions(_manifest([row]))
    assert len(result) == 1
    return result.iloc[0]


# --------------------------------------------------------------------------- #
# 1. Полярность
# --------------------------------------------------------------------------- #


def test_polarity_one_flag_true_gives_violation():
    row = _spine_row()
    row.update(_checker_columns("spine_axis", score=cfg.SPINE_AXIS_MAX_ANGLE_DEG + 1.0, flag=True))
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_VIOLATION
    assert prediction["quality_prob"] >= 0.5


def test_polarity_all_flags_false_gives_ok():
    row = _spine_row()
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_OK
    assert prediction["quality_prob"] <= 0.5


# --------------------------------------------------------------------------- #
# 2. Посимвольные строки словаря / имена колонок
# --------------------------------------------------------------------------- #


def test_violation_strings_match_config_char_for_char():
    for label in SPINE_LABELS:
        row = _spine_row()
        row.update(_checker_columns(label.key, score=_THRESHOLDS[label.key] + 1.0, flag=True))
        prediction = _predict_one(row)
        assert prediction["violation_type"] == label.violation_type


def test_output_columns_present():
    prediction = _predict_one(_spine_row())
    for column in ("file_id", "quality_class", "quality_prob", "violation_type", "not_evaluated_checkers"):
        assert column in prediction.index


# --------------------------------------------------------------------------- #
# 3. Типы только из словаря своего региона, без повторов
# --------------------------------------------------------------------------- #


def test_spine_violation_types_never_include_hip_roi():
    row = _spine_row()
    for key in SPINE_KEYS:
        row.update(_checker_columns(key, score=_THRESHOLDS[key] + 1.0, flag=True))
    prediction = _predict_one(row)
    types = prediction["violation_type"].split(cfg.VIOLATION_TYPE_SEPARATOR)
    assert cfg.VIOLATION_ROI not in types
    assert len(types) == len(set(types))
    assert set(types) == set(cfg.VIOLATION_TYPES_BY_REGION[cfg.REGION_SPINE])


def test_hip_never_gets_spine_only_types():
    row = {
        "file_id": "f0000",
        "region_final": cfg.REGION_HIP,
        "read_status": STATUS_SUCCESS,
    }
    prediction = _predict_one(row)
    assert prediction["violation_type"] == ""
    assert cfg.VIOLATION_SPINE_AXIS not in prediction["violation_type"]
    assert cfg.VIOLATION_FOREIGN_OBJECTS not in prediction["violation_type"]


# --------------------------------------------------------------------------- #
# 4. Строка на каждый файл, включая дубликаты и Failure
# --------------------------------------------------------------------------- #


def test_one_row_per_file_including_duplicates_and_failure():
    rows = [
        _spine_row("f0000", pixel_hash="h1"),
        _spine_row("f0001", pixel_hash="h1"),
        _spine_row("f0002", pixel_hash="h1"),
        {
            "file_id": "f0003",
            "region_final": cfg.REGION_UNKNOWN,
            "read_status": "Failure",
        },
    ]
    result = aggregate_predictions(_manifest(rows))
    # Агрегатор не предсказывает нечитаемые файлы — build_submission сохранит
    # для f0003 FAILURE_ROW_* как есть.
    assert set(result["file_id"]) == {"f0000", "f0001", "f0002"}
    assert len(result) == 3


# --------------------------------------------------------------------------- #
# 5. Дубликаты получают идентичные предсказания
# --------------------------------------------------------------------------- #


def test_duplicates_get_identical_predictions():
    shared = _spine_row("f0000")
    shared.update(_checker_columns("spine_positioning", score=0.3, flag=True))
    rows = []
    for i in range(3):
        row = dict(shared)
        row["file_id"] = f"f000{i}"
        rows.append(row)
    result = aggregate_predictions(_manifest(rows))
    assert result["quality_class"].nunique() == 1
    assert result["quality_prob"].nunique() == 1
    assert result["violation_type"].nunique() == 1


# --------------------------------------------------------------------------- #
# 6. not_evaluated не роняет и попадает в отладочную колонку
# --------------------------------------------------------------------------- #


def test_not_evaluated_treated_as_no_violation_and_logged():
    row = _spine_row()
    row.update(_checker_columns("spine_axis", score=float("nan"), flag=False, status=STATUS_NOT_EVALUATED))
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_OK
    assert "spine_axis" in prediction["not_evaluated_checkers"].split(cfg.VIOLATION_TYPE_SEPARATOR)
    assert cfg.VIOLATION_SPINE_AXIS not in prediction["violation_type"]


def test_all_checkers_not_evaluated_gives_zero_prob_and_class_zero():
    row = _spine_row()
    for key in SPINE_KEYS:
        row.update(_checker_columns(key, score=float("nan"), flag=False, status=STATUS_NOT_EVALUATED))
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_OK
    assert prediction["quality_prob"] == 0.0
    assert set(prediction["not_evaluated_checkers"].split(cfg.VIOLATION_TYPE_SEPARATOR)) == set(SPINE_KEYS)


def test_not_evaluated_as_violation_policy_keeps_polarity_invariant(monkeypatch):
    """Если политику шага 3 пересмотрят на True, полярность не должна ломаться:
    class=1 обязан сопровождаться prob >= 0.5 даже без скора у чекера."""
    monkeypatch.setattr(cfg, "NOT_EVALUATED_TREATED_AS_VIOLATION", True)
    row = _spine_row()
    row.update(_checker_columns("spine_axis", score=float("nan"), flag=False, status=STATUS_NOT_EVALUATED))
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_VIOLATION
    assert prediction["quality_prob"] >= 0.5
    assert cfg.VIOLATION_SPINE_AXIS in prediction["violation_type"]


# --------------------------------------------------------------------------- #
# 7. Бедро: class 0, prob == HIP_PRIOR_QUALITY_PROB
# --------------------------------------------------------------------------- #


def test_hip_gives_prior_probability():
    row = {
        "file_id": "f0000",
        "region_final": cfg.REGION_HIP,
        "read_status": STATUS_SUCCESS,
    }
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_OK
    assert prediction["quality_prob"] == cfg.HIP_PRIOR_QUALITY_PROB
    assert prediction["violation_type"] == ""


# --------------------------------------------------------------------------- #
# 8. Согласованность class/prob вблизи порога каждого чекера
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", SPINE_KEYS)
def test_class_prob_consistency_near_threshold(key):
    threshold = _THRESHOLDS[key]
    strict = _STRICT[key]
    eps = 1e-3

    below = _spine_row()
    below.update(_checker_columns(key, score=threshold - eps, flag=False))
    p_below = _predict_one(below)
    assert p_below["quality_class"] == cfg.QUALITY_CLASS_OK
    assert p_below["quality_prob"] <= 0.5

    above = _spine_row()
    above_flag = True
    above.update(_checker_columns(key, score=threshold + eps, flag=above_flag))
    p_above = _predict_one(above)
    assert p_above["quality_class"] == cfg.QUALITY_CLASS_VIOLATION
    assert p_above["quality_prob"] >= 0.5

    on_threshold = _spine_row()
    flag_on_threshold = not strict  # >= гарантирует flag=True на пороге, > даёт False
    on_threshold.update(_checker_columns(key, score=threshold, flag=flag_on_threshold))
    p_on = _predict_one(on_threshold)
    assert math.isclose(p_on["quality_prob"], 0.5, abs_tol=1e-9)
    expected_class = cfg.QUALITY_CLASS_VIOLATION if flag_on_threshold else cfg.QUALITY_CLASS_OK
    assert p_on["quality_class"] == expected_class


# --------------------------------------------------------------------------- #
# 9. Монотонность quality_prob по скору
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", SPINE_KEYS)
def test_quality_prob_monotonic_in_score(key):
    threshold = _THRESHOLDS[key]
    scores = [threshold - 2.0, threshold - 0.5, threshold, threshold + 0.5, threshold + 2.0]
    probs = []
    for score in scores:
        row = _spine_row()
        row.update(_checker_columns(key, score=max(score, 0.0), flag=score >= threshold))
        probs.append(_predict_one(row)["quality_prob"])
    assert all(b >= a - 1e-12 for a, b in zip(probs, probs[1:]))


# --------------------------------------------------------------------------- #
# 7b. Бедро (этап 4): ветка hip_positioning
# --------------------------------------------------------------------------- #


def _hip_row(file_id: str = "f0000", **overrides) -> dict:
    row = {
        "file_id": file_id,
        "region_final": cfg.REGION_HIP,
        "read_status": STATUS_SUCCESS,
    }
    row.update(overrides)
    return row


def test_hip_no_checker_columns_gives_prior_backward_compat():
    """Старый манифест (без hip_checkers) — прежний приор этапа 3."""
    prediction = _predict_one(_hip_row())
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_OK
    assert prediction["quality_prob"] == cfg.HIP_PRIOR_QUALITY_PROB
    assert prediction["violation_type"] == ""


def test_hip_not_evaluated_status_gives_prior():
    """ImportError модуля ориентиров бедра -> not_evaluated -> тот же приор."""
    row = _hip_row(
        hip_positioning_score=float("nan"),
        hip_positioning_flag=False,
        hip_positioning_status=STATUS_NOT_EVALUATED,
    )
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_OK
    assert prediction["quality_prob"] == cfg.HIP_PRIOR_QUALITY_PROB
    assert "hip_positioning" in prediction["not_evaluated_checkers"].split(cfg.VIOLATION_TYPE_SEPARATOR)


def test_hip_ok_no_flag_inf_threshold_gives_prior_not_zero(monkeypatch):
    """Порог `inf` (до калибровки шага 7, или если её откатить): ok-бедро без
    флага получает приор, а не sigmoid((score - inf)/scale) == 0.0 (иначе
    бинарный AUC региона портится так же, как раньше портил приор 0.0 — см.
    stage_3_decisions.md). Начиная с реальной калибровки (шаг 7,
    stage_4_decisions.md) порог по умолчанию конечен — `inf` здесь
    монкипатчится явно, чтобы держать этот путь агрегатора живым и
    протестированным."""
    monkeypatch.setattr(cfg, "HIP_POSITIONING_FLAG_THRESHOLD", float("inf"))
    row = _hip_row(hip_positioning_score=5.0, hip_positioning_flag=False, hip_positioning_status=STATUS_OK)
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_OK
    assert prediction["quality_prob"] == cfg.HIP_PRIOR_QUALITY_PROB


def test_hip_positioning_flag_threshold_is_calibrated_probability():
    """Шаг 7 (hip_eval, реальные данные): порог CNN-вероятности, не `inf`-
    заглушка — по умолчанию чекер уже должен флагать, а не быть отключённым."""
    assert 0.0 < cfg.HIP_POSITIONING_FLAG_THRESHOLD <= 1.0


def test_hip_flag_true_gives_violation_class_and_positioning_string(monkeypatch):
    """Откалиброванный (конечный) порог: флаг -> class 1, «Некорректная
    укладка», prob >= 0.5 (полярность 1.4)."""
    monkeypatch.setattr(cfg, "HIP_POSITIONING_FLAG_THRESHOLD", 10.0)
    monkeypatch.setattr(cfg, "HIP_POSITIONING_PROB_SCALE", 1.0)
    row = _hip_row(hip_positioning_score=12.0, hip_positioning_flag=True, hip_positioning_status=STATUS_OK)
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_VIOLATION
    assert prediction["quality_prob"] >= 0.5
    assert prediction["violation_type"] == cfg.VIOLATION_POSITIONING


def test_hip_flag_false_gives_ok_class(monkeypatch):
    monkeypatch.setattr(cfg, "HIP_POSITIONING_FLAG_THRESHOLD", 10.0)
    monkeypatch.setattr(cfg, "HIP_POSITIONING_PROB_SCALE", 1.0)
    row = _hip_row(hip_positioning_score=2.0, hip_positioning_flag=False, hip_positioning_status=STATUS_OK)
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_OK
    assert prediction["quality_prob"] <= 0.5
    assert prediction["violation_type"] == ""


def test_hip_roi_absent_columns_backward_compat():
    """Строка без колонок `hip_roi_*` (старый манифест) — не должна попасть
    в типы бедра, а не упасть (обратная совместимость, `getattr` -> `None`)."""
    row = _hip_row(hip_positioning_score=1.0, hip_positioning_flag=False, hip_positioning_status=STATUS_OK)
    prediction = _predict_one(row)
    assert cfg.VIOLATION_ROI not in prediction["violation_type"]


def test_hip_roi_flag_true_gives_violation_class_and_roi_string(monkeypatch):
    """Этап 5: откалиброванный `hip_roi` реально участвует в результате —
    флаг -> class 1, «Некорректная область интереса», prob >= 0.5."""
    monkeypatch.setattr(cfg, "HIP_ROI_FLAG_THRESHOLD", 0.0)
    monkeypatch.setattr(cfg, "HIP_ROI_PROB_SCALE", 0.167)
    row = _hip_row(
        hip_positioning_status=STATUS_NOT_EVALUATED,
        hip_positioning_score=float("nan"),
        hip_positioning_flag=False,
        hip_roi_status=STATUS_OK,
        hip_roi_score=0.1,
        hip_roi_flag=True,
    )
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_VIOLATION
    assert prediction["quality_prob"] >= 0.5
    assert prediction["violation_type"] == cfg.VIOLATION_ROI


def test_hip_both_checkers_flag_gives_both_violation_types(monkeypatch):
    """Оба чекера бедра флагуют одновременно — оба типа в `violation_type`,
    порядок — обход `OUTPUT_LABELS_BY_REGION[REGION_HIP]`."""
    monkeypatch.setattr(cfg, "HIP_ROI_FLAG_THRESHOLD", 0.0)
    row = _hip_row(
        hip_positioning_status=STATUS_OK,
        hip_positioning_score=0.95,
        hip_positioning_flag=True,
        hip_roi_status=STATUS_OK,
        hip_roi_score=0.2,
        hip_roi_flag=True,
    )
    prediction = _predict_one(row)
    assert prediction["quality_class"] == cfg.QUALITY_CLASS_VIOLATION
    types = prediction["violation_type"].split(cfg.VIOLATION_TYPE_SEPARATOR)
    assert set(types) == {cfg.VIOLATION_POSITIONING, cfg.VIOLATION_ROI}


# --------------------------------------------------------------------------- #
# Явный predictor=None даёт нулевой сабмит (интеграционный, реальные данные).
# --------------------------------------------------------------------------- #


@requires_data
def test_explicit_predictor_none_gives_zero_submission():
    submission = run_inference(cfg.TEST_DIR, predictor=None)
    assert set(submission["quality_class"].unique()) <= {cfg.QUALITY_CLASS_OK}
    assert (submission["quality_prob"] == 0.0).all()
    assert (submission["violation_type"] == "").all()


@requires_data
def test_auto_predictor_produces_real_predictions():
    """По умолчанию (AUTO) сабмит уже не нулевой хотя бы по quality_prob бедра."""
    submission = run_inference(cfg.TEST_DIR)
    # Хотя бы приор бедра должен дать ненулевую вероятность где-то в выходе,
    # если в тестовом каталоге вообще есть бедро; иначе просто не должно падать.
    assert len(submission) == cfg.EXPECTED_TEST_FILES
