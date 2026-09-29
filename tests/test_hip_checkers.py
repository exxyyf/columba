"""Этап 4, шаг 9: чекер укладки бедра (`hip_checkers.py`) — инварианты.

Модуль ориентиров бедра (`columba.hip_landmarks`) разрабатывался
параллельно и может отсутствовать на диске: `check_hip_positioning` здесь
тестируется на фейковых `landmarks`/`signals` (никакого импорта
`hip_landmarks`), `run_hip_checkers` — через фейковый модуль в `sys.modules`
(monkeypatch), интеграционные/маскировочные тесты с реальными ориентирами —
`skipif` на отсутствие модуля.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

from columba import config as cfg
from columba.hip_checkers import check_hip_positioning, check_hip_roi, run_hip_checkers
from columba.spine_checkers import CheckerResult
from columba.spine_keypoints import STATUS_NOT_EVALUATED, STATUS_OK

from conftest import requires_data

try:
    importlib.import_module("columba.hip_landmarks")
    HAS_HIP_LANDMARKS = True
except ImportError:
    HAS_HIP_LANDMARKS = False

requires_hip_landmarks = pytest.mark.skipif(
    not HAS_HIP_LANDMARKS, reason="модуль columba.hip_landmarks ещё не готов (пишется параллельно)"
)


class FakeLandmarks:
    """Заглушка контракта модуля A (`HipLandmarks`)."""

    def __init__(self, status=STATUS_OK, reason="", side="left"):
        self.status = status
        self.reason = reason
        self.side = side
        self.head_center_xy = (10.0, 20.0)
        self.shaft_points_xy = [(10.0, 30.0), (10.0, 40.0)]
        self.lesser_trochanter_xy = (12.0, 25.0)
        self.greater_trochanter_xy = (5.0, 22.0)

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "head_center_xy": self.head_center_xy,
            "shaft_points_xy": self.shaft_points_xy,
            "lesser_trochanter_xy": self.lesser_trochanter_xy,
            "greater_trochanter_xy": self.greater_trochanter_xy,
        }


def _pixels(height=180, width=280) -> np.ndarray:
    return np.random.default_rng(0).uniform(0.0, 1.0, size=(height, width)).astype(np.float32)


def _signals(**overrides) -> dict:
    base = {
        "shaft_tilt_deg": 1.0,
        "lesser_trochanter_prominence": 60.0,
        "lesser_trochanter_corridor_dev": 5.0,
        "neck_shaft_angle_deg": 130.0,
        "head_offset_mm": 3.0,
    }
    base.update(overrides)
    return base


class FakeCnnPredictor:
    """Заглушка `hip_cnn.HipPositioningPredictor` — фиксированная вероятность
    (или `None`, чтобы сымитировать неудачный инференс/препроцессинг)."""

    def __init__(self, probability: float | None):
        self.probability = probability
        self.calls: list[tuple] = []

    def predict_pixels(self, pixels_normalized, side):
        self.calls.append((pixels_normalized, side))
        return self.probability


# --------------------------------------------------------------------------- #
# check_hip_positioning: score/flag/status — драйвер CNN, шаг 7
# --------------------------------------------------------------------------- #


def test_score_comes_from_cnn_probability():
    predictor = FakeCnnPredictor(0.42)
    result = check_hip_positioning(_pixels(), FakeLandmarks(), _signals(), cnn_predictor=predictor)
    assert isinstance(result, CheckerResult)
    assert result.checker == "hip_positioning"
    assert result.status == STATUS_OK
    assert result.score == pytest.approx(0.42)
    assert result.signals["cnn_probability"] == pytest.approx(0.42)


def test_flag_follows_threshold_from_config(monkeypatch):
    monkeypatch.setattr(cfg, "HIP_POSITIONING_FLAG_THRESHOLD", 0.5)
    below = check_hip_positioning(_pixels(), FakeLandmarks(), _signals(), cnn_predictor=FakeCnnPredictor(0.499))
    above = check_hip_positioning(_pixels(), FakeLandmarks(), _signals(), cnn_predictor=FakeCnnPredictor(0.5))
    assert not below.flag
    assert above.flag  # >=, как у чекеров позвоночника (укладка/предметы)


def test_calibrated_threshold_is_a_probability():
    """Шаг 7 (hip_eval.cnn_operating_point, реальные данные): порог — best-F1
    на train OOF вероятностей CNN, обязан лежать в [0, 1] (домен вероятности,
    не геометрического сигнала)."""
    assert 0.0 < cfg.HIP_POSITIONING_FLAG_THRESHOLD <= 1.0


def test_signals_include_geometry_and_cnn_probability():
    """Геометрические сигналы и ориентиры остаются для объяснимости, даже
    когда решение принимает CNN (докстринг модуля, шаг 7 decisions)."""
    signals = _signals()
    landmarks = FakeLandmarks()
    result = check_hip_positioning(_pixels(), landmarks, signals, cnn_predictor=FakeCnnPredictor(0.1))
    for key in signals:
        assert key in result.signals
    for key in landmarks.to_dict():
        assert key in result.signals
    assert "cnn_probability" in result.signals


def test_cnn_predictor_receives_side_from_landmarks():
    predictor = FakeCnnPredictor(0.1)
    landmarks = FakeLandmarks(side="right")
    check_hip_positioning(_pixels(), landmarks, _signals(), cnn_predictor=predictor)
    assert predictor.calls[-1][1] == "right"


# --------------------------------------------------------------------------- #
# side_confident (этап 9, п. 9.5): только объяснимость, не флаг/скор
# --------------------------------------------------------------------------- #


def test_side_confident_default_none_omits_signal():
    """Обратная совместимость: без явного `side_confident` (старые вызовы,
    например текущий `inference.py`) сигнал не появляется — поведение как до
    этапа 9.5."""
    result = check_hip_positioning(_pixels(), FakeLandmarks(), _signals(), cnn_predictor=FakeCnnPredictor(0.1))
    assert "hip_side_confident" not in result.signals


def test_side_confident_true_is_recorded_without_changing_flag_or_score():
    predictor = FakeCnnPredictor(0.9)
    result = check_hip_positioning(
        _pixels(), FakeLandmarks(), _signals(), cnn_predictor=predictor, side_confident=True
    )
    assert result.signals["hip_side_confident"] is True
    assert "hip_side_low_confidence_reason" not in result.signals
    assert result.score == pytest.approx(0.9)


def test_side_low_confidence_adds_explanatory_reason_but_flag_unchanged():
    """Низкая уверенность стороны пишется в `signals` для объяснимости, но НЕ
    подавляет флаг (см. докстринг `_side_confidence_signals` — просадка
    recall при неверной оценке доли low-confidence на закрытом тесте)."""
    monkeypatch_predictor = FakeCnnPredictor(0.95)
    confident_result = check_hip_positioning(
        _pixels(), FakeLandmarks(), _signals(), cnn_predictor=monkeypatch_predictor, side_confident=True
    )
    low_conf_result = check_hip_positioning(
        _pixels(), FakeLandmarks(), _signals(), cnn_predictor=monkeypatch_predictor, side_confident=False
    )
    assert low_conf_result.signals["hip_side_confident"] is False
    assert "hip_side_low_confidence_reason" in low_conf_result.signals
    assert low_conf_result.flag == confident_result.flag
    assert low_conf_result.score == pytest.approx(confident_result.score)


def test_side_confident_recorded_even_when_not_evaluated():
    """`side_confident` доступен ещё до успешной детекции ориентиров/CNN —
    признак пишется и в `not_evaluated`-путях (объяснимость не должна
    теряться из-за более раннего честного отказа)."""
    landmarks = FakeLandmarks(status=STATUS_NOT_EVALUATED, reason="контраст низкий")
    result = check_hip_positioning(_pixels(), landmarks, _signals(), cnn_predictor=None, side_confident=False)
    assert result.status == STATUS_NOT_EVALUATED
    assert result.signals["hip_side_confident"] is False


# --------------------------------------------------------------------------- #
# not_evaluated: landmarks.status != ok, без CNN-предиктора, или инференс не удался
# --------------------------------------------------------------------------- #


def test_not_evaluated_when_landmarks_not_ok():
    landmarks = FakeLandmarks(status=STATUS_NOT_EVALUATED, reason="контраст низкий")
    result = check_hip_positioning(_pixels(), landmarks, _signals(), cnn_predictor=FakeCnnPredictor(0.9))
    assert result.status == STATUS_NOT_EVALUATED
    assert result.reason
    assert not result.flag
    assert result.score != result.score  # NaN


def test_not_evaluated_when_cnn_predictor_is_none():
    """Веса CNN не подключены: геометрия одна статистически не отличима от
    случайного (nested CV AUC 0.564, ДИ пересекает 0.5) — честная деградация,
    не флагование по слабому сигналу."""
    result = check_hip_positioning(_pixels(), FakeLandmarks(), _signals(), cnn_predictor=None)
    assert result.status == STATUS_NOT_EVALUATED
    assert result.reason
    assert not result.flag
    assert result.score != result.score  # NaN


def test_not_evaluated_when_cnn_inference_fails():
    """`predict_pixels` возвращает `None` (честная деградация препроцессинга
    внутри `HipPositioningPredictor`) -> not_evaluated, не исключение."""
    result = check_hip_positioning(_pixels(), FakeLandmarks(), _signals(), cnn_predictor=FakeCnnPredictor(None))
    assert result.status == STATUS_NOT_EVALUATED
    assert result.reason


def test_not_evaluated_on_degenerate_frame():
    result = check_hip_positioning(
        np.zeros((4, 4), dtype=np.float32), FakeLandmarks(), _signals(), cnn_predictor=FakeCnnPredictor(0.9)
    )
    assert result.status == STATUS_NOT_EVALUATED


# --------------------------------------------------------------------------- #
# check_hip_roi: score/flag/status — этап 5, шаг 4/9
# --------------------------------------------------------------------------- #


def _roi_signals(**overrides) -> dict:
    base = {
        "field_height_deficit": -0.1,
        "rows_px": 280.0,
        "top_margin_px": 20.0,
        "bottom_margin_px": 22.0,
        "side_margin_px": 40.0,
    }
    base.update(overrides)
    return base


def test_hip_roi_score_matches_field_height_deficit():
    result = check_hip_roi(_pixels(), FakeLandmarks(), _roi_signals(field_height_deficit=0.05))
    assert isinstance(result, CheckerResult)
    assert result.checker == "hip_roi"
    assert result.status == STATUS_OK
    assert result.score == pytest.approx(0.05)


def test_hip_roi_flag_follows_threshold_from_config(monkeypatch):
    monkeypatch.setattr(cfg, "HIP_ROI_FLAG_THRESHOLD", 0.0)
    below = check_hip_roi(_pixels(), FakeLandmarks(), _roi_signals(field_height_deficit=-0.001))
    above = check_hip_roi(_pixels(), FakeLandmarks(), _roi_signals(field_height_deficit=0.0))
    assert not below.flag
    assert above.flag  # >=, как у остальных чекеров


def test_hip_roi_signal_can_be_negative_unlike_clipped_spine_deficit():
    """Важное отличие от `spine_positioning.crest_deficit`: не отсекается
    нулём снизу (docstring `compute_hip_roi_signals`) — иначе почти все
    train-негативы (116/122 с rows_px >= REF) схлопывались бы в один и тот
    же скор 0.0, портя сигмоиду `quality_prob` агрегатора."""
    result = check_hip_roi(_pixels(), FakeLandmarks(), _roi_signals(field_height_deficit=-0.4))
    assert result.score == pytest.approx(-0.4)
    assert not result.flag


def test_hip_roi_does_not_require_landmarks_ok():
    """`field_height_deficit` — высота растра, не нуждается в ориентирах;
    `landmarks.status != ok` не должен давать `not_evaluated` (в отличие от
    `check_hip_positioning`) — только NaN у margin-сигналов."""
    landmarks = FakeLandmarks(status=STATUS_NOT_EVALUATED, reason="контраст низкий")
    result = check_hip_roi(_pixels(), landmarks, _roi_signals(field_height_deficit=0.1))
    assert result.status == STATUS_OK
    assert result.flag


def test_hip_roi_signals_include_margins_when_landmarks_ok():
    signals = _roi_signals()
    result = check_hip_roi(_pixels(), FakeLandmarks(), signals)
    for key in signals:
        assert key in result.signals
    for key in FakeLandmarks().to_dict():
        assert key in result.signals


def test_hip_roi_not_evaluated_when_score_is_nan():
    result = check_hip_roi(_pixels(), FakeLandmarks(), _roi_signals(field_height_deficit=float("nan")))
    assert result.status == STATUS_NOT_EVALUATED
    assert result.reason
    assert not result.flag


def test_hip_roi_not_evaluated_on_degenerate_frame():
    result = check_hip_roi(np.zeros((4, 4), dtype=np.float32), FakeLandmarks(), _roi_signals())
    assert result.status == STATUS_NOT_EVALUATED


# --------------------------------------------------------------------------- #
# run_hip_checkers: фейковый модуль hip_landmarks через sys.modules
# --------------------------------------------------------------------------- #


def _install_fake_hip_landmarks_module(monkeypatch, landmarks=None, signals=None, roi_signals=None, calls=None):
    fake = types.ModuleType("columba.hip_landmarks")

    def detect_hip_landmarks(pixels_normalized, side=None):
        if calls is not None:
            calls["detect"] = calls.get("detect", 0) + 1
        return landmarks if landmarks is not None else FakeLandmarks()

    def compute_hip_signals(pixels_normalized, landmarks_arg):
        if calls is not None:
            calls["signals"] = calls.get("signals", 0) + 1
        return signals if signals is not None else _signals()

    def compute_hip_roi_signals(pixels_normalized, landmarks_arg):
        if calls is not None:
            calls["roi_signals"] = calls.get("roi_signals", 0) + 1
        return roi_signals if roi_signals is not None else _roi_signals()

    fake.detect_hip_landmarks = detect_hip_landmarks
    fake.compute_hip_signals = compute_hip_signals
    fake.compute_hip_roi_signals = compute_hip_roi_signals
    fake.HipLandmarks = FakeLandmarks
    fake.HIP_SIGNAL_KEYS = (
        "shaft_tilt_deg",
        "lesser_trochanter_prominence",
        "lesser_trochanter_corridor_dev",
        "neck_shaft_angle_deg",
        "head_offset_mm",
    )
    fake.HIP_ROI_SIGNAL_KEYS = (
        "field_height_deficit",
        "rows_px",
        "top_margin_px",
        "bottom_margin_px",
        "side_margin_px",
    )
    monkeypatch.setitem(sys.modules, "columba.hip_landmarks", fake)
    return fake


def test_run_hip_checkers_returns_both_keys(monkeypatch):
    _install_fake_hip_landmarks_module(monkeypatch)
    results = run_hip_checkers(_pixels(), cnn_predictor=FakeCnnPredictor(0.1))
    assert set(results) == {"hip_positioning", "hip_roi"}
    assert results["hip_positioning"].status == STATUS_OK
    assert results["hip_roi"].status == STATUS_OK


def test_run_hip_checkers_without_predictor_positioning_not_evaluated_roi_unaffected(monkeypatch):
    """`hip_roi` не зависит от CNN вовсе (docstring `check_hip_roi`) — без
    предиктора флагуется только `hip_positioning`."""
    _install_fake_hip_landmarks_module(monkeypatch)
    results = run_hip_checkers(_pixels())
    assert results["hip_positioning"].status == STATUS_NOT_EVALUATED
    assert results["hip_roi"].status == STATUS_OK


def test_run_hip_checkers_propagates_import_error_when_module_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "columba.hip_landmarks", None)  # имитирует отсутствие модуля
    with pytest.raises(ImportError):
        run_hip_checkers(_pixels())


def test_run_hip_checkers_forwards_side_confident(monkeypatch):
    """`run_hip_checkers` пробрасывает `side_confident` только в
    `hip_positioning` (аргумент опциональный, дефолт `None` не ломает старые
    вызовы из `inference.py`, который пока его не передаёт — этап 9.5)."""
    _install_fake_hip_landmarks_module(monkeypatch)
    results = run_hip_checkers(_pixels(), cnn_predictor=FakeCnnPredictor(0.9), side_confident=False)
    assert results["hip_positioning"].signals["hip_side_confident"] is False
    assert "hip_side_confident" not in results["hip_roi"].signals


def test_run_hip_checkers_default_side_confident_omits_signal(monkeypatch):
    _install_fake_hip_landmarks_module(monkeypatch)
    results = run_hip_checkers(_pixels(), cnn_predictor=FakeCnnPredictor(0.9))
    assert "hip_side_confident" not in results["hip_positioning"].signals


# --------------------------------------------------------------------------- #
# Реальный модуль ориентиров (когда готов): маскирование углов не меняет выход
# --------------------------------------------------------------------------- #


HIP_TEST_FILES = (cfg.TEST_DIR / "CR000000_ППОБ.dcm", cfg.TEST_DIR / "CR000001_ЛПОБ.dcm")


@requires_data
@requires_hip_landmarks
@pytest.mark.parametrize("hip_file", HIP_TEST_FILES)
def test_masking_corners_does_not_change_checker_output(hip_file):
    """Инвариант 6 / шаг 8: синтетическое зануление углов реального кадра
    бедра не должно менять скор/флаг чекера (по аналогии со
    `spine_unet_check.mask_corners`/тестом позвоночника). С реальным
    CNN-предиктором (веса скачаны, см. WEIGHTS.md) — маскирование не должно
    сдвигать вероятность CNN, не только геометрию."""
    from columba.dicom_io import normalize, read_dicom
    from columba.hip_cnn import load_hip_positioning_predictor
    from columba.hip_landmarks import compute_hip_signals, detect_hip_landmarks

    predictor = load_hip_positioning_predictor()
    if predictor is None:
        pytest.skip("веса hip_positioning_cnn.pt не скачаны в этом окружении")

    result = read_dicom(hip_file)
    assert result.ok
    image = normalize(result.pixels, result.tags)

    landmarks = detect_hip_landmarks(image)
    if landmarks.status != STATUS_OK:
        pytest.skip("реальный детектор ориентиров не сработал на этом кадре")
    baseline_signals = compute_hip_signals(image, landmarks)
    baseline = check_hip_positioning(image, landmarks, baseline_signals, cnn_predictor=predictor)

    masked = image.copy()
    masked[: image.shape[0] // 5, : image.shape[1] // 5] = 0.0
    masked[-image.shape[0] // 6 :, -image.shape[1] // 4 :] = 0.0
    masked_landmarks = detect_hip_landmarks(masked)
    if masked_landmarks.status != STATUS_OK:
        pytest.skip("реальный детектор ориентиров не сработал на маскированном кадре")
    masked_signals = compute_hip_signals(masked, masked_landmarks)
    masked_result = check_hip_positioning(masked, masked_landmarks, masked_signals, cnn_predictor=predictor)

    assert masked_result.flag == baseline.flag
    if baseline.score == baseline.score and masked_result.score == masked_result.score:
        assert masked_result.score == pytest.approx(baseline.score, rel=0.15, abs=0.1)


# --------------------------------------------------------------------------- #
# Интеграция: describe_inputs, кэш дублей, независимость от тегов/имён
# --------------------------------------------------------------------------- #

HIP_CHECKER_COLUMNS = [
    f"{key}_{field}" for key in ("hip_positioning", "hip_roi") for field in ("score", "flag", "status")
]


@requires_data
@requires_hip_landmarks
def test_describe_inputs_attaches_hip_checkers_for_hip_only(tmp_path):
    import shutil

    from columba.inference import describe_inputs

    for name in ("CR000000_ПОП.dcm", "CR000000_ППОБ.dcm"):
        shutil.copy(cfg.TEST_DIR / name, tmp_path / name)
    frame = describe_inputs(tmp_path)
    for column in HIP_CHECKER_COLUMNS + ["hip_signals"]:
        assert column in frame.columns

    spine_row = frame[frame["region_final"] == cfg.REGION_SPINE].iloc[0]
    hip_row = frame[frame["region_final"] == cfg.REGION_HIP].iloc[0]
    assert hip_row["hip_positioning_status"] in (STATUS_OK, STATUS_NOT_EVALUATED)
    # hip_roi не зависит от CNN — должен быть ok, если ориентиры/кадр в порядке.
    assert hip_row["hip_roi_status"] in (STATUS_OK, STATUS_NOT_EVALUATED)
    # Позвоночник — нейтральный пропуск для колонок бедра.
    assert pd.isna(spine_row["hip_positioning_status"]) and pd.isna(spine_row["hip_signals"])
    assert pd.isna(spine_row["hip_roi_status"])


@requires_data
@requires_hip_landmarks
def test_duplicates_hit_hip_checkers_once(tmp_path, monkeypatch):
    """Кэш по хэшу пикселей охватывает чекер бедра: 3 копии — 1 вызов."""
    import shutil

    from columba import hip_checkers as hip_checkers_module
    from columba.inference import describe_inputs

    calls = {"n": 0}
    original = hip_checkers_module.run_hip_checkers

    def counting(pixels, **kwargs):
        calls["n"] += 1
        return original(pixels, **kwargs)

    monkeypatch.setattr(hip_checkers_module, "run_hip_checkers", counting)
    for index in range(3):
        shutil.copy(HIP_TEST_FILES[0], tmp_path / f"copy_{index}.dcm")
    frame = describe_inputs(tmp_path)
    assert len(frame) == 3
    assert calls["n"] == 1
    assert frame["hip_positioning_status"].notna().all()


# --------------------------------------------------------------------------- #
# Инвариант 7: эндопротезы и 248 px кадры — как обычные бёдра, без спецветки
# --------------------------------------------------------------------------- #

# dedup_group_id -> ожидание по инварианту «эндопротезы — как обычные
# бёдра» (stages/stage_4.md):
# g0204/g0162 — исследование с эндопротезированием (правая сторона, g0162,
# без таргета вовсе); g0004/g0111 — 248 px кадры, region_final этапа 1
# (эвристика/CNN) относит их к бедру, хотя region в targets.csv (метка
# этапа 0, грубее) — "unknown".
HIP_EDGE_CASE_GROUP_IDS = ("g0204", "g0162", "g0004", "g0111")


@requires_data
@requires_hip_landmarks
@pytest.mark.parametrize("group_id", HIP_EDGE_CASE_GROUP_IDS)
def test_endoprosthesis_and_248px_frames_pass_as_ordinary_hip(group_id, manifest, tmp_path):
    """Чекер не должен падать/спецветвиться на эндопротезах или узких 248 px
    кадрах — они проходят тот же путь `describe_inputs`, что и любое бедро."""
    import shutil

    from columba.inference import describe_inputs

    rows = manifest[(manifest["dedup_group_id"] == group_id) & manifest["is_group_representative"]]
    if rows.empty:
        pytest.skip(f"{group_id} отсутствует в этом срезе манифеста")
    src = cfg.STUDIES_DIR / rows.iloc[0]["relative_path"]
    dst = tmp_path / src.name
    shutil.copy(src, dst)

    frame = describe_inputs(tmp_path)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["region_final"] == cfg.REGION_HIP, f"{group_id}: ожидался region_final=hip, получено {row['region_final']}"
    assert row["hip_positioning_status"] in (STATUS_OK, STATUS_NOT_EVALUATED)
