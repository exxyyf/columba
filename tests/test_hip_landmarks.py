"""Этап 4, шаги 2–4: ориентиры бедра — инварианты, деградация, разметка.

Обязательные тесты этапа 4 (часть про `hip_landmarks.py`):
анизотропный угол диафиза на синтетике, зеркальная симметрия, устойчивость
к маскированию углов, `not_evaluated` без исключений, схема и полнота JSON
разметки.
"""

from __future__ import annotations

import numpy as np
import pytest

from columba import config as cfg
from columba.hip_landmarks import (
    HIP_ROI_SIGNAL_KEYS,
    HIP_SIGNAL_KEYS,
    compute_hip_roi_signals,
    compute_hip_signals,
    detect_hip_landmarks,
    load_annotations,
    synth_crop_hip_field,
    validate_annotation_entry,
)
from columba.spine_keypoints import STATUS_NOT_EVALUATED, STATUS_OK

from conftest import requires_data

HIP_TEST_HEIGHT = 320
HIP_TEST_WIDTH = 280


def synthetic_femur(
    angle_deg: float = 0.0,
    height: int = HIP_TEST_HEIGHT,
    width: int = HIP_TEST_WIDTH,
    head_dx_mm: float = -30.0,
) -> np.ndarray:
    """Упрощённый силуэт бедра с известным ФИЗИЧЕСКИМ наклоном диафиза.

    Анатомическая форма (сверху вниз, доли высоты кадра): круглая головка
    (~12%) -> узкий перешеек шейки (~27%, минимум ширины) -> широкая зона
    вертелов (~38%, максимум ширины: большой — латерально, малый — медиально,
    несимметричные добавки поверх симметричной огибающей) -> сужение к
    диафизу постоянной ширины (полоса из config). Диафиз наклонён на
    `angle_deg` от вертикали в мм (пивот — середина полосы диафиза). Головка
    смещена на `head_dx_mm` от оси диафиза (отрицательное значение -> головка
    левее диафиза, "медиальная" сторона слева).
    """
    y_px = np.arange(height)[:, None].astype(np.float64)
    x_px = np.arange(width)[None, :].astype(np.float64)
    y_mm = y_px * cfg.PIXEL_SPACING_MM_Y
    x_mm = x_px * cfg.PIXEL_SPACING_MM_X
    height_mm = height * cfg.PIXEL_SPACING_MM_Y

    shaft_lo, shaft_hi = cfg.HIP_SHAFT_BAND_FRACTIONS
    pivot_row = 0.5 * (shaft_lo + shaft_hi) * height
    pivot_mm = pivot_row * cfg.PIXEL_SPACING_MM_Y
    center_mm = (width - 1) / 2.0 * cfg.PIXEL_SPACING_MM_X
    axis_x_mm = center_mm + np.tan(np.deg2rad(angle_deg)) * (y_mm - pivot_mm)

    medial_sign = -1.0 if head_dx_mm < 0 else 1.0  # головка на стороне меньших x

    neck_frac, troch_frac = 0.27, 0.38
    neck_hw_mm, troch_hw_mm, shaft_hw_mm = 6.0, 17.0, 11.0
    y_frac = (y_mm / height_mm).ravel()
    half_width_mm = np.interp(
        y_frac,
        [0.0, neck_frac, troch_frac, shaft_lo, 1.0],
        [neck_hw_mm, neck_hw_mm, troch_hw_mm, shaft_hw_mm, shaft_hw_mm],
    ).reshape(y_mm.shape)
    # Ось шейки плавно смещается от положения головки (head_dx_mm) к оси
    # диафиза между головкой и вертельной зоной — иначе шейка и головка не
    # соединяются в один силуэт (нужно для трекинга связной компоненты).
    head_frac = 0.12
    axis_shift_mm = np.interp(
        y_frac, [0.0, head_frac, troch_frac, 1.0], [head_dx_mm, head_dx_mm, 0.0, 0.0]
    ).reshape(y_mm.shape)
    axis_x_mm = axis_x_mm + axis_shift_mm

    def bump(center_frac: float, amount_mm: float, half_span_frac: float) -> np.ndarray:
        y0_mm = center_frac * height_mm
        sigma_mm = half_span_frac * height_mm
        return amount_mm * np.exp(-0.5 * ((y_mm - y0_mm) / sigma_mm) ** 2)

    greater_bump = bump(troch_frac, amount_mm=20.0, half_span_frac=0.035)
    lesser_bump = bump(0.32, amount_mm=9.0, half_span_frac=0.025)

    # Медиальная сторона (к головке) получает выступ малого вертела,
    # латеральная (от головки) — выступ большого вертела.
    if medial_sign < 0:
        left_extra, right_extra = lesser_bump, greater_bump
    else:
        left_extra, right_extra = greater_bump, lesser_bump

    left_edge_mm = axis_x_mm - half_width_mm - left_extra
    right_edge_mm = axis_x_mm + half_width_mm + right_extra
    shaft = np.clip(np.minimum(x_mm - left_edge_mm, right_edge_mm - x_mm), 0.0, 4.0) / 4.0
    # Выше головки кости нет (край кадра/фон) — иначе плоская "шейка"-заглушка
    # над диском создаёт ложный минимум ширины ВЫШЕ истинного перешейка шейки.
    shaft = shaft * (y_mm / height_mm >= head_frac)

    head_y_mm = head_frac * height_mm
    head_x_mm = center_mm + np.tan(np.deg2rad(angle_deg)) * (head_y_mm - pivot_mm) + head_dx_mm
    radius_mm = 15.0
    dist_mm = np.hypot(x_mm - head_x_mm, y_mm - head_y_mm)
    head = np.clip(1.0 - dist_mm / radius_mm, 0.0, 1.0)

    image = np.maximum(shaft, head).astype(np.float32)
    return image


# --------------------------------------------------------------------------- #
# Анизотропный угол диафиза
# --------------------------------------------------------------------------- #


def test_shaft_tilt_respects_anisotropy():
    true_angle = 9.0
    image = synthetic_femur(true_angle)
    landmarks = detect_hip_landmarks(image, side="left")
    assert landmarks.status == STATUS_OK
    signals = compute_hip_signals(image, landmarks)
    assert signals["shaft_tilt_deg"] == pytest.approx(true_angle, abs=1.5)

    # Наивный угол по сырым пикселям (без учёта PIXEL_SPACING) обязан
    # отличаться: тангенс масштабируется в ANISOTROPY_Y_OVER_X раз.
    points = np.asarray(landmarks.shaft_points_xy)
    (x0, y0), (x1, y1) = points[0], points[-1]
    naive = float(np.degrees(np.arctan2(abs(x1 - x0), abs(y1 - y0))))
    assert abs(naive - true_angle) > 2.0

    vertical = detect_hip_landmarks(synthetic_femur(0.0), side="left")
    vertical_signals = compute_hip_signals(synthetic_femur(0.0), vertical)
    assert abs(vertical_signals["shaft_tilt_deg"]) < 2.0


# --------------------------------------------------------------------------- #
# Зеркальная симметрия
# --------------------------------------------------------------------------- #


def test_mirror_symmetry_flips_side_keeps_signals():
    image = synthetic_femur(angle_deg=6.0, head_dx_mm=-28.0)
    landmarks = detect_hip_landmarks(image)
    signals = compute_hip_signals(image, landmarks)
    assert landmarks.status == STATUS_OK

    mirrored = np.fliplr(image).copy()
    mirrored_landmarks = detect_hip_landmarks(mirrored)
    mirrored_signals = compute_hip_signals(mirrored, mirrored_landmarks)
    assert mirrored_landmarks.status == STATUS_OK

    assert mirrored_landmarks.side != landmarks.side

    for key in ("shaft_tilt_deg", "lesser_trochanter_prominence", "neck_shaft_angle_deg"):
        assert mirrored_signals[key] == pytest.approx(signals[key], abs=1.5)
    # head_offset_mm приведён к виду левого бедра явным знаком по стороне —
    # должен совпадать (не просто по модулю) у кадра и его отражения.
    assert mirrored_signals["head_offset_mm"] == pytest.approx(signals["head_offset_mm"], abs=3.0)


# --------------------------------------------------------------------------- #
# Устойчивость к маскированию
# --------------------------------------------------------------------------- #


def test_masking_corners_does_not_change_landmarks():
    image = synthetic_femur(angle_deg=4.0, head_dx_mm=-25.0)
    baseline = detect_hip_landmarks(image, side="left")
    baseline_signals = compute_hip_signals(image, baseline)
    assert baseline.status == STATUS_OK

    masked = image.copy()
    masked[: image.shape[0] // 6, : image.shape[1] // 5] = 0.0
    masked[-image.shape[0] // 6 :, -image.shape[1] // 5 :] = 0.0
    masked_landmarks = detect_hip_landmarks(masked, side="left")
    masked_signals = compute_hip_signals(masked, masked_landmarks)
    assert masked_landmarks.status == STATUS_OK

    assert masked_signals["shaft_tilt_deg"] == pytest.approx(baseline_signals["shaft_tilt_deg"], abs=1.0)
    np.testing.assert_allclose(
        np.asarray(masked_landmarks.shaft_points_xy), np.asarray(baseline.shaft_points_xy), atol=3.0
    )
    if baseline.head_center_xy is not None:
        np.testing.assert_allclose(
            masked_landmarks.head_center_xy, baseline.head_center_xy, atol=3.0
        )


# --------------------------------------------------------------------------- #
# Деградация без исключений
# --------------------------------------------------------------------------- #


def test_not_evaluated_on_degenerate_inputs_without_exceptions():
    empty = np.zeros((4, 4), dtype=np.float32)
    landmarks = detect_hip_landmarks(empty)
    assert landmarks.status == STATUS_NOT_EVALUATED and landmarks.reason
    signals = compute_hip_signals(empty, landmarks)
    assert set(signals) == set(HIP_SIGNAL_KEYS)
    assert all(v != v for v in signals.values())  # все NaN

    dark = np.zeros((HIP_TEST_HEIGHT, HIP_TEST_WIDTH), dtype=np.float32)
    dark_landmarks = detect_hip_landmarks(dark)
    assert dark_landmarks.status == STATUS_NOT_EVALUATED
    dark_signals = compute_hip_signals(dark, dark_landmarks)
    assert all(v != v for v in dark_signals.values())


def test_signal_keys_order_and_completeness():
    assert HIP_SIGNAL_KEYS == (
        "shaft_tilt_deg",
        "lesser_trochanter_prominence",
        "lesser_trochanter_corridor_dev",
        "neck_shaft_angle_deg",
        "head_offset_mm",
    )
    image = synthetic_femur(5.0)
    landmarks = detect_hip_landmarks(image, side="left")
    signals = compute_hip_signals(image, landmarks)
    assert tuple(signals) == HIP_SIGNAL_KEYS


# --------------------------------------------------------------------------- #
# Этап 9, п. 9.5: синтетические кропы нарушенных отступов поля
# --------------------------------------------------------------------------- #


def test_synth_crop_hip_field_shrinks_only_the_requested_edge():
    image = synthetic_femur(0.0)
    height, width = image.shape

    top = synth_crop_hip_field(image, np.random.default_rng(0), edge="top")
    assert top.shape[0] < height and top.shape[1] == width

    bottom = synth_crop_hip_field(image, np.random.default_rng(0), edge="bottom")
    assert bottom.shape[0] < height and bottom.shape[1] == width

    left = synth_crop_hip_field(image, np.random.default_rng(0), edge="left")
    assert left.shape[1] < width and left.shape[0] == height

    right = synth_crop_hip_field(image, np.random.default_rng(0), edge="right")
    assert right.shape[1] < width and right.shape[0] == height


def test_synth_crop_hip_field_is_deterministic_given_seed():
    image = synthetic_femur(0.0)
    a = synth_crop_hip_field(image, np.random.default_rng(42), edge="top")
    b = synth_crop_hip_field(image, np.random.default_rng(42), edge="top")
    assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)


def test_synth_crop_hip_field_random_edge_is_one_of_four(monkeypatch):
    image = synthetic_femur(0.0)
    seen = set()
    for seed in range(20):
        cropped = synth_crop_hip_field(image, np.random.default_rng(seed))
        if cropped.shape[0] < image.shape[0]:
            seen.add("y")
        if cropped.shape[1] < image.shape[1]:
            seen.add("x")
    assert seen  # хотя бы одно измерение уменьшилось хоть раз


def test_synth_crop_hip_field_rejects_unknown_edge():
    with pytest.raises(ValueError):
        synth_crop_hip_field(synthetic_femur(0.0), np.random.default_rng(0), edge="diagonal")


def test_synth_crop_hip_field_violates_protocol_margin_on_top():
    """Синтетический кроп сверху действительно нарушает протокольные 3 см —
    сигнал `rows_px` (высота растра) падает ниже предыдущей высоты минимум
    на HIP_ROI_TOP_MARGIN_MM."""
    image = synthetic_femur(0.0)
    cropped = synth_crop_hip_field(image, np.random.default_rng(1), edge="top")
    cut_mm = (image.shape[0] - cropped.shape[0]) * cfg.PIXEL_SPACING_MM_Y
    assert cut_mm >= cfg.HIP_ROI_TOP_MARGIN_MM - 1e-6


def test_synth_crop_hip_field_violates_protocol_margin_on_side():
    image = synthetic_femur(0.0)
    cropped = synth_crop_hip_field(image, np.random.default_rng(1), edge="left")
    cut_mm = (image.shape[1] - cropped.shape[1]) * cfg.PIXEL_SPACING_MM_X
    assert cut_mm >= cfg.HIP_ROI_SIDE_MARGIN_MM - 1e-6


def test_synth_crop_hip_field_signals_still_computable():
    """Синтетика — только источник калибровочных примеров (docstring
    `synth_crop_hip_field`): чекер сигналов должен честно отработать на
    результате (не упасть, роль NaN — только при полной деградации кадра)."""
    image = synthetic_femur(0.0)
    cropped = synth_crop_hip_field(image, np.random.default_rng(2), edge="bottom")
    landmarks = detect_hip_landmarks(cropped, side="left")
    roi_signals = compute_hip_roi_signals(cropped, landmarks)
    assert set(roi_signals) == set(HIP_ROI_SIGNAL_KEYS)
    assert roi_signals["rows_px"] == pytest.approx(float(cropped.shape[0]))


# --------------------------------------------------------------------------- #
# Разметка: схема и полнота
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not cfg.HIP_LANDMARKS_JSON.exists(), reason="нет файла разметки")
def test_annotation_file_schema():
    annotations = load_annotations()
    assert len(annotations) > 0
    for gid, entry in annotations.items():
        assert validate_annotation_entry(entry) == [], gid


@requires_data
@pytest.mark.skipif(not cfg.HIP_LANDMARKS_JSON.exists(), reason="нет файла разметки")
def test_annotation_covers_hip_groups(manifest):
    from columba.build_hip_annotations import hip_representative_frame

    annotations = load_annotations()
    expected = set(hip_representative_frame()["dedup_group_id"])
    assert expected == set(annotations)
