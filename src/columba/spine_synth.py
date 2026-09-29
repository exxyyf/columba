"""Этап 2, шаг 7: синтетика для модуля позвоночника.

Три генератора, все — воспроизводимые функции с seed, работающие с
нормализованными пикселями:

* поворот кадра на заданный угол — ТОЛЬКО в изотропном пространстве с
  обратным ресемплингом в исходную сетку (позитивы `spine_axis`);
* вставка синтетического «металла» (яркие эллипсы/цепочки с размытием краёв)
  — позитивы `spine_objects`;
* кроп верха/низа кадра (срезание гребней/Th12) — позитивы `spine_positioning`.

Синтетика создаётся только из train-фолда и помечается флагом is_synthetic
на стороне вызывающего кода; в валидацию не попадает никогда.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config as cfg
from .spine_keypoints import iso_height, mm_to_iso_px, resample_isotropic_np


# --------------------------------------------------------------------------- #
# Повороты в изотропном пространстве
# --------------------------------------------------------------------------- #


def rotate_isotropic(iso_image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Повернуть изотропный кадр на angle_deg.

    В экранных координатах (y вниз) положительный угол наклоняет вертикальные
    структуры так, что их верх уходит ВЛЕВО. Для чекера оси важен только
    модуль угла; соответствие изображения и `rotate_points_iso` закрыто
    юнит-тестом.
    """
    tensor = torch.as_tensor(np.ascontiguousarray(iso_image), dtype=torch.float32)
    theta_rad = float(np.deg2rad(angle_deg))
    cos, sin = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    # grid_sample с этой theta берёт для выходного пикселя точку входа,
    # повёрнутую на -angle: содержимое кадра поворачивается на +angle.
    theta = torch.tensor([[cos, -sin * tensor.shape[0] / tensor.shape[1], 0.0],
                          [sin * tensor.shape[1] / tensor.shape[0], cos, 0.0]],
                         dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, (1, 1, *tensor.shape), align_corners=False)
    return F.grid_sample(
        tensor.reshape(1, 1, *tensor.shape), grid, mode="bilinear",
        padding_mode="zeros", align_corners=False,
    )[0, 0].numpy()


def rotate_points_iso(points_xy: np.ndarray, iso_shape: tuple[int, int], angle_deg: float) -> np.ndarray:
    """Повернуть точки (x, y) вокруг центра кадра согласованно с изображением."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    height, width = iso_shape
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    theta_rad = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta_rad), np.sin(theta_rad)
    # Экранные координаты (y вниз): поворот содержимого на +angle означает
    # для точек матрицу [[cos, sin], [-sin, cos]] относительно центра.
    rotation = np.array([[cos, sin], [-sin, cos]])
    return (points - center) @ rotation.T + center


def rotate_raw_frame(pixels_normalized: np.ndarray, angle_deg: float) -> np.ndarray:
    """Поворот исходного кадра: изотропный ресемплинг -> поворот -> обратно.

    Возвращает кадр той же формы, что вход (анизотропная сетка).
    """
    raw_height, raw_width = pixels_normalized.shape
    iso = resample_isotropic_np(pixels_normalized)
    rotated = rotate_isotropic(iso, angle_deg)
    tensor = torch.as_tensor(rotated, dtype=torch.float32).reshape(1, 1, *rotated.shape)
    back = F.interpolate(tensor, size=(raw_height, raw_width), mode="bilinear", align_corners=False)
    return back[0, 0].numpy()


# --------------------------------------------------------------------------- #
# Синтетический металл
# --------------------------------------------------------------------------- #


def insert_synthetic_metal(
    pixels_normalized: np.ndarray,
    rng: np.random.Generator,
    *,
    kind: str | None = None,
) -> np.ndarray:
    """Вставить яркий «металлический» объект (цепочка бус или клипса).

    Статистика подобрана по реальным позитивам: насыщенная яркость (около
    максимума диапазона), резкие, слегка размытые края; цепочка — дуга в
    верхней части кадра, клипса — компактный продолговатый объект.
    """
    kind = kind or rng.choice(["chain", "clip"])
    height, width = pixels_normalized.shape
    canvas = np.zeros_like(pixels_normalized, dtype=np.float64)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    # Физические координаты, мм — эллипсы рисуются круглыми в мм.
    y_mm = yy * cfg.PIXEL_SPACING_MM_Y
    x_mm = xx * cfg.PIXEL_SPACING_MM_X

    if kind == "chain":
        # Дуга «бус» поперёк верхней части кадра: цепочка мелких эллипсов.
        n_beads = int(rng.integers(12, 22))
        x0_mm = rng.uniform(0.15, 0.35) * width * cfg.PIXEL_SPACING_MM_X
        x1_mm = rng.uniform(0.65, 0.85) * width * cfg.PIXEL_SPACING_MM_X
        y_top_mm = rng.uniform(0.03, 0.15) * height * cfg.PIXEL_SPACING_MM_Y
        sag_mm = rng.uniform(10.0, 35.0)
        bead_r_mm = rng.uniform(1.2, 2.2)
        ts = np.linspace(0.0, 1.0, n_beads)
        for t in ts:
            bx = x0_mm + (x1_mm - x0_mm) * t
            by = y_top_mm + sag_mm * np.sin(np.pi * t)
            dist2 = (x_mm - bx) ** 2 + (y_mm - by) ** 2
            canvas = np.maximum(canvas, np.exp(-dist2 / (2 * bead_r_mm**2)))
    else:
        # Клипса: продолговатый эллипс в случайном месте мягких тканей.
        cx_mm = rng.uniform(0.25, 0.75) * width * cfg.PIXEL_SPACING_MM_X
        cy_mm = rng.uniform(0.1, 0.65) * height * cfg.PIXEL_SPACING_MM_Y
        major_mm = rng.uniform(8.0, 18.0)
        minor_mm = rng.uniform(1.5, 3.5)
        angle = rng.uniform(0, np.pi)
        dx = (x_mm - cx_mm) * np.cos(angle) + (y_mm - cy_mm) * np.sin(angle)
        dy = -(x_mm - cx_mm) * np.sin(angle) + (y_mm - cy_mm) * np.cos(angle)
        canvas = np.exp(-((dx / major_mm) ** 2 + (dy / minor_mm) ** 2) ** 2)

    intensity = rng.uniform(0.85, 1.0)
    result = np.maximum(pixels_normalized, np.clip(canvas, 0, 1) * intensity)
    return result.astype(np.float32)


# --------------------------------------------------------------------------- #
# Кропы захвата поля
# --------------------------------------------------------------------------- #


def crop_field(
    pixels_normalized: np.ndarray,
    rng: np.random.Generator,
    *,
    side: str | None = None,
) -> np.ndarray:
    """Срезать низ (гребни) или верх (Th12/рёбра) кадра.

    Имитация неправильно выбранного поля сканирования: кадр становится короче,
    анатомическая граница исчезает за краем.
    """
    side = side or rng.choice(["bottom", "top"])
    height = pixels_normalized.shape[0]
    cut_mm = rng.uniform(cfg.SPINE_FIELD_MARGIN_MM, 2.2 * cfg.SPINE_FIELD_MARGIN_MM)
    cut_px = int(round(cut_mm / cfg.PIXEL_SPACING_MM_Y))
    cut_px = min(cut_px, height // 3)
    if side == "bottom":
        return pixels_normalized[: height - cut_px].copy()
    return pixels_normalized[cut_px:].copy()
