"""Шаг 4: черновое определение региона и латеральности бедра.

Регион — по ширине кадра (эвристика подтверждена организаторами).
Сторона бедра — по взаимному положению головки/вертлужной впадины и диафиза:
у правого бедра головка медиальнее диафиза и лежит ПРАВЕЕ него на изображении,
у левого — левее. Признак относительный, поэтому не зависит от кадрирования.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    HIP_SIDE_MIN_ABS_SCORE,
    MASK_STEP_MIN_DEPTH,
    MASK_STEP_MIN_RUN,
    REGION_HIP,
    REGION_UNKNOWN,
    WIDTH_TO_REGION,
    ZONE_HIP_LEFT,
    ZONE_HIP_RIGHT,
)

SIDE_RIGHT = "right"
SIDE_LEFT = "left"

HEAD_BAND = 0.45  # верхняя доля кадра, где ищем головку/впадину
SHAFT_BAND = 0.80  # нижняя доля кадра, где ищем диафиз
HEAD_PERCENTILE = 97.0
SHAFT_PERCENTILE = 92.0


def classify_region(columns: int | None) -> str:
    """Регион по ширине кадра: 300 px — позвоночник, 280 px — бедро."""
    if columns is None:
        return REGION_UNKNOWN
    return WIDTH_TO_REGION.get(int(columns), REGION_UNKNOWN)


def classify_region_from_pixels(pixels) -> str:
    """То же, но ширина берётся из самого массива, а не из тега.

    В закрытом тесте дополнительных тегов и меток зоны не будет, поэтому
    регион определяется исключительно по изображению. Тег `Columns` на нашей
    выгрузке совпадает с формой массива во всех 499 файлах, но опираться на
    него в инференсе незачем.
    """
    array = np.asarray(pixels)
    if array.ndim < 2:
        return REGION_UNKNOWN
    return classify_region(int(array.shape[1]))


@dataclass(frozen=True)
class HipSide:
    side: str | None
    score: float
    confident: bool


def hip_side(pixels: np.ndarray) -> HipSide:
    """Определить сторону бедра.

    score = x(головка) - x(диафиз), обе координаты нормированы на ширину.
    score > 0 -> правое бедро, score < 0 -> левое.
    """
    array = np.asarray(pixels, dtype=np.float64)
    height, width = array.shape[:2]

    top = array[: max(1, int(height * HEAD_BAND))]
    bottom = array[int(height * SHAFT_BAND) :]
    if top.size == 0 or bottom.size == 0:
        return HipSide(None, float("nan"), False)

    head_x = _bright_centroid_x(top, HEAD_PERCENTILE, width)
    shaft_x = _bright_centroid_x(bottom, SHAFT_PERCENTILE, width)
    if np.isnan(head_x) or np.isnan(shaft_x):
        return HipSide(None, float("nan"), False)

    score = float(head_x - shaft_x)
    side = SIDE_RIGHT if score > 0 else SIDE_LEFT
    return HipSide(side, score, abs(score) >= HIP_SIDE_MIN_ABS_SCORE)


def _bright_centroid_x(band: np.ndarray, percentile: float, width: int) -> float:
    threshold = np.percentile(band, percentile)
    xs = np.nonzero(band >= threshold)[1]
    if xs.size == 0:
        return float("nan")
    return float(xs.mean()) / width


def zone_key(region: str, side: str | None) -> str | None:
    """Зона таблицы «Калибровка», которой соответствует снимок."""
    if region == REGION_HIP:
        if side == SIDE_RIGHT:
            return ZONE_HIP_RIGHT
        if side == SIDE_LEFT:
            return ZONE_HIP_LEFT
        return None
    if region == REGION_UNKNOWN:
        return None
    return region  # spine


def resolve_sides_within_study(scores: list[float]) -> list[str | None]:
    """Развести две проекции бедра одного исследования по сторонам.

    Внутри исследования признак используется в относительном виде: снимок с
    большим score — правое бедро, с меньшим — левое. Это устойчивее, чем
    абсолютный порог, когда оба значения оказались близко к нулю.
    """
    finite = [s for s in scores if s == s]  # отсеиваем NaN
    if len(scores) == 2 and len(finite) == 2:
        first, second = scores
        if first == second:
            return [None, None]
        return [SIDE_RIGHT, SIDE_LEFT] if first > second else [SIDE_LEFT, SIDE_RIGHT]
    return [None if s != s else (SIDE_RIGHT if s > 0 else SIDE_LEFT) for s in scores]


# --------------------------------------------------------------------------- #
# Чёрные прямоугольники маскирования
# --------------------------------------------------------------------------- #


def detect_mask_steps(
    pixels: np.ndarray,
    *,
    min_run: int = MASK_STEP_MIN_RUN,
    min_depth: int = MASK_STEP_MIN_DEPTH,
) -> list[dict[str, int | str]]:
    """Найти прямоугольные чёрные вырезы по краям кадра.

    Маскирование выглядит как идеально прямой вырез в силуэте: подряд идущие
    строки (или столбцы) с ОДИНАКОВЫМ отступом непустых пикселей от края,
    заметно большим медианного отступа по этой же стороне.
    """
    array = np.asarray(pixels)
    nonzero = array > 0
    height, width = array.shape[:2]

    rows_any = nonzero.any(axis=1)
    cols_any = nonzero.any(axis=0)
    if not rows_any.any() or not cols_any.any():
        return []

    left = np.where(rows_any, np.argmax(nonzero, axis=1), -1)
    right = np.where(rows_any, np.argmax(nonzero[:, ::-1], axis=1), -1)
    top = np.where(cols_any, np.argmax(nonzero, axis=0), -1)
    bottom = np.where(cols_any, np.argmax(nonzero[::-1, :], axis=0), -1)

    found: list[dict[str, int | str]] = []
    for edge, profile in (("left", left), ("right", right), ("top", top), ("bottom", bottom)):
        found.extend(_edge_steps(edge, profile, min_run=min_run, min_depth=min_depth))
    return found


def _edge_steps(edge: str, profile: np.ndarray, *, min_run: int, min_depth: int) -> list[dict[str, int | str]]:
    valid = profile[profile >= 0]
    if valid.size < min_run:
        return []
    baseline = float(np.median(valid))

    steps: list[dict[str, int | str]] = []
    index, length = 0, len(profile)
    while index < length:
        end = index
        while end + 1 < length and profile[end + 1] == profile[index]:
            end += 1
        run = end - index + 1
        depth = float(profile[index]) - baseline
        if run >= min_run and profile[index] >= 0 and depth >= min_depth:
            steps.append(
                {
                    "edge": edge,
                    "start": int(index),
                    "end": int(end),
                    "inset": int(profile[index]),
                    "depth": int(round(depth)),
                    "area_px": int(round(run * depth)),
                }
            )
        index = end + 1
    return steps
