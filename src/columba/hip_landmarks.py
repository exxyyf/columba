"""Этап 4, шаги 2–4: ориентиры бедра и сигналы для чекера укладки.

Классический детектор (без обучения), по образцу `spine_keypoints.py`, но с
трекингом связной компоненты бедренной кости (после ревью v1, см.
stages/stage_4.md): глобальные крайние точки силуэта по строке
цепляли таз/седалищную кость в том же ряду — вместо этого силуэт
ОТСЛЕЖИВАЕТСЯ построчно снизу вверх от компоненты, касающейся нижнего края
кадра (диафиз всегда уходит вниз), с ограничением на скорость роста ширины
по строке — это физически не даёт трекеру перескочить на несвязанный объект
(таз, инородные тела) в той же строке.

Порядок ориентиров: диафиз (нижняя полоса, Theil-Sen по трекнутому центру) ->
шейка/перешеек (минимум ширины трекнутой компоненты НАД зоной вертелов) ->
головка (локальный максимум ширины НАД перешейком, центроид кластера) ->
большой/малый вертел (локальный выступ трекнутого края ОТ ЛОКАЛЬНОЙ базовой
линии — медианы соседних строк с исключённым зазором вокруг самой точки — в
полосе МЕЖДУ шейкой и диафизом, сторона — по положению головки).

Сторона бедра не входит в геометрию детектора впрямую: медиальная/латеральная
сторона определяется относительно головки (`medial_sign`) без явного
отражения кадра. Единственное место, где сторона учитывается явно, — знак
`head_offset_mm` (сигнал приводится к виду левого бедра, инвариант 1.1/1.4).

Детектор читает ТОЛЬКО пиксели: сторона либо передаётся явно (из этапа 1),
либо определяется по пикселям через `regions.hip_side`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as cfg
from . import regions
from .spine_checkers import theil_sen_angle_deg
from .spine_keypoints import (
    STATUS_NOT_EVALUATED,
    STATUS_OK,
    iso_to_raw,
    mm_to_iso_px,
    raw_to_iso,
    resample_isotropic_np,
    smooth_1d,
    smooth_2d,
)

HIP_SIGNAL_KEYS: tuple[str, ...] = (
    "shaft_tilt_deg",
    "lesser_trochanter_prominence",
    "lesser_trochanter_corridor_dev",
    "neck_shaft_angle_deg",
    "head_offset_mm",
)


@dataclass
class HipLandmarks:
    """Результат детекции ориентиров на одном кадре бедра."""

    status: str
    reason: str = ""
    side: str | None = None
    head_center_xy: tuple[float, float] | None = None
    shaft_points_xy: list[tuple[float, float]] = field(default_factory=list)
    lesser_trochanter_xy: tuple[float, float] | None = None
    greater_trochanter_xy: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "side": self.side,
            "head_center_xy": self.head_center_xy,
            "shaft_points_xy": [list(p) for p in self.shaft_points_xy],
            "lesser_trochanter_xy": self.lesser_trochanter_xy,
            "greater_trochanter_xy": self.greater_trochanter_xy,
        }


def _not_evaluated(reason: str, side: str | None) -> HipLandmarks:
    return HipLandmarks(status=STATUS_NOT_EVALUATED, reason=reason, side=side)


# --------------------------------------------------------------------------- #
# Трекинг связной компоненты бедренной кости, снизу вверх
# --------------------------------------------------------------------------- #


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Непрерывные отрезки True в строке; включительные (start, end)."""
    xs = np.nonzero(row)[0]
    if xs.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x != prev + 1:
            runs.append((start, prev))
            start = x
        prev = x
    runs.append((start, prev))
    return runs


def track_femur_column(
    fg: np.ndarray, *, growth_px: float, max_missing_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Левый/правый край КОМПОНЕНТЫ, связанной с нижним краем кадра, по строкам.

    Трекинг снизу вверх: на каждом шаге окно поиска — текущий интервал
    ± `growth_px`; новый интервал — объединение всех отрезков строки,
    пересекающихся с окном, но рост интервала за шаг ограничен `growth_px`
    (не даёт трекеру перепрыгнуть на несвязанный объект в той же строке —
    таз/седалищную кость/инородное тело). До `max_missing_rows` пустых строк
    подряд (например, тонкая полоса маскирования) не обрывают трекинг —
    интервал переносится без изменений. NaN — строка вне отслеженного участка.
    """
    height, width = fg.shape
    left = np.full(height, np.nan)
    right = np.full(height, np.nan)

    y = height - 1
    while y >= 0 and not fg[y].any():
        y -= 1
    if y < 0:
        return left, right

    runs = _row_runs(fg[y])
    cur_lo, cur_hi = max(runs, key=lambda r: r[1] - r[0])
    left[y], right[y] = cur_lo, cur_hi
    missing = 0
    for yy in range(y - 1, -1, -1):
        runs = _row_runs(fg[yy])
        window_lo, window_hi = cur_lo - growth_px, cur_hi + growth_px
        overlapping = [r for r in runs if r[1] >= window_lo and r[0] <= window_hi]
        if not overlapping:
            missing += 1
            if missing > max_missing_rows:
                break
            continue
        missing = 0
        new_lo = min(r[0] for r in overlapping)
        new_hi = max(r[1] for r in overlapping)
        cur_lo = max(new_lo, cur_lo - growth_px)
        cur_hi = min(new_hi, cur_hi + growth_px)
        left[yy], right[yy] = cur_lo, cur_hi
    return left, right


def _band_rows(height: int, fractions: tuple[float, float]) -> tuple[int, int]:
    lo, hi = fractions
    return max(0, int(round(height * lo))), min(height, int(round(height * hi)))


def _theil_sen_line(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float] | None:
    """Наклон (dx/dy) и свободный член линии x(y), робастно (Theil-Sen)."""
    angle = theil_sen_angle_deg(ys, xs)
    if angle != angle:
        return None
    slope = float(np.tan(np.radians(angle)))
    intercept = float(np.median(xs - slope * ys))
    return slope, intercept


def _local_baseline(edge: np.ndarray, y: int, *, window_rows: int, gap_rows: int, lo: int, hi: int) -> float | None:
    """Медиана края в окрестности строки y, с исключённым зазором вокруг неё.

    `lo`/`hi` — границы допустимого диапазона строк (полуоткрытый [lo, hi)).
    Используется для измерения ЛОКАЛЬНОГО выступа вертела относительно
    типичного положения соседнего контура, а не относительно прямой всего
    диафиза (которая на уровне шейки уже не описывает контур кости).
    """
    idx = np.concatenate(
        [
            np.arange(max(lo, y - window_rows), max(lo, y - gap_rows)),
            np.arange(min(hi, y + gap_rows + 1), min(hi, y + window_rows + 1)),
        ]
    )
    values = edge[idx]
    values = values[values == values]
    if values.size < 3:
        return None
    return float(np.median(values))


# --------------------------------------------------------------------------- #
# Детектор
# --------------------------------------------------------------------------- #


def detect_hip_landmarks(pixels_normalized: np.ndarray, side: str | None = None) -> HipLandmarks:
    """Найти ориентиры бедра на нормализованном кадре.

    Возвращает координаты в ИСХОДНЫХ пикселях кадра (не изотропных, не
    отражённых). При вырожденном/малоконтрастном кадре или ненайденном
    диафизе — честный `not_evaluated` с причиной, без исключений. Головка и
    вертелы могут остаться `None` при неудаче (например, диафиз пойман, а
    шейка обрезана краем кадра) — это НЕ приводит к `not_evaluated` всего
    кадра, только к NaN у зависящих сигналов.
    """
    if pixels_normalized.ndim != 2 or min(pixels_normalized.shape) < 8:
        return _not_evaluated("кадр вырожден", side)
    if float(pixels_normalized.max() - pixels_normalized.min()) < cfg.HIP_MIN_DYNAMIC_RANGE:
        return _not_evaluated("кадр без контраста", side)

    if side is None:
        side = regions.hip_side(pixels_normalized).side

    raw_height = int(pixels_normalized.shape[0])
    iso = resample_isotropic_np(pixels_normalized)
    iso_smooth = smooth_2d(iso, mm_to_iso_px(cfg.HIP_SMOOTH_SIGMA_MM))
    height, width = iso_smooth.shape
    fg = iso_smooth > cfg.HIP_SILHOUETTE_FG_THRESHOLD

    left_edge, right_edge = track_femur_column(
        fg,
        growth_px=mm_to_iso_px(cfg.HIP_TRACK_GROWTH_MM),
        max_missing_rows=cfg.HIP_TRACK_MAX_MISSING_ROWS,
    )
    mid_edge = (left_edge + right_edge) / 2.0
    width_profile = right_edge - left_edge

    # --- диафиз: срединная линия ТРЕКНУТОЙ компоненты в нижней полосе ------ #
    shaft_y0, shaft_y1 = _band_rows(height, cfg.HIP_SHAFT_BAND_FRACTIONS)
    shaft_rows_all = np.array([y for y in range(shaft_y0, shaft_y1) if mid_edge[y] == mid_edge[y]])
    if len(shaft_rows_all) < cfg.HIP_SHAFT_MIN_ROWS:
        return _not_evaluated("диафиз не найден", side)
    # Фильтр по ширине: строки, где трекнутая ширина заметно отличается от
    # типичной ширины диафиза (контаминация тазом/вертельной зоной на входе
    # в полосу), не участвуют в Theil-Sen. Порядок: типичная ширина — медиана
    # ПО ВСЕЙ полосе (до фильтра), затем фильтр в обе стороны.
    typical_width = float(np.median(width_profile[shaft_rows_all]))
    lo_w, hi_w = cfg.HIP_SHAFT_WIDTH_TOLERANCE
    width_ok = (width_profile[shaft_rows_all] >= lo_w * typical_width) & (
        width_profile[shaft_rows_all] <= hi_w * typical_width
    )
    shaft_rows = shaft_rows_all[width_ok] if width_ok.sum() >= cfg.HIP_SHAFT_MIN_ROWS else shaft_rows_all
    shaft_mid = mid_edge[shaft_rows]
    if len(shaft_mid) >= 3:
        shaft_mid = smooth_1d(shaft_mid, mm_to_iso_px(cfg.HIP_SHAFT_SMOOTH_MM))
    line = _theil_sen_line(shaft_rows.astype(np.float64), shaft_mid)
    if line is None:
        return _not_evaluated("диафиз слишком короткий для Theil-Sen", side)
    slope, intercept = line

    def x_shaft(y: float) -> float:
        return slope * y + intercept

    n_points = min(cfg.HIP_SHAFT_N_POINTS, len(shaft_rows))
    sample_idx = np.linspace(0, len(shaft_rows) - 1, n_points).astype(int)
    shaft_points_iso = np.stack([shaft_mid[sample_idx], shaft_rows[sample_idx].astype(np.float64)], axis=1)
    shaft_points_raw = iso_to_raw(shaft_points_iso, raw_height)
    shaft_points_xy = [(round(float(x), 1), round(float(y), 1)) for x, y in shaft_points_raw]

    # --- зона вертелов: самая широкая строка трекнутой компоненты выше диафиза #
    # На реальных кадрах бедра силуэт по ЛЮБОМУ разумному порогу яркости часто
    # без видимого разрыва переходит в таз/вертлужную впадину выше головки
    # (проекционное наложение, не брак трекинга) — поэтому "перешеек шейки"
    # НЕ ищется профилем ширины (ненадёжно, см. ревью v1: попытка искала
    # локальный минимум и либо ловила точку обрыва трекинга, либо просто не
    # находила разворота на реальных кадрах). Вместо этого: сторону
    # медиальный/латеральный определяем по АСИММЕТРИИ выступа в самой широкой
    # строке (большой вертел почти всегда доминирующий выступ), а голову ищем
    # ЯРКОСТНЫМ методом в медиальном коридоре ограниченной ширины — это не
    # даёт головке "утечь" в латеральный таз, даже если тот ярче.
    search_hi = shaft_y0  # эксклюзивно, строки [0, shaft_y0)
    proximal_rows = np.array([y for y in range(0, search_hi) if width_profile[y] == width_profile[y]])
    head_center_xy: tuple[float, float] | None = None
    medial_sign = 0.0
    troch_peak_row: int | None = None
    if proximal_rows.size >= cfg.HIP_NECK_MIN_ROWS:
        troch_peak_row = int(proximal_rows[np.argmax(width_profile[proximal_rows])])
        left_ext = x_shaft(troch_peak_row) - left_edge[troch_peak_row]
        right_ext = right_edge[troch_peak_row] - x_shaft(troch_peak_row)
        if left_ext != right_ext:
            # Больший выступ — латеральная сторона (большой вертел); медиальная
            # (к головке) — противоположная.
            medial_sign = 1.0 if left_ext > right_ext else -1.0

        if medial_sign != 0.0:
            # --- головка: яркостный поиск в ограниченном медиальном коридоре -- #
            margin_rows = int(round(mm_to_iso_px(cfg.HIP_HEAD_SEARCH_MARGIN_MM)))
            head_hi = max(0, troch_peak_row - margin_rows)
            if head_hi >= cfg.HIP_NECK_MIN_ROWS:
                offset_lo = mm_to_iso_px(cfg.HIP_HEAD_MEDIAL_OFFSET_MM[0])
                offset_hi = mm_to_iso_px(cfg.HIP_HEAD_MEDIAL_OFFSET_MM[1])
                head_pixels: list[tuple[int, int, float]] = []  # (y, x, value)
                for y in range(0, head_hi):
                    center = x_shaft(y)
                    if medial_sign > 0:
                        x_lo, x_hi = center + offset_lo, center + offset_hi
                    else:
                        x_lo, x_hi = center - offset_hi, center - offset_lo
                    x_lo_i, x_hi_i = max(0, int(round(x_lo))), min(width, int(round(x_hi)) + 1)
                    if x_hi_i <= x_lo_i:
                        continue
                    row_slice = iso_smooth[y, x_lo_i:x_hi_i]
                    for local_x, value in enumerate(row_slice):
                        head_pixels.append((y, x_lo_i + local_x, float(value)))
                if head_pixels:
                    values = np.array([p[2] for p in head_pixels])
                    threshold = np.percentile(values, cfg.HIP_HEAD_PERCENTILE)
                    bright = [p for p in head_pixels if p[2] >= threshold]
                    if len(bright) >= cfg.HIP_HEAD_MIN_PIXELS:
                        ys_h = np.array([p[0] for p in bright], dtype=np.float64)
                        xs_h = np.array([p[1] for p in bright], dtype=np.float64)
                        head_x_iso, head_y_iso = float(xs_h.mean()), float(ys_h.mean())
                        head_point_raw = iso_to_raw(np.array([[head_x_iso, head_y_iso]]), raw_height)[0]
                        head_center_xy = (
                            round(float(head_point_raw[0]), 1),
                            round(float(head_point_raw[1]), 1),
                        )

    # --- вертелы: локальный выступ трекнутого края от ЛОКАЛЬНОЙ базы ------- #
    # Зона поиска — окно вокруг самой широкой строки (там сидят оба вертела).
    greater_trochanter_xy: tuple[float, float] | None = None
    lesser_trochanter_xy: tuple[float, float] | None = None
    if medial_sign != 0.0 and troch_peak_row is not None:
        zone_half = int(round(mm_to_iso_px(cfg.HIP_TROCHANTER_ZONE_HALF_MM)))
        troch_lo = max(0, troch_peak_row - zone_half)
        troch_hi = min(shaft_y0, troch_peak_row + zone_half)
        window_rows = int(round(mm_to_iso_px(cfg.HIP_TROCHANTER_BASELINE_WINDOW_MM)))
        gap_rows = int(round(mm_to_iso_px(cfg.HIP_TROCHANTER_BASELINE_GAP_MM)))
        lateral_edge = left_edge if medial_sign > 0 else right_edge
        medial_edge = right_edge if medial_sign > 0 else left_edge

        def best_local_bump(edge: np.ndarray, sign: float) -> tuple[int, float, float] | None:
            """(строка, x края, величина выступа), выступ = sign*(edge - baseline)."""
            best = None
            for y in range(troch_lo, troch_hi):
                value = edge[y]
                if value != value:
                    continue
                baseline = _local_baseline(
                    edge, y, window_rows=window_rows, gap_rows=gap_rows, lo=troch_lo, hi=troch_hi
                )
                if baseline is None:
                    continue
                prominence = sign * (value - baseline)
                if best is None or prominence > best[2]:
                    best = (y, value, prominence)
            return best

        greater = best_local_bump(lateral_edge, -1.0 if medial_sign > 0 else 1.0)
        if greater is not None and greater[2] >= mm_to_iso_px(cfg.HIP_TROCHANTER_MIN_EXTENT_MM):
            y_g, x_g, _ = greater
            point_raw = iso_to_raw(np.array([[x_g, float(y_g)]]), raw_height)[0]
            greater_trochanter_xy = (round(float(point_raw[0]), 1), round(float(point_raw[1]), 1))

        lesser = best_local_bump(medial_edge, 1.0 if medial_sign > 0 else -1.0)
        if lesser is not None:
            y_l, x_l, _ = lesser
            point_raw = iso_to_raw(np.array([[x_l, float(y_l)]]), raw_height)[0]
            lesser_trochanter_xy = (round(float(point_raw[0]), 1), round(float(point_raw[1]), 1))

    return HipLandmarks(
        status=STATUS_OK,
        side=side,
        head_center_xy=head_center_xy,
        shaft_points_xy=shaft_points_xy,
        lesser_trochanter_xy=lesser_trochanter_xy,
        greater_trochanter_xy=greater_trochanter_xy,
    )


# --------------------------------------------------------------------------- #
# Сигналы для чекера укладки/ротации
# --------------------------------------------------------------------------- #


def _nan_signals() -> dict[str, float]:
    return {key: float("nan") for key in HIP_SIGNAL_KEYS}


def _lesser_trochanter_prominence_mm(pixels_normalized: np.ndarray, landmarks: HipLandmarks) -> float:
    """Пересчёт локального выступа малого вертела по сохранённой точке.

    `HipLandmarks` хранит только координаты точек (контракт), не массивы
    контура — поэтому здесь силуэт и трекинг компоненты считаются заново
    (детерминированная чистая функция от пикселей, тот же путь, что в
    `detect_hip_landmarks`), а выступ измеряется от ЛОКАЛЬНОЙ базы (медиана
    соседних строк медиального края, с исключённым зазором), а не от прямой
    диафиза — иначе значение включает всю полуширину шейки (см. ревью).
    """
    raw_height = int(pixels_normalized.shape[0])
    iso = resample_isotropic_np(pixels_normalized)
    iso_smooth = smooth_2d(iso, mm_to_iso_px(cfg.HIP_SMOOTH_SIGMA_MM))
    height, width = iso_smooth.shape
    fg = iso_smooth > cfg.HIP_SILHOUETTE_FG_THRESHOLD
    left_edge, right_edge = track_femur_column(
        fg, growth_px=mm_to_iso_px(cfg.HIP_TRACK_GROWTH_MM), max_missing_rows=cfg.HIP_TRACK_MAX_MISSING_ROWS
    )

    lesser_iso = raw_to_iso(np.array([landmarks.lesser_trochanter_xy]), raw_height)[0]
    y_l = int(round(float(lesser_iso[1])))
    x_l = float(lesser_iso[0])
    shaft_y0, _ = _band_rows(height, cfg.HIP_SHAFT_BAND_FRACTIONS)

    # Определяем, какой край (левый/правый) ближе к точке малого вертела —
    # это и есть медиальный край, независимо от прежнего расчёта стороны.
    dist_left = abs(x_l - left_edge[y_l]) if left_edge[y_l] == left_edge[y_l] else np.inf
    dist_right = abs(x_l - right_edge[y_l]) if right_edge[y_l] == right_edge[y_l] else np.inf
    medial_edge = left_edge if dist_left <= dist_right else right_edge

    window_rows = int(round(mm_to_iso_px(cfg.HIP_TROCHANTER_BASELINE_WINDOW_MM)))
    gap_rows = int(round(mm_to_iso_px(cfg.HIP_TROCHANTER_BASELINE_GAP_MM)))
    baseline = _local_baseline(
        medial_edge, y_l, window_rows=window_rows, gap_rows=gap_rows, lo=0, hi=shaft_y0
    )
    if baseline is None:
        return float("nan")
    prominence_px = abs(float(medial_edge[y_l]) - baseline)
    return prominence_px * cfg.ISO_PIXEL_MM


def compute_hip_signals(pixels_normalized: np.ndarray, landmarks: HipLandmarks) -> dict[str, float]:
    """Пять именованных сигналов ротации/укладки бедра, NaN — если не вычислим.

    Все геометрические величины — через изотропное пространство (шаг
    ISO_PIXEL_MM по обеим осям), поэтому расстояния и углы уже учитывают
    анизотропию пикселя (инвариант 1.1).
    """
    if landmarks.status != STATUS_OK:
        return _nan_signals()

    raw_height = int(pixels_normalized.shape[0])
    raw_width = int(pixels_normalized.shape[1])
    signals = _nan_signals()

    shaft_points_raw = np.asarray(landmarks.shaft_points_xy, dtype=np.float64)
    shaft_line = None
    if len(shaft_points_raw) >= 2:
        shaft_iso = raw_to_iso(shaft_points_raw, raw_height)
        angle = theil_sen_angle_deg(shaft_iso[:, 1], shaft_iso[:, 0])
        if angle == angle:
            signals["shaft_tilt_deg"] = round(abs(angle), 2)
            slope = float(np.tan(np.radians(angle)))
            intercept = float(np.median(shaft_iso[:, 0] - slope * shaft_iso[:, 1]))
            shaft_line = (slope, intercept)

    head_iso = None
    if landmarks.head_center_xy is not None:
        head_iso = raw_to_iso(np.array([landmarks.head_center_xy]), raw_height)[0]
        offset_iso_px = float(head_iso[0]) - (raw_width - 1) / 2.0
        offset_mm = offset_iso_px * cfg.ISO_PIXEL_MM
        if landmarks.side == "right":
            offset_mm = -offset_mm  # приведение знака к виду левого бедра
        signals["head_offset_mm"] = round(offset_mm, 1)

    if landmarks.lesser_trochanter_xy is not None:
        prominence_mm = _lesser_trochanter_prominence_mm(pixels_normalized, landmarks)
        if prominence_mm == prominence_mm:
            signals["lesser_trochanter_prominence"] = round(prominence_mm, 1)
            low, high = cfg.HIP_LESSER_TROCHANTER_CORRIDOR_MM
            dev = max(0.0, low - prominence_mm, prominence_mm - high)
            signals["lesser_trochanter_corridor_dev"] = round(dev, 1)

    if head_iso is not None and shaft_line is not None and len(shaft_points_raw) >= 1:
        slope, intercept = shaft_line
        shaft_iso_all = raw_to_iso(shaft_points_raw, raw_height)
        neck_anchor = shaft_iso_all[np.argmin(shaft_iso_all[:, 1])]  # ближайшая к головке точка диафиза
        neck_vec = np.array([float(head_iso[0]) - neck_anchor[0], float(head_iso[1]) - neck_anchor[1]])
        shaft_vec = np.array([slope, 1.0])
        neck_norm = float(np.linalg.norm(neck_vec))
        shaft_norm = float(np.linalg.norm(shaft_vec))
        if neck_norm > 0 and shaft_norm > 0:
            cosang = float(np.clip(np.dot(neck_vec, shaft_vec) / (neck_norm * shaft_norm), -1.0, 1.0))
            signals["neck_shaft_angle_deg"] = round(float(np.degrees(np.arccos(cosang))), 1)

    return signals


# --------------------------------------------------------------------------- #
# Этап 5, шаг 4: сигналы чекера области интереса (`hip_roi`)
# --------------------------------------------------------------------------- #

HIP_ROI_SIGNAL_KEYS: tuple[str, ...] = (
    "field_height_deficit",
    "rows_px",
    "top_margin_px",
    "bottom_margin_px",
    "side_margin_px",
)


def compute_hip_roi_signals(pixels_normalized: np.ndarray, landmarks: HipLandmarks) -> dict[str, float]:
    """Сигналы отступов поля от ориентиров (этап 5) — по образцу
    `spine_checkers.check_spine_positioning`'s `crest_deficit`/`top_margin_deficit`,
    но БЕЗ отсечения снизу нулём (важное отличие, см. ниже).

    `field_height_deficit` — решающий сигнал (калибровка `hip_eval`, шаг 4):
    `1 - rows_px/REF`, REF — `config.HIP_ROI_FIELD_HEIGHT_REF_PX` (best-F1
    порог nested CV на реальных данных). Положительный — кадр короче
    референса (подозрительно), отрицательный — длиннее (запас поля), НЕ
    отсекается нулём в отличие от `spine_positioning.crest_deficit`: почти
    все train-негативы (116 из 122) имеют `rows_px >= REF` — отсечение
    схлопывало бы их скор в одну точку (0.0), а `quality_prob` агрегатора
    (сигмоида вокруг порога) давала бы им всем одинаковые ~0.5 вместо
    уверенного «не нарушение» (см. stages/stage_5.md). Считается по
    одним ИСХОДНЫМ пикселям кадра (высота растра) — ориентиры не нужны,
    доступен даже когда `detect_hip_landmarks` не нашёл диафиз/головку (в
    отличие от `compute_hip_signals`, целиком завязанного на
    `landmarks.status == ok`). Остальные сигналы — margin'ы от конкретных
    ориентиров, NaN при их отсутствии, логируются только для объяснимости
    (во флаг не входят, слишком мало train-позитивов для надёжного выбора
    многомерного правила — см. stages/stage_5.md).
    """
    raw_height = int(pixels_normalized.shape[0])
    raw_width = int(pixels_normalized.shape[1])
    ref = cfg.HIP_ROI_FIELD_HEIGHT_REF_PX
    signals: dict[str, float] = {
        "rows_px": float(raw_height),
        "field_height_deficit": round(1.0 - raw_height / ref, 3),
        "top_margin_px": float("nan"),
        "bottom_margin_px": float("nan"),
        "side_margin_px": float("nan"),
    }
    if landmarks.status != STATUS_OK:
        return signals
    if landmarks.head_center_xy is not None:
        signals["top_margin_px"] = round(float(landmarks.head_center_xy[1]), 1)
    if landmarks.shaft_points_xy:
        lowest_y = max(point[1] for point in landmarks.shaft_points_xy)
        signals["bottom_margin_px"] = round(float(raw_height - lowest_y), 1)
    if landmarks.greater_trochanter_xy is not None:
        x = landmarks.greater_trochanter_xy[0]
        signals["side_margin_px"] = round(float(min(x, raw_width - x)), 1)
    return signals


# --------------------------------------------------------------------------- #
# Этап 9, п. 9.5: синтетические примеры нарушенных отступов поля
# --------------------------------------------------------------------------- #


def synth_crop_hip_field(
    pixels_normalized: np.ndarray, rng: np.random.Generator, *, edge: str | None = None
) -> np.ndarray:
    """Обрезать один край кадра бедра, имитируя неправильно выбранное поле
    сканирования («кропы качественных снимков с нарушенными отступами
    поля», см. stages/stage_9.md, раздел 9.5).

    По образцу `spine_synth.crop_field`, но для четырёх краёв (там только
    верх/низ) и с hip-специфичными масштабами: `top`/`bottom` режут по Y на
    `HIP_ROI_TOP_MARGIN_MM` (протокольные «3 см»), `left`/`right` — по X на
    `HIP_ROI_SIDE_MARGIN_MM` («2 см»), с запасом (1.0-1.8x), чтобы результат
    гарантированно нарушал соответствующий протокольный отступ. Используется
    ТОЛЬКО как источник дополнительных калибровочных примеров (проверка
    сигналов `top_margin_px`/`side_margin_px`/`rows_px` на заведомо
    нарушенном поле) — валидация чекера остаётся на реальной разметке
    (`stage_9_decisions.md`, раздел 9.5), синтетика в неё не подмешивается.
    """
    edge = edge or str(rng.choice(["top", "bottom", "left", "right"]))
    height, width = pixels_normalized.shape[:2]
    if edge in ("top", "bottom"):
        cut_mm = rng.uniform(cfg.HIP_ROI_TOP_MARGIN_MM, 1.8 * cfg.HIP_ROI_TOP_MARGIN_MM)
        cut_px = int(round(cut_mm / cfg.PIXEL_SPACING_MM_Y))
        cut_px = min(cut_px, height // 3)
        if edge == "bottom":
            return pixels_normalized[: height - cut_px].copy()
        return pixels_normalized[cut_px:].copy()
    if edge in ("left", "right"):
        cut_mm = rng.uniform(cfg.HIP_ROI_SIDE_MARGIN_MM, 1.8 * cfg.HIP_ROI_SIDE_MARGIN_MM)
        cut_px = int(round(cut_mm / cfg.PIXEL_SPACING_MM_X))
        cut_px = min(cut_px, width // 3)
        if edge == "left":
            return pixels_normalized[:, cut_px:].copy()
        return pixels_normalized[:, : width - cut_px].copy()
    raise ValueError(f"неизвестный край: {edge!r}")


# --------------------------------------------------------------------------- #
# Файл разметки
# --------------------------------------------------------------------------- #

ANNOTATION_VERSION = 1


def save_annotations(annotations: dict, path: Path | str = cfg.HIP_LANDMARKS_JSON) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": ANNOTATION_VERSION, "images": annotations}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_annotations(path: Path | str = cfg.HIP_LANDMARKS_JSON) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != ANNOTATION_VERSION:
        raise ValueError(f"неизвестная версия разметки: {payload.get('version')}")
    return payload["images"]


_POINT_KEYS = ("head_center", "lesser_trochanter", "greater_trochanter")


def validate_annotation_entry(entry: dict) -> list[str]:
    """Проверка схемы одной записи разметки; возвращает список проблем."""
    problems: list[str] = []
    if entry.get("side") not in ("left", "right", None):
        problems.append(f"side: неожиданное значение {entry.get('side')!r}")
    shaft = entry.get("shaft_points")
    if not isinstance(shaft, list) or len(shaft) < 2:
        problems.append("shaft_points: нужен список из >=2 точек")
    else:
        for point in shaft:
            if not (isinstance(point, list) and len(point) == 2):
                problems.append(f"shaft_points: точка не пара {point!r}")
    for key in _POINT_KEYS:
        value = entry.get(key)
        if value is not None and not (isinstance(value, list) and len(value) == 2):
            problems.append(f"{key}: ожидается пара [x, y] или null")
    if entry.get("reviewed") not in (True, False):
        problems.append("reviewed: ожидается bool")
    return problems
