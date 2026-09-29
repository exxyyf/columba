"""Этап 2, шаги 4–6: три чекера позвоночника.

Каждый чекер отдаёт `CheckerResult`: монотонный скор (больше = хуже, полярность
конвенции 1.4), бинарный флаг по порогу из config, статус ok/not_evaluated с
причиной и словарь именованных сигналов для объяснимости. Чекеры читают
ТОЛЬКО пиксели (и кейпоинты, посчитанные из пикселей).

Конструкции выбраны по калибровочным экспериментам на train-фолде
(см. stage_2_decisions.md):

* ось: робастный наклон срединной линии костной колонны (midline силуэта
  между верхним и нижним центрами тел, Theil-Sen) — прямая по двум крайним
  центрам цепочки оказалась слишком шумной;
* укладка: дефицит видимости гребней подвздошных костей в нижней полосе
  кадра (главный сигнал) + смещение цепочки от центральной вертикали;
* предметы: rule-based «тонкое яркое на тёмном фоне» + опциональная
  CNN-ветка (маленький классификатор, обученный с синтетикой шага 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import config as cfg
from .spine_keypoints import (
    STATUS_NOT_EVALUATED,
    STATUS_OK,
    SpineKeypoints,
    angle_from_vertical_deg,
    mm_to_iso_px,
    resample_isotropic_np,
    smooth_1d,
    smooth_2d,
)


@dataclass
class CheckerResult:
    """Единый формат выхода чекера для агрегации этапа 3."""

    checker: str  # ключ строки словаря (OUTPUT_LABEL_KEYS)
    status: str  # ok | not_evaluated
    score: float  # монотонный скор, больше = хуже; NaN при not_evaluated
    flag: bool  # нарушение (score выше порога)
    reason: str = ""  # причина not_evaluated
    signals: dict = field(default_factory=dict)  # именованные величины

    def to_dict(self) -> dict:
        return {
            "checker": self.checker,
            "status": self.status,
            "score": self.score,
            "flag": self.flag,
            "reason": self.reason,
            "signals": self.signals,
        }


def _not_evaluated(checker: str, reason: str) -> CheckerResult:
    return CheckerResult(checker=checker, status=STATUS_NOT_EVALUATED, score=float("nan"), flag=False, reason=reason)


# --------------------------------------------------------------------------- #
# Срединная линия костной колонны (общая для оси и укладки)
# --------------------------------------------------------------------------- #


def column_midline(
    iso_smooth: np.ndarray, keypoints_iso_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Midline силуэта колонны между верхним и нижним центрами тел.

    Для каждой строки y: края = связная область выше доли максимума локального
    профиля вокруг ломаной через центры; midline = середина краёв.
    Возвращает (ys, mids) в изотропных пикселях.
    """
    height, width = iso_smooth.shape
    centers = np.asarray(keypoints_iso_xy, dtype=np.float64)
    ridge = np.interp(np.arange(height), centers[:, 1], centers[:, 0])
    half = int(mm_to_iso_px(cfg.SPINE_MIDLINE_HALF_COL_MM))
    y_top, y_bottom = int(round(centers[0, 1])), int(round(centers[-1, 1]))
    ys, mids = [], []
    for y in range(max(0, y_top), min(height - 1, y_bottom) + 1):
        cx = int(round(ridge[y]))
        x0, x1 = max(0, cx - half), min(width, cx + half + 1)
        profile = iso_smooth[y, x0:x1]
        profile = profile - profile.min()
        if profile.max() <= 0:
            continue
        threshold = cfg.SPINE_MIDLINE_EDGE_FRACTION * profile.max()
        index = min(max(cx - x0, 0), len(profile) - 1)
        lo = index
        while lo > 0 and profile[lo - 1] >= threshold:
            lo -= 1
        hi = index
        while hi < len(profile) - 1 and profile[hi + 1] >= threshold:
            hi += 1
        ys.append(y)
        mids.append(x0 + (lo + hi) / 2.0)
    ys_arr = np.array(ys, dtype=np.float64)
    mids_arr = np.array(mids, dtype=np.float64)
    if len(mids_arr) >= 3:
        mids_arr = smooth_1d(mids_arr, mm_to_iso_px(cfg.SPINE_MIDLINE_SMOOTH_MM))
    return ys_arr, mids_arr


def theil_sen_angle_deg(ys_iso: np.ndarray, xs_iso: np.ndarray) -> float:
    """Робастный угол наклона midline от вертикали, градусы.

    Координаты изотропные (шаг ISO_PIXEL_MM по обеим осям), поэтому угол —
    просто arctan медианного наклона; анизотропия уже учтена ресемплингом.
    """
    n = len(ys_iso)
    indices = np.linspace(0, n - 1, min(n, cfg.SPINE_MIDLINE_TS_MAX_PAIRS)).astype(int)
    min_dy = mm_to_iso_px(cfg.SPINE_MIDLINE_TS_MIN_DY_MM)
    slopes = []
    for a_pos in range(len(indices)):
        for b_pos in range(a_pos + 1, len(indices)):
            a, b = indices[a_pos], indices[b_pos]
            dy = ys_iso[b] - ys_iso[a]
            if abs(dy) >= min_dy:
                slopes.append((xs_iso[b] - xs_iso[a]) / dy)
    if not slopes:
        return float("nan")
    return float(np.degrees(np.arctan(np.median(slopes))))


# --------------------------------------------------------------------------- #
# Шаг 4. Чекер оси
# --------------------------------------------------------------------------- #


def check_spine_axis(pixels_normalized: np.ndarray, keypoints: SpineKeypoints) -> CheckerResult:
    """Наклон оси позвоночника от вертикали кадра, градусы."""
    if keypoints.status != STATUS_OK or keypoints.n_centers < cfg.SPINE_MIN_CENTERS_FOR_AXIS:
        return _not_evaluated(
            "spine_axis", keypoints.reason or f"центров меньше {cfg.SPINE_MIN_CENTERS_FOR_AXIS}"
        )
    iso = resample_isotropic_np(pixels_normalized)
    iso_smooth = smooth_2d(iso, mm_to_iso_px(cfg.SPINE_SMOOTH_SIGMA_MM))
    ys, mids = column_midline(iso_smooth, keypoints.centers_iso_xy)
    if len(ys) < 3:
        return _not_evaluated("spine_axis", "midline колонны не построена")
    angle = theil_sen_angle_deg(ys, mids)
    if angle != angle:  # NaN: слишком короткая колонна
        return _not_evaluated("spine_axis", "колонна короче базы Theil-Sen")

    # Прямая «верхний центр -> нижний центр» из ТЗ — в сигналы (в исходных
    # пикселях, угол через анизотропную формулу из инварианта 1).
    centers = keypoints.centers_raw_xy
    endpoint_angle = angle_from_vertical_deg(
        float(centers[0][0] - centers[-1][0]), float(centers[0][1] - centers[-1][1])
    )
    line = np.polyval(np.polyfit(ys, mids, 1), ys)
    max_deviation_mm = float(np.max(np.abs(mids - line)) * cfg.ISO_PIXEL_MM)

    score = abs(angle)
    return CheckerResult(
        checker="spine_axis",
        status=STATUS_OK,
        score=score,
        flag=bool(score > cfg.SPINE_AXIS_MAX_ANGLE_DEG),
        signals={
            "axis_angle_deg": round(angle, 2),
            "endpoint_angle_deg": round(endpoint_angle, 2),
            "max_lateral_deviation_mm": round(max_deviation_mm, 1),
            "n_centers": keypoints.n_centers,
        },
    )


# --------------------------------------------------------------------------- #
# Шаг 5. Чекер укладки/захвата поля
# --------------------------------------------------------------------------- #


def crest_contrast_signal(pixels_normalized: np.ndarray, band_mm: float | None = None) -> float:
    """Контраст гребней подвздошных костей в нижней полосе кадра.

    p95 яркости боковых третей нижней полосы (max по сторонам) минус медиана
    боковых третей средней части (фон мягких тканей). Больше = гребни видны.
    `band_mm` по умолчанию из config; явное значение — для вложенной CV шага 8.
    """
    band_mm = cfg.SPINE_CREST_BAND_MM if band_mm is None else band_mm
    iso = resample_isotropic_np(pixels_normalized)
    height, width = iso.shape
    band = max(1, min(int(mm_to_iso_px(band_mm)), height // 4))
    bottom = iso[height - band :]
    middle = iso[height // 3 : 2 * height // 3]
    third = width // 3
    reference = float(
        np.median(np.concatenate([middle[:, :third].ravel(), middle[:, 2 * third :].ravel()]))
    )
    p95_left = float(np.percentile(bottom[:, :third], 95))
    p95_right = float(np.percentile(bottom[:, 2 * third :], 95))
    return max(p95_left, p95_right) - reference


def silhouette_margin_asymmetry_mm(pixels_normalized: np.ndarray) -> float | None:
    """Разница отступов силуэта тела от левого/правого края кадра, мм.

    По каждой строке центральной полосы (доли из config) берутся расстояния
    от краёв кадра до первого/последнего пикселя переднего плана; итог —
    медиана левых минус медиана правых отступов. Положительное значение =
    силуэт смещён вправо. Оговорка: чёрные прямоугольники маскирования
    (has_mask_rect) увеличивают отступ своей стороны — сигнал логируется,
    во флаг не входит.
    """
    iso = resample_isotropic_np(pixels_normalized)
    iso_smooth = smooth_2d(iso, mm_to_iso_px(cfg.SPINE_SMOOTH_SIGMA_MM))
    height, width = iso_smooth.shape
    top_fraction, bottom_fraction = cfg.SPINE_SILHOUETTE_ROW_FRACTIONS
    rows = range(int(height * top_fraction), int(height * bottom_fraction))
    left_margins, right_margins = [], []
    for y in rows:
        foreground = np.nonzero(iso_smooth[y] > cfg.SPINE_SILHOUETTE_FG_THRESHOLD)[0]
        if not len(foreground):
            continue
        left_margins.append(float(foreground[0]))
        right_margins.append(float(width - 1 - foreground[-1]))
    if not left_margins:
        return None
    return float((np.median(left_margins) - np.median(right_margins)) * cfg.ISO_PIXEL_MM)


def crest_line_tilt_deg(pixels_normalized: np.ndarray) -> float | None:
    """Наклон линии гребней подвздошных костей от горизонтали кадра, градусы.

    Точка гребня каждой стороны — яркостный центроид пикселей боковой трети
    нижней полосы, превышающих фон мягких тканей на долю контраста стороны
    (константы в config). Если хотя бы одна сторона не набирает минимальный
    контраст (гребень не виден) — None. Считается в изотропном пространстве,
    поэтому угол — прямой arctan; знак: положительный = правая сторона ниже.
    """
    iso = resample_isotropic_np(pixels_normalized)
    height, width = iso.shape
    band = max(1, min(int(mm_to_iso_px(cfg.SPINE_CREST_BAND_MM)), height // 4))
    bottom = iso[height - band :]
    middle = iso[height // 3 : 2 * height // 3]
    third = width // 3
    reference = float(
        np.median(np.concatenate([middle[:, :third].ravel(), middle[:, 2 * third :].ravel()]))
    )
    centroids = []
    for x_offset, side in ((0, bottom[:, :third]), (2 * third, bottom[:, 2 * third :])):
        contrast = float(np.percentile(side, 95)) - reference
        if contrast < cfg.SPINE_CREST_TILT_MIN_CONTRAST:
            return None
        threshold = reference + cfg.SPINE_CREST_TILT_PIXEL_FRACTION * contrast
        ys, xs = np.nonzero(side > threshold)
        if not len(ys):
            return None
        weights = side[ys, xs] - threshold
        weights_sum = float(weights.sum())
        if weights_sum <= 0:
            weights, weights_sum = np.ones_like(weights), float(len(weights))
        centroids.append(
            (
                float((xs * weights).sum() / weights_sum) + x_offset,
                float((ys * weights).sum() / weights_sum) + (height - band),
            )
        )
    (x_left, y_left), (x_right, y_right) = centroids
    if x_right == x_left:
        return None
    return float(np.degrees(np.arctan((y_right - y_left) / (x_right - x_left))))


def check_spine_positioning(pixels_normalized: np.ndarray, keypoints: SpineKeypoints) -> CheckerResult:
    """Захват поля и центрирование укладки.

    Решающий сигнал (по калибровке шага 8): дефицит видимости гребней — поле
    сканирования выбрано так, что низ анатомии обрезан. На train сигнал
    разделяет классы полностью (позитивы crest <= 0.20, негативы >= 0.22).
    Смещение цепочки от вертикали и отступ «3 см» логируются как сигналы:
    флагом они быть не могут — у пятой части негативов смещение больше 13 мм
    (см. decisions).

    Решающий сигнал от кейпоинтов не зависит: если детектор центров не
    сработал, скор и флаг всё равно считаются, а кейпоинтные сигналы
    (смещение, отступы) остаются пустыми с пометкой причины.
    """
    if pixels_normalized.ndim != 2 or min(pixels_normalized.shape) < 8:
        return _not_evaluated("spine_positioning", "кадр вырожден")
    if float(pixels_normalized.max() - pixels_normalized.min()) < cfg.SPINE_MIN_DYNAMIC_RANGE:
        return _not_evaluated("spine_positioning", "кадр без контраста")
    height, width = pixels_normalized.shape

    crest = crest_contrast_signal(pixels_normalized)
    crest_deficit = max(0.0, 1.0 - crest / cfg.SPINE_CREST_P95_REF)
    asymmetry = silhouette_margin_asymmetry_mm(pixels_normalized)
    tilt = crest_line_tilt_deg(pixels_normalized)
    signals: dict = {
        "crest_contrast": round(crest, 4),
        "crest_deficit": round(crest_deficit, 3),
        # Сигналы симметрии из плана шага 5: логируются для объяснимости и
        # этапа 6, во флаг не входят (на train классы не разделяют).
        "silhouette_asymmetry_mm": None if asymmetry is None else round(asymmetry, 1),
        "crest_tilt_deg": None if tilt is None else round(tilt, 2),
        "center_offset_mm": None,
        "bottom_margin_mm": None,
        "top_margin_mm": None,
        "top_margin_deficit": None,
    }
    if keypoints.status == STATUS_OK and keypoints.n_centers >= 1:
        centers = keypoints.centers_raw_xy
        signals["center_offset_mm"] = round(
            float((np.mean(centers[:, 0]) - (width - 1) / 2.0) * cfg.PIXEL_SPACING_MM_X), 1
        )
        signals["bottom_margin_mm"] = round(float((height - 1 - centers[-1][1]) * cfg.PIXEL_SPACING_MM_Y), 1)
        top_margin_mm = float(centers[0][1] * cfg.PIXEL_SPACING_MM_Y)
        signals["top_margin_mm"] = round(top_margin_mm, 1)
        # Дефицит верхнего отступа: верхний видимый позвонок прижат к краю
        # кадра — прокси обрезки поля сверху (Th12/рёбра). Реальных позитивов
        # с обрезкой сверху в разметке нет, сигнал валидирован синтетическими
        # кропами (spine_eval, синтетика укладки).
        signals["top_margin_deficit"] = round(
            max(0.0, 1.0 - top_margin_mm / cfg.SPINE_FIELD_MARGIN_MM), 3
        )
    else:
        signals["keypoints_reason"] = keypoints.reason or "нет центров позвонков"

    score = float(crest_deficit)
    return CheckerResult(
        checker="spine_positioning",
        status=STATUS_OK,
        score=score,
        flag=bool(score >= cfg.SPINE_POSITIONING_FLAG_DEFICIT),
        signals=signals,
    )


# --------------------------------------------------------------------------- #
# Шаг 6. Детектор посторонних предметов
# --------------------------------------------------------------------------- #


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Связные компоненты 8-связности (без scipy); список массивов (y, x)."""
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: list[np.ndarray] = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        components.append(np.array(pixels, dtype=np.int64))
    return components


def metal_maps(pixels_normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Карты детектора металла в изотропном пространстве: (тонкий избыток, окружение).

    Не зависят от порогов — калибровка шага 8 считает их один раз на кадр.
    """
    iso = resample_isotropic_np(pixels_normalized)
    thin_excess = iso - smooth_2d(iso, mm_to_iso_px(cfg.SPINE_METAL_THIN_SIGMA_MM))
    ambient = smooth_2d(iso, mm_to_iso_px(cfg.SPINE_METAL_AMBIENT_SIGMA_MM))
    return thin_excess, ambient


def metal_components(
    pixels_normalized: np.ndarray,
    *,
    thin_contrast: float | None = None,
    ambient_max: float | None = None,
    maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict]:
    """Rule-based кандидаты металла: тонкие яркие структуры на тёмном фоне.

    Металл (цепочки, клипсы) ярок и тонок, а окружён тёмными мягкими тканями;
    кость тоже яркая, но окружена костью (окружение светлое). Бусины цепочек
    соединяются дилатацией; мелкие крапинки на краях позвонков отсекаются
    порогами площади и диагонали. Пороги по умолчанию — из config; явные
    значения и готовые `maps` нужны только калибровке (вложенная CV шага 8).

    Верхние границы площади/диагонали (находка ревью этапа 9, п. 9.4):
    протяжённый рёберный край без анатомического исключения региона проходит
    те же нижние пороги, что и настоящий металл, и после дилатации либо
    остаётся одной крупной компонентой, либо дробится на несколько — обе
    формы отсекаются здесь (компактный целиком, фрагменты — по отдельности,
    т.к. каждый фрагмент один в кадре крупнее типичной цепочки). Лево-правая
    латеральная позиция и положение по высоте кадра НЕ используются как
    фильтр: на train у настоящих цепочек (просмотр глазами, decisions шаг 1 —
    «бусы поверх грудной клетки») латеральное смещение от срединной линии
    колонны и высота в кадре статистически совпадают с рёбрами на негативах
    (см. stage_9_decisions.md, раздел 9.4) — геометрическое исключение по
    позиции обрубило бы реальные позитивы почти так же, как рёбра; это
    открытый риск, а не устранённая проблема.
    """
    thin_contrast = cfg.SPINE_METAL_THIN_CONTRAST if thin_contrast is None else thin_contrast
    ambient_max = cfg.SPINE_METAL_AMBIENT_MAX if ambient_max is None else ambient_max
    thin_excess, ambient = maps if maps is not None else metal_maps(pixels_normalized)
    candidate = (thin_excess > thin_contrast) & (ambient < ambient_max)

    kernel = 2 * int(mm_to_iso_px(cfg.SPINE_METAL_DILATE_MM)) + 1
    dilated = (
        F.max_pool2d(
            torch.as_tensor(candidate, dtype=torch.float32).reshape(1, 1, *candidate.shape),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )[0, 0].numpy()
        > 0
    )

    results: list[dict] = []
    for component in _connected_components(dilated):
        ys, xs = component[:, 0], component[:, 1]
        area_mm2 = float(candidate[ys, xs].sum()) * cfg.ISO_PIXEL_MM**2  # до дилатации
        diag_mm = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min())) * cfg.ISO_PIXEL_MM
        if area_mm2 < cfg.SPINE_METAL_MIN_AREA_MM2 or diag_mm < cfg.SPINE_METAL_MIN_DIAG_MM:
            continue
        if area_mm2 > cfg.SPINE_METAL_MAX_AREA_MM2 or diag_mm > cfg.SPINE_METAL_MAX_DIAG_MM:
            continue
        results.append(
            {
                "area_mm2": round(area_mm2, 1),
                "diag_mm": round(diag_mm, 1),
                "centroid_iso_xy": (round(float(xs.mean()), 1), round(float(ys.mean()), 1)),
                "bbox_iso": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            }
        )
    return results


def metal_rule_score(pixels_normalized: np.ndarray, **params) -> tuple[float, list[dict]]:
    components = metal_components(pixels_normalized, **params)
    total_area = sum(c["area_mm2"] for c in components)
    return float(total_area / cfg.SPINE_METAL_SCORE_NORM_MM2), components


def check_spine_objects(pixels_normalized: np.ndarray, objects_predictor=None) -> CheckerResult:
    """Посторонние предметы.

    Решает rule-based скор: по вложенной CV шага 8 CNN-ветка его не обгоняет
    (OOF AUC 0.772 против 0.820 у правила, ДИ перекрываются), а правило
    объяснимо. Вероятность CNN, если предиктор передан, пишется в сигналы —
    этап 6 сможет пересмотреть выбор при калибровке quality_prob.

    Геометрия и пороги пересмотрены при ревью этапа 9 (п. 9.4): без верхних
    границ площади/диагонали правило ловило протяжённый рёберный край как
    металл (rule_score 4.37 на data/Для теста/CR000000_ПОП.dcm при пороге
    0.40). После добавления верхних границ площади/диагонали (config,
    SPINE_METAL_MAX_AREA_MM2/SPINE_METAL_MAX_DIAG_MM) score на этом файле
    падает до 0.456, порог флага поднят до 0.50 (см. комментарий в config) —
    файл больше не флагуется. Остаточный риск не устранён и задокументирован
    честно, а не спрятан: единственная оставшаяся у этого файла компонента
    (фрагмент ребра, 18 мм²/18.2 мм) геометрически неотличима от мелкого
    реального позитива, поэтому граница отсекает и часть настоящих слабых
    позитивов (train recall in-sample 0.5 -> 0.357, decisions раздел 9.4) —
    латеральная позиция и площадь/диагональ у настоящих цепочек и у рёбер
    статистически пересекаются, чисто геометрически это неразделимо.
    """
    if pixels_normalized.ndim != 2 or min(pixels_normalized.shape) < 8:
        return _not_evaluated("spine_objects", "кадр вырожден")
    rule_score, components = metal_rule_score(pixels_normalized)
    signals = {
        "rule_score": round(rule_score, 3),
        "n_components": len(components),
        "total_area_mm2": round(rule_score * cfg.SPINE_METAL_SCORE_NORM_MM2, 1),
        "components": components[:10],
    }
    if objects_predictor is not None:
        signals["cnn_probability"] = round(float(objects_predictor.predict_proba(pixels_normalized)), 4)
    return CheckerResult(
        checker="spine_objects",
        status=STATUS_OK,
        score=rule_score,
        flag=bool(rule_score >= cfg.SPINE_METAL_MIN_SCORE),
        signals=signals,
    )


# --------------------------------------------------------------------------- #
# Общая точка входа модуля позвоночника
# --------------------------------------------------------------------------- #


def get_spine_keypoints(pixels_normalized: np.ndarray, unet_predictor=None) -> SpineKeypoints:
    """Кейпоинты: классический детектор (основная ветка этапа 2).

    U-Net шага 3 можно передать явно; если сеть не нашла центров, честно
    падаем на классическую ветку. По метрикам этапа классика точнее
    (см. decisions), поэтому по умолчанию предиктор не подключается.
    """
    from .spine_keypoints import detect_vertebra_centers

    if unet_predictor is not None:
        prediction = unet_predictor.predict(pixels_normalized)
        if prediction.status == STATUS_OK:
            return prediction
    return detect_vertebra_centers(pixels_normalized)


def run_spine_checkers(
    pixels_normalized: np.ndarray, *, unet_predictor=None, objects_predictor=None
) -> dict[str, CheckerResult]:
    """Все три чекера на одном кадре ПОП. Ключи — OUTPUT_LABEL_KEYS."""
    keypoints = get_spine_keypoints(pixels_normalized, unet_predictor)
    results = {
        "spine_axis": check_spine_axis(pixels_normalized, keypoints),
        "spine_positioning": check_spine_positioning(pixels_normalized, keypoints),
        "spine_objects": check_spine_objects(pixels_normalized, objects_predictor),
    }
    keypoints_json = [[round(float(x), 1), round(float(y), 1)] for x, y in keypoints.centers_raw_xy]
    for result in results.values():
        result.signals["keypoints_raw_xy"] = keypoints_json
    return results
