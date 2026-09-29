"""Этап 2, шаги 2–3: ориентиры тел позвонков на кадре ПОП.

Здесь три вещи:

* преобразования координат исходные пиксели <-> изотропное пространство
  (та же схема ресемплинга, что у CNN этапа 1; инволюция закрыта тестом);
* классический детектор центров тел позвонков: хребтовая линия через
  динамическое программирование по яркости + минимумы профиля на
  межпозвонковых дисках. Работает без обучения и весов — это ветка честной
  деградации, когда весов нейросети нет, и источник кандидатов для разметки;
* формат файла разметки `artifacts/annotations/spine_keypoints.json`.

Все физические константы — в config (мм), переводятся через pixel spacing.
Детектор читает ТОЛЬКО пиксели: ни тегов, ни имён файлов.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as cfg

STATUS_OK = "ok"
STATUS_NOT_EVALUATED = "not_evaluated"


# --------------------------------------------------------------------------- #
# Координаты: исходные пиксели <-> изотропное пространство
# --------------------------------------------------------------------------- #
# Ресемплинг (region_cnn.resample_to_isotropic) растягивает ось Y интерполяцией
# `align_corners=False` до new_h = round(H * ANISOTROPY_Y_OVER_X). Центр
# выходного пикселя i лежит во входной координате (i + 0.5) / s - 0.5, где
# s = new_h / H. Отсюда прямое и обратное отображения координат точек.


def iso_height(raw_height: int) -> int:
    """Высота изотропного кадра для исходной высоты raw_height."""
    return int(round(raw_height * cfg.ANISOTROPY_Y_OVER_X))


def raw_to_iso(points_xy: np.ndarray, raw_height: int) -> np.ndarray:
    """(x, y) исходного кадра -> (x, y) изотропного. X не меняется."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2).copy()
    scale = iso_height(raw_height) / raw_height
    points[:, 1] = (points[:, 1] + 0.5) * scale - 0.5
    return points


def iso_to_raw(points_xy: np.ndarray, raw_height: int) -> np.ndarray:
    """(x, y) изотропного кадра -> (x, y) исходного. Инволюция с raw_to_iso."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2).copy()
    scale = iso_height(raw_height) / raw_height
    points[:, 1] = (points[:, 1] + 0.5) / scale - 0.5
    return points


def resample_isotropic_np(pixels_normalized: np.ndarray) -> np.ndarray:
    """Изотропный кадр как numpy (обёртка над ресемплингом этапа 1)."""
    from .region_cnn import resample_to_isotropic

    return resample_to_isotropic(pixels_normalized).numpy()


def angle_from_vertical_deg(dx_px: float, dy_px: float) -> float:
    """Наклон отрезка (в ИСХОДНЫХ пикселях) от вертикали кадра, градусы.

    Единственная точка пересчёта угла: анизотропия учитывается умножением
    смещений на физический шаг соответствующей оси (инвариант 1 этапа).
    Знак: положительный — верх отрезка правее низа.
    """
    return float(
        np.degrees(
            np.arctan2(dx_px * cfg.PIXEL_SPACING_MM_X, abs(dy_px) * cfg.PIXEL_SPACING_MM_Y)
        )
    )


def mm_to_iso_px(mm: float) -> float:
    """Миллиметры -> изотропные пиксели (шаг ISO_PIXEL_MM)."""
    return mm / cfg.ISO_PIXEL_MM


# --------------------------------------------------------------------------- #
# Вспомогательное сглаживание (без scipy)
# --------------------------------------------------------------------------- #


def _gaussian_kernel(sigma_px: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma_px)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma_px) ** 2)
    return kernel / kernel.sum()


def smooth_1d(values: np.ndarray, sigma_px: float) -> np.ndarray:
    kernel = _gaussian_kernel(sigma_px)
    radius = len(kernel) // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_2d(image: np.ndarray, sigma_px: float) -> np.ndarray:
    kernel = _gaussian_kernel(sigma_px)
    radius = len(kernel) // 2
    padded = np.pad(np.asarray(image, dtype=np.float64), ((radius, radius), (0, 0)), mode="edge")
    rows = np.apply_along_axis(np.convolve, 0, padded, kernel, "valid")
    padded = np.pad(rows, ((0, 0), (radius, radius)), mode="edge")
    return np.apply_along_axis(np.convolve, 1, padded, kernel, "valid")


# --------------------------------------------------------------------------- #
# Классический детектор центров тел позвонков
# --------------------------------------------------------------------------- #


@dataclass
class SpineKeypoints:
    """Результат детекции на одном кадре ПОП."""

    status: str
    reason: str = ""
    centers_raw_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    centers_iso_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    ridge_iso_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    profile: np.ndarray = field(default_factory=lambda: np.zeros(0))
    profile_peaks_iso_y: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_centers(self) -> int:
        return int(len(self.centers_raw_xy))


def ridge_path(iso_smooth: np.ndarray) -> np.ndarray:
    """Хребтовая линия: x(y), максимизирующая яркость вдоль пути.

    DP сверху вниз, шаг пути ±1 px по x на строку (в изотропном пространстве
    наклон до ~15 градусов укладывается с запасом), небольшой штраф за сдвиг
    удерживает линию от дрейфа на рёбра.
    """
    height, width = iso_smooth.shape
    margin = int(round(mm_to_iso_px(cfg.SPINE_RIDGE_MARGIN_MM)))
    margin = min(margin, (width - 3) // 2)
    band = iso_smooth[:, margin : width - margin]
    n = band.shape[1]

    penalty = cfg.SPINE_RIDGE_LATERAL_PENALTY
    score = np.full((height, n), -np.inf)
    move = np.zeros((height, n), dtype=np.int8)
    score[0] = band[0]
    for y in range(1, height):
        prev = score[y - 1]
        stay = prev
        left = np.concatenate(([-np.inf], prev[:-1] - penalty))
        right = np.concatenate((prev[1:] - penalty, [-np.inf]))
        choices = np.stack([stay, left, right])
        best = np.argmax(choices, axis=0)
        score[y] = band[y] + choices[best, np.arange(n)]
        move[y] = best

    path = np.zeros(height, dtype=np.int64)
    path[-1] = int(np.argmax(score[-1]))
    for y in range(height - 1, 0, -1):
        step = move[y, path[y]]
        path[y - 1] = path[y] - (1 if step == 1 else -1 if step == 2 else 0)
    return path + margin


def profile_center_maxima(profile: np.ndarray) -> np.ndarray:
    """Индексы центров тел позвонков на профиле вдоль хребтовой линии.

    Профиль детрендируется (минус низкочастотная составляющая), чтобы яркий
    крестец и тёмный верх кадра не влияли на пороги. Тело позвонка — светлая
    полоса поперёк линии: локальный максимум детрендированного профиля выше
    порога; при близких максимумах остаётся более высокий (шаг из config).
    Прямая детекция центров устойчивее сегментации по дискам: пропущенный
    диск не убивает два соседних центра.
    """
    trend = smooth_1d(profile, mm_to_iso_px(cfg.SPINE_PROFILE_DETREND_SIGMA_MM))
    detrended = profile - trend
    interior = np.arange(1, len(profile) - 1)
    is_max = (detrended[interior] >= detrended[interior - 1]) & (
        detrended[interior] >= detrended[interior + 1]
    )
    candidates = [i for i in interior[is_max] if detrended[i] >= cfg.SPINE_CENTER_MIN_HEIGHT]
    candidates.sort(key=lambda i: -detrended[i])  # сначала самые выраженные
    min_spacing = mm_to_iso_px(cfg.SPINE_CENTER_MIN_SPACING_MM)
    chosen: list[int] = []
    for i in candidates:
        if all(abs(i - j) >= min_spacing for j in chosen):
            chosen.append(i)
    return np.array(sorted(chosen), dtype=np.int64)


def detect_vertebra_centers(pixels_normalized: np.ndarray) -> SpineKeypoints:
    """Найти центры тел позвонков на нормализованном кадре ПОП.

    Возвращает центры в исходных пикселях (и изотропных — для отладки).
    При неудаче — честный `not_evaluated` с причиной, без исключений.
    """
    if pixels_normalized.ndim != 2 or min(pixels_normalized.shape) < 8:
        return SpineKeypoints(status=STATUS_NOT_EVALUATED, reason="кадр вырожден")
    raw_height = int(pixels_normalized.shape[0])
    iso = resample_isotropic_np(pixels_normalized)
    iso_smooth = smooth_2d(iso, mm_to_iso_px(cfg.SPINE_SMOOTH_SIGMA_MM))

    path = ridge_path(iso_smooth)

    half = int(round(mm_to_iso_px(cfg.SPINE_PROFILE_HALF_WIDTH_MM)))
    height, width = iso_smooth.shape
    profile = np.empty(height)
    for y in range(height):
        lo = max(0, path[y] - half)
        hi = min(width, path[y] + half + 1)
        profile[y] = iso_smooth[y, lo:hi].mean()
    profile = smooth_1d(profile, mm_to_iso_px(cfg.SPINE_PROFILE_SMOOTH_SIGMA_MM))

    maxima = profile_center_maxima(profile)
    min_brightness = cfg.SPINE_CENTER_MIN_BRIGHTNESS_FRACTION * float(np.median(profile))
    centers_iso: list[tuple[float, float]] = []
    for y_center in maxima:
        if profile[int(y_center)] < min_brightness:
            continue  # тусклый максимум: фон/украшение у края кадра, не кость
        centers_iso.append((float(path[int(y_center)]), float(y_center)))

    if len(centers_iso) < 1:
        return SpineKeypoints(
            status=STATUS_NOT_EVALUATED,
            reason="ни одного правдоподобного тела позвонка",
            ridge_iso_x=path.astype(np.float64),
            profile=profile,
        )

    centers_iso_arr = np.array(centers_iso, dtype=np.float64)
    centers_raw = iso_to_raw(centers_iso_arr, raw_height)
    return SpineKeypoints(
        status=STATUS_OK,
        centers_raw_xy=centers_raw,
        centers_iso_xy=centers_iso_arr,
        ridge_iso_x=path.astype(np.float64),
        profile=profile,
        profile_peaks_iso_y=maxima.astype(np.float64),
    )


# --------------------------------------------------------------------------- #
# Файл разметки
# --------------------------------------------------------------------------- #

ANNOTATION_VERSION = 1
ANNOTATION_SOURCE_AUTO = "auto"
ANNOTATION_SOURCE_REVIEWED = "auto+review"


def save_annotations(annotations: dict, path: Path | str = cfg.SPINE_KEYPOINTS_JSON) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": ANNOTATION_VERSION, "images": annotations}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_annotations(path: Path | str = cfg.SPINE_KEYPOINTS_JSON) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != ANNOTATION_VERSION:
        raise ValueError(f"неизвестная версия разметки: {payload.get('version')}")
    return payload["images"]


def validate_annotation_entry(entry: dict) -> list[str]:
    """Проверка схемы одной записи; возвращает список проблем."""
    problems = []
    centers = entry.get("centers_xy_raw")
    if not isinstance(centers, list) or len(centers) < 2:
        problems.append("centers_xy_raw: нужен список из >=2 точек")
    else:
        rows, cols = entry.get("raw_shape", (None, None))
        for point in centers:
            if not (isinstance(point, list) and len(point) == 2):
                problems.append(f"точка не пара: {point}")
                continue
            x, y = point
            if rows and cols and not (0 <= x < cols and 0 <= y < rows):
                problems.append(f"точка вне кадра: {point}")
        ys = [p[1] for p in centers if isinstance(p, list) and len(p) == 2]
        if ys != sorted(ys):
            problems.append("центры не упорядочены сверху вниз")
    for key in ("top_ribs_visible", "bottom_crest_visible"):
        if entry.get(key) not in (True, False, None):
            problems.append(f"{key}: ожидается bool или null")
    if not isinstance(entry.get("raw_shape"), list) or len(entry.get("raw_shape", [])) != 2:
        problems.append("raw_shape: ожидается [rows, cols]")
    return problems
