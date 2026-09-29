"""Этап 8, бонус 1: визуализация нарушений — почти бесплатно из кейпоинтов.

Каждый чекер этапов 2/4/5 уже считает именованные ориентиры/сигналы для
объяснимости (`CheckerResult.signals`) — этот модуль просто рисует их поверх
кадра и подписывает результат чекеров, не добавляя новой логики детекции.
Один кадр — один вызов `render_violation_overlay`, не привязан к жёсткому
списку примеров (в отличие от `spine_visualize.py`, EDA-заготовки этапа 2).

Запуск (CLI): `uv run python -m columba.visualize <файл.dcm | каталог> [-o путь]`
— по умолчанию каждый кадр сохраняется как `artifacts/eda/violations/<имя>.png`.

Используется также сервисом этапа 7 (`service.py`, `POST /visualize`) —
`figure_to_png_bytes` отдаёт те же байты PNG, что видит CLI.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from . import config as cfg
from .dicom_io import normalize, read_dicom
from .inventory import DICOM_SUFFIX
from .regions import classify_region_from_pixels, hip_side
from .spine_checkers import (
    check_spine_axis,
    check_spine_objects,
    check_spine_positioning,
    column_midline,
    get_spine_keypoints,
    metal_components,
)
from .spine_keypoints import (
    STATUS_OK,
    iso_to_raw,
    mm_to_iso_px,
    resample_isotropic_np,
    smooth_2d,
)

STATUS_RU = {"ok": "ok", "not_evaluated": "н/д"}


def _checker_line(result) -> str:
    if result.status != STATUS_OK:
        return f"{result.checker}: н/д ({result.reason})"
    flag = "⚠ НАРУШЕНИЕ" if result.flag else "норма"
    return f"{result.checker}: {flag} (score={result.score:.3g})"


# --------------------------------------------------------------------------- #
# Позвоночник
# --------------------------------------------------------------------------- #


def _render_spine(pixels: np.ndarray, ax) -> dict:
    keypoints = get_spine_keypoints(pixels)
    axis_result = check_spine_axis(pixels, keypoints)
    positioning_result = check_spine_positioning(pixels, keypoints)
    objects_result = check_spine_objects(pixels)

    if keypoints.status == STATUS_OK:
        centers = keypoints.centers_raw_xy
        ax.plot(centers[:, 0], centers[:, 1], "r+", markersize=9, markeredgewidth=1.6, label="центры тел")
        iso_smooth = smooth_2d(resample_isotropic_np(pixels), mm_to_iso_px(cfg.SPINE_SMOOTH_SIGMA_MM))
        ys, mids = column_midline(iso_smooth, keypoints.centers_iso_xy)
        if len(ys):
            midline_raw = iso_to_raw(np.stack([mids, ys], axis=1), pixels.shape[0])
            ax.plot(midline_raw[:, 0], midline_raw[:, 1], color="cyan", linewidth=1.4, label="ось (midline)")

    for component in metal_components(pixels):
        x0, y0, x1, y1 = component["bbox_iso"]
        box = iso_to_raw(np.array([[x0, y0], [x1, y1]], dtype=float), pixels.shape[0])
        ax.add_patch(
            _rectangle(box[0, 0], box[0, 1], box[1, 0] - box[0, 0], box[1, 1] - box[0, 1], "orange")
        )

    checkers = {"spine_axis": axis_result, "spine_positioning": positioning_result, "spine_objects": objects_result}
    return {"region": cfg.REGION_SPINE, "checkers": checkers}


# --------------------------------------------------------------------------- #
# Бедро
# --------------------------------------------------------------------------- #


def _render_hip(pixels: np.ndarray, side: str | None, ax, cnn_predictor=None) -> dict:
    from .hip_checkers import check_hip_positioning, check_hip_roi
    from .hip_landmarks import compute_hip_roi_signals, compute_hip_signals, detect_hip_landmarks

    landmarks = detect_hip_landmarks(pixels, side=side)
    signals = compute_hip_signals(pixels, landmarks)
    roi_signals = compute_hip_roi_signals(pixels, landmarks)
    positioning_result = check_hip_positioning(pixels, landmarks, signals, cnn_predictor=cnn_predictor)
    roi_result = check_hip_roi(pixels, landmarks, roi_signals)

    if landmarks.status == STATUS_OK:
        points = {
            "головка": (landmarks.head_center_xy, "lime"),
            "больш. вертел": (landmarks.greater_trochanter_xy, "yellow"),
            "мал. вертел": (landmarks.lesser_trochanter_xy, "magenta"),
        }
        for label, (point, color) in points.items():
            if point is not None:
                ax.plot(point[0], point[1], "o", color=color, markersize=6, label=label)
        if landmarks.shaft_points_xy:
            shaft = np.asarray(landmarks.shaft_points_xy)
            ax.plot(shaft[:, 0], shaft[:, 1], color="cyan", linewidth=1.4, label="ось диафиза")

    height = pixels.shape[0]
    ref = cfg.HIP_ROI_FIELD_HEIGHT_REF_PX
    if height < ref:
        ax.axhline(height - 1, color="red", linewidth=1.2, linestyle="--", label=f"кадр короче REF={ref:g}px")

    checkers = {"hip_positioning": positioning_result, "hip_roi": roi_result}
    return {"region": cfg.REGION_HIP, "side": side, "checkers": checkers}


def _rectangle(x, y, w, h, color):
    import matplotlib.pyplot as plt

    return plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.4)


# --------------------------------------------------------------------------- #
# Общая точка входа
# --------------------------------------------------------------------------- #


def render_violation_overlay(
    pixels_normalized: np.ndarray,
    region: str | None = None,
    side: str | None = None,
    hip_cnn_predictor=None,
):
    """Кадр + ориентиры + результаты чекеров -> (figure, info).

    `region`/`side`, если не переданы, определяются по пикселям
    (`classify_region_from_pixels`/`regions.hip_side`) — тем же способом,
    что использует закрытый тест (без тегов/имён файлов, инвариант этапа 1).
    `hip_cnn_predictor` — опциональный `hip_cnn.HipPositioningPredictor`;
    без него `hip_positioning` честно уходит в `not_evaluated` (как в проде
    без весов, см. `hip_checkers.check_hip_positioning`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if region is None:
        region = classify_region_from_pixels(pixels_normalized)

    fig, ax = plt.subplots(figsize=(5.0, 7.2))
    ax.imshow(pixels_normalized, cmap="gray", aspect=cfg.ANISOTROPY_Y_OVER_X)

    if region == cfg.REGION_SPINE:
        info = _render_spine(pixels_normalized, ax)
    elif region == cfg.REGION_HIP:
        if side is None:
            side = hip_side(pixels_normalized).side
        info = _render_hip(pixels_normalized, side, ax, cnn_predictor=hip_cnn_predictor)
    else:
        info = {"region": region, "checkers": {}}

    ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower left", fontsize=7, framealpha=0.65)  # значение подобрано так, чтобы не совпасть с PIXEL_SPACING_MM_X (см. test_pixel_spacing_defined_in_one_place)

    # Ревью этапа 9, п. 9.6 (WCAG): регион/сторона/результаты чекеров больше не
    # впечатаны как текст заголовка внутри PNG — тот же текст, что раньше был
    # в title (см. `_checker_line`/`summary_lines`), теперь доступен как
    # структурированные данные (`info`, JSON-эндпоинт `/visualize?format=json`)
    # и рендерится хостом как настоящий HTML-текст (селектируемый, читаемый
    # скринридером), а не как пиксели картинки.
    fig.tight_layout()
    return fig, info


def summary_lines(info: dict) -> list[str]:
    """Текстовая сводка `info` (регион/сторона/чекеры) построчно.

    То же содержимое, что раньше подписывалось прямо в PNG (см. история
    `render_violation_overlay`) — теперь используется CLI-выводом и как
    основа JSON-ответа `/visualize`, не рисуется поверх кадра (задача 9.6).
    """
    lines = [f"регион: {info['region']}" + (f" ({info['side']})" if info.get("side") else "")]
    lines += [_checker_line(result) for result in info["checkers"].values()]
    return lines


def figure_to_png_bytes(fig) -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_dicom_file(path: Path, hip_cnn_predictor=None):
    """Прочитать DICOM и построить визуализацию -> (figure, info) либо
    (None, None) при нечитаемом файле (честная деградация, не исключение)."""
    result = read_dicom(path)
    if not result.ok:
        return None, None
    pixels = normalize(result.pixels, result.tags)
    return render_violation_overlay(pixels, hip_cnn_predictor=hip_cnn_predictor)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

DEFAULT_OUTPUT_DIR = cfg.EDA_DIR / "violations"


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from .hip_cnn import load_hip_positioning_predictor

    parser = argparse.ArgumentParser(
        prog="python -m columba.visualize",
        description="Кадр + ориентиры + результаты чекеров -> PNG (этап 8, бонус 1).",
    )
    parser.add_argument("input", help="DICOM-файл или каталог (рекурсивно)")
    parser.add_argument("-o", "--output", help="путь PNG (только для одного файла) или каталог")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ошибка: не найдено: {input_path}", file=sys.stderr)
        return 1
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob(f"*{DICOM_SUFFIX}"))
    if not files:
        print(f"ошибка: нет файлов {DICOM_SUFFIX}: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if (args.output and len(files) > 1) else DEFAULT_OUTPUT_DIR
    single_output = Path(args.output) if (args.output and len(files) == 1) else None

    predictor = load_hip_positioning_predictor()
    for file_path in files:
        fig, info = render_dicom_file(file_path, hip_cnn_predictor=predictor)
        if fig is None:
            print(f"{file_path.name}: пропущен (нечитаемый файл)")
            continue
        out_path = single_output or (output_dir / f"{file_path.stem}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        flags = [k for k, r in info["checkers"].items() if getattr(r, "flag", False)]
        print(f"{file_path.name} ({info['region']}): {', '.join(flags) or 'норма'} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
