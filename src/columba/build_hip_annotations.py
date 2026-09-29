"""Сборка разметки ориентиров бедра (этап 4, шаг 2).

Полуавтоматический процесс, по образцу `build_spine_annotations.py`:

    uv run python -m columba.build_hip_annotations

1. Классический детектор (`hip_landmarks.detect_hip_landmarks`) даёт
   кандидатов на всех уникальных снимках бедра (150 размеченных + `g0162`
   без таргета + `g0004`/`g0111`, 248 px, эндопротез — этап 1 относит их к
   бедру по `REGION_BY_STANDARD_WIDTH`).
2. Кандидаты просмотрены глазами через оверлеи-гриды (`render_review_grids`,
   пишутся в scratch, не в репозиторий) — найденные ошибки исправляются
   точечными правками `MANUAL_CORRECTIONS`.
3. Итог пишется в `artifacts/annotations/hip_landmarks.json`, ключ
   `dedup_group_id`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .dicom_io import read_dicom, normalize
from .hip_landmarks import compute_hip_signals, detect_hip_landmarks, save_annotations
from .inventory import load_manifest
from .spine_keypoints import STATUS_OK

# Точечные правки по итогам просмотра оверлеев (координаты исходных пикселей,
# ключи полей — как в HipLandmarks.to_dict()). Значение None у точки —
# «ориентир не виден на кадре, не размечаем», а не ошибка детектора.
MANUAL_CORRECTIONS: dict[str, dict] = {}

# Изображения, которые дополнительно включаются в разметку сверх
# `region == "hip"` в manifest.csv (эндопротез g0162 без строки в targets.csv;
# 248-px кадры эндопротеза g0004/g0111 — region в manifest.csv ещё "unknown",
# этап 1 относит их к бедру эвристикой REGION_BY_STANDARD_WIDTH).
EXTRA_GROUP_IDS: tuple[str, ...] = ("g0162", "g0004", "g0111")


def hip_representative_frame() -> pd.DataFrame:
    manifest = load_manifest()
    representatives = manifest[manifest["is_group_representative"].fillna(False)]
    is_hip = representatives["region"] == cfg.REGION_HIP
    is_extra = representatives["dedup_group_id"].isin(EXTRA_GROUP_IDS)
    frame = representatives[is_hip | is_extra].sort_values("dedup_group_id")
    return frame


def apply_corrections(gid: str, entry: dict) -> tuple[dict, bool]:
    correction = MANUAL_CORRECTIONS.get(gid)
    if not correction:
        return entry, False
    result = dict(entry)
    result.update(correction)
    return result, True


def build_annotations(verbose: bool = True) -> dict:
    frame = hip_representative_frame()
    annotations: dict[str, dict] = {}
    n_not_evaluated = 0
    for row in frame.itertuples():
        result = read_dicom(str(cfg.STUDIES_DIR / row.relative_path))
        if not result.ok:
            continue
        pixels = normalize(result.pixels, result.tags)
        landmarks = detect_hip_landmarks(pixels)
        if landmarks.status != STATUS_OK:
            n_not_evaluated += 1
        entry = {
            "side": landmarks.side,
            "head_center": list(landmarks.head_center_xy) if landmarks.head_center_xy else None,
            "shaft_points": [list(p) for p in landmarks.shaft_points_xy],
            "lesser_trochanter": list(landmarks.lesser_trochanter_xy)
            if landmarks.lesser_trochanter_xy
            else None,
            "greater_trochanter": list(landmarks.greater_trochanter_xy)
            if landmarks.greater_trochanter_xy
            else None,
            "raw_shape": [int(pixels.shape[0]), int(pixels.shape[1])],
            "status": landmarks.status,
            "reason": landmarks.reason,
            "reviewed": False,
        }
        entry, corrected = apply_corrections(row.dedup_group_id, entry)
        if corrected:
            entry["reviewed"] = True
        annotations[row.dedup_group_id] = entry
    if verbose:
        n_corrected = sum(1 for v in annotations.values() if v["reviewed"])
        print(
            f"разметка: {len(annotations)} снимков, not_evaluated {n_not_evaluated}, "
            f"ручных правок {n_corrected}",
            flush=True,
        )
    return annotations


def render_review_grids(out_dir: Path, columns: int = 5) -> list[Path]:
    """Оверлеи кандидатов детектора для просмотра глазами (в scratch, не в репо)."""
    frame = hip_representative_frame()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(frame.itertuples())
    paths: list[Path] = []
    per_grid = columns * 5
    for chunk_start in range(0, len(records), per_grid):
        chunk = records[chunk_start : chunk_start + per_grid]
        rows = (len(chunk) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 4.0 * rows), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for axis, row in zip(axes.ravel(), chunk):
            result = read_dicom(str(cfg.STUDIES_DIR / row.relative_path))
            if not result.ok:
                continue
            pixels = normalize(result.pixels, result.tags)
            landmarks = detect_hip_landmarks(pixels)
            axis.imshow(pixels, cmap="gray", aspect=cfg.ANISOTROPY_Y_OVER_X)
            title = f"{row.dedup_group_id} ({landmarks.side}, {landmarks.status})"
            if landmarks.status == STATUS_OK:
                if landmarks.head_center_xy:
                    axis.scatter(*landmarks.head_center_xy, c="lime", s=30, marker="o")
                if landmarks.shaft_points_xy:
                    xs = [p[0] for p in landmarks.shaft_points_xy]
                    ys = [p[1] for p in landmarks.shaft_points_xy]
                    axis.plot(xs, ys, c="cyan", linewidth=1.2)
                if landmarks.greater_trochanter_xy:
                    axis.scatter(*landmarks.greater_trochanter_xy, c="red", s=30, marker="^")
                if landmarks.lesser_trochanter_xy:
                    axis.scatter(*landmarks.lesser_trochanter_xy, c="yellow", s=30, marker="v")
            axis.set_title(title, fontsize=7)
        figure.suptitle(f"hip landmarks review {chunk_start}-{chunk_start + len(chunk)}", fontsize=10)
        figure.tight_layout()
        path = out_dir / f"hip_landmarks_review_{chunk_start:03d}.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        paths.append(path)
    return paths


HIP_OVERLAY_PREFIX = "hip_landmarks_overlay"


def landmarks_overlay_frame() -> pd.DataFrame:
    """Все размеченные снимки бедра с меткой `hip_positioning`, порядок вывода:
    сначала позитивы, потом негативы (для отчёта ревью), внутри групп — по gid.
    """
    targets = pd.read_csv(cfg.TARGETS_CSV)
    manifest = load_manifest()
    paths = manifest[manifest["is_group_representative"].fillna(False)][
        ["dedup_group_id", "relative_path"]
    ]
    frame = targets[(targets["region"] == cfg.REGION_HIP) & (targets["has_target"] == True)].merge(
        paths, on="dedup_group_id"
    )
    frame = frame.sort_values(["label_hip_positioning", "dedup_group_id"], ascending=[False, True])
    return frame


def render_landmarks_overlay_grids(
    out_dir: Path = cfg.EDA_DIR, per_grid: int = 20, columns: int = 5
) -> list[Path]:
    """Оверлеи ориентиров (головка/диафиз/вертелы) — артефакт этапа для ревью.

    Порядок: сначала все позитивы `hip_positioning` (36), потом негативы.
    Пишется в `artifacts/eda/hip_landmarks_overlay_{n}.png`. Воспроизводимо:

        uv run python -m columba.build_hip_annotations --overlays
    """
    frame = landmarks_overlay_frame()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(frame.itertuples())
    paths: list[Path] = []
    for chunk_start in range(0, len(records), per_grid):
        chunk = records[chunk_start : chunk_start + per_grid]
        rows = (len(chunk) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 4.4 * rows), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for axis, row in zip(axes.ravel(), chunk):
            result = read_dicom(str(cfg.STUDIES_DIR / row.relative_path))
            if not result.ok:
                continue
            pixels = normalize(result.pixels, result.tags)
            landmarks = detect_hip_landmarks(pixels)
            signals = compute_hip_signals(pixels, landmarks)
            axis.imshow(pixels, cmap="gray", aspect=cfg.ANISOTROPY_Y_OVER_X)
            if landmarks.status == STATUS_OK:
                if landmarks.head_center_xy:
                    axis.add_patch(
                        plt.Circle(landmarks.head_center_xy, 10, fill=False, color="lime", linewidth=1.4)
                    )
                if landmarks.shaft_points_xy:
                    xs = [p[0] for p in landmarks.shaft_points_xy]
                    ys = [p[1] for p in landmarks.shaft_points_xy]
                    axis.plot(xs, ys, c="cyan", linewidth=1.3)
                    axis.scatter(xs, ys, c="cyan", s=8)
                if landmarks.greater_trochanter_xy:
                    axis.scatter(*landmarks.greater_trochanter_xy, c="red", s=45, marker="^", label="GT")
                if landmarks.lesser_trochanter_xy:
                    axis.scatter(*landmarks.lesser_trochanter_xy, c="yellow", s=45, marker="v", label="LT")
            label = int(row.label_hip_positioning) if row.label_hip_positioning == row.label_hip_positioning else "?"
            prom = signals.get("lesser_trochanter_prominence", float("nan"))
            prom_str = f"{prom:.0f}мм" if prom == prom else "н/д"
            title = f"{row.dedup_group_id} ({row.split}) pos={label} prom={prom_str}"
            axis.set_title(title, fontsize=7)
        figure.suptitle(
            f"hip_landmarks overlay {chunk_start + 1}-{chunk_start + len(chunk)} "
            f"(головка=круг, диафиз=линия Theil-Sen, ▲=большой вертел, ▼=малый вертел)",
            fontsize=9,
        )
        figure.tight_layout()
        index = chunk_start // per_grid + 1
        path = out_dir / f"{HIP_OVERLAY_PREFIX}_{index}.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        paths.append(path)
    return paths


def main() -> None:
    import sys

    if "--overlays" in sys.argv:
        paths = render_landmarks_overlay_grids()
        for path in paths:
            print(f"записано: {path}", flush=True)
        return

    annotations = build_annotations()
    path = save_annotations(annotations)
    print(f"записано: {path}", flush=True)


if __name__ == "__main__":
    main()
