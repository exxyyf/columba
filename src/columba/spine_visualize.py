"""Визуализация чекеров позвоночника (задел этапа 8, артефакт этапа 2).

Запуск: uv run python -m columba.spine_visualize [dedup_id ...]

Рисует кадр + центры тел позвонков + линию оси (midline) + подсвеченные
компоненты «металла»; по умолчанию — по два примера на каждый тип нарушения
и один чистый снимок. Итог — artifacts/eda/spine_checker_examples.png.
"""

from __future__ import annotations

import sys

import numpy as np

from . import config as cfg
from .dicom_io import normalize, read_dicom_strict
from .inventory import load_manifest
from .spine_checkers import (
    check_spine_axis,
    column_midline,
    get_spine_keypoints,
    metal_components,
)
from .spine_keypoints import STATUS_OK, iso_to_raw, mm_to_iso_px, resample_isotropic_np, smooth_2d

DEFAULT_EXAMPLES = ("g0166", "g0075", "g0024", "g0212", "g0134", "g0081", "g0013")
OUTPUT_PNG = cfg.EDA_DIR / "spine_checker_examples.png"


def render(dedup_ids: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = load_manifest()
    representatives = manifest[manifest["is_group_representative"].fillna(False)].set_index("dedup_group_id")

    n = len(dedup_ids)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 9))
    axes = np.atleast_1d(axes)
    for ax, gid in zip(axes, dedup_ids):
        result = read_dicom_strict(representatives.loc[gid, "abs_path"])
        pixels = normalize(result.pixels, result.tags)
        keypoints = get_spine_keypoints(pixels)
        axis = check_spine_axis(pixels, keypoints)
        ax.imshow(pixels, cmap="gray", aspect=cfg.ANISOTROPY_Y_OVER_X)
        if keypoints.status == STATUS_OK:
            centers = keypoints.centers_raw_xy
            ax.plot(centers[:, 0], centers[:, 1], "r+", markersize=9, markeredgewidth=1.6,
                    label="центры тел")
            iso_smooth = smooth_2d(resample_isotropic_np(pixels), mm_to_iso_px(cfg.SPINE_SMOOTH_SIGMA_MM))
            ys, mids = column_midline(iso_smooth, keypoints.centers_iso_xy)
            if len(ys):
                midline_raw = iso_to_raw(np.stack([mids, ys], axis=1), pixels.shape[0])
                ax.plot(midline_raw[:, 0], midline_raw[:, 1], color="cyan", linewidth=1.2,
                        label="ось (midline)")
        for component in metal_components(pixels):
            x0, y0, x1, y1 = component["bbox_iso"]
            box = iso_to_raw(np.array([[x0, y0], [x1, y1]], dtype=float), pixels.shape[0])
            ax.add_patch(plt.Rectangle((box[0, 0], box[0, 1]), box[1, 0] - box[0, 0],
                                       box[1, 1] - box[0, 1], fill=False, edgecolor="orange",
                                       linewidth=1.4))
        title = gid
        if axis.status == STATUS_OK:
            title += f"\nось {axis.signals['axis_angle_deg']}°"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=130)
    print(f"записано: {OUTPUT_PNG}", flush=True)


if __name__ == "__main__":
    ids = sys.argv[1:] or list(DEFAULT_EXAMPLES)
    render(ids)
