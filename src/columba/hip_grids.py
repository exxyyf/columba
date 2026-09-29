"""Этап 4 (подготовка): гриды позитивов/негативов бедра как файлы-артефакты.

По образцу `spine_grids.py` (этап 2). Нужны для просмотра глазами, на чём
основывается разведка данных этапа 4 (`stages/stage_4.md`): как выглядит
«Некорректная укладка» и «Некорректная область интереса» у бедра в этой
выгрузке.

Правое бедро отзеркаливается по горизонтали, чтобы совпадать по ориентации
с левым (левое бедро — эталон, отмечается в подписи «зерк.», если применялось).

Запуск: uv run python -m columba.hip_grids
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
from .inventory import load_manifest

POSITIONING_FILE = "hip_positives_positioning.png"
ROI_FILE = "hip_positives_roi.png"
NEGATIVES_FILE = "hip_negatives_sample.png"
NEGATIVES_SAMPLE_N = 12


def hip_targets_frame() -> pd.DataFrame:
    targets = pd.read_csv(cfg.TARGETS_CSV)
    manifest = load_manifest()
    paths = manifest[manifest["is_group_representative"].fillna(False)][
        ["dedup_group_id", "abs_path", "markup_comment"]
    ]
    frame = targets[(targets["region"] == cfg.REGION_HIP) & (targets["has_target"] == True)].merge(
        paths, on="dedup_group_id"
    )
    frame["markup_comment"] = frame["markup_comment"].fillna("")
    return frame.sort_values("dedup_group_id")


def _caption(record: dict) -> str:
    comment = str(record.get("markup_comment") or "").strip()
    comment = (comment[:28] + "…") if len(comment) > 28 else comment
    side = record.get("hip_side") or "?"
    mirrored = " зерк." if side == "right" else ""
    text = f"{record['dedup_group_id']} ({record['split']}, {side}{mirrored})"
    if comment:
        text += f"\n{comment}"
    return text


def render_grid(frame: pd.DataFrame, path: Path, title: str) -> None:
    count = len(frame)
    if not count:
        return
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 4.2 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, record in zip(axes.ravel(), frame.to_dict("records")):
        result = read_dicom(record["abs_path"])
        if not result.ok:
            continue
        pixels = normalize(result.pixels, result.tags)
        if record.get("hip_side") == "right":
            pixels = np.fliplr(pixels)
        axis.imshow(pixels, cmap="gray", aspect=cfg.ANISOTROPY_Y_OVER_X)
        axis.set_title(_caption(record), fontsize=7)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    frame = hip_targets_frame()
    cfg.EDA_DIR.mkdir(parents=True, exist_ok=True)

    positioning = frame[frame["label_hip_positioning"] == 1]
    render_grid(
        positioning,
        cfg.EDA_DIR / POSITIONING_FILE,
        f"Позитивы «{cfg.VIOLATION_POSITIONING}» — бедро ({len(positioning)} снимков)",
    )
    print(f"{POSITIONING_FILE}: {len(positioning)} позитивов", flush=True)

    roi = frame[frame["label_hip_roi"] == 1]
    render_grid(
        roi, cfg.EDA_DIR / ROI_FILE, f"Позитивы «{cfg.VIOLATION_ROI}» — бедро ({len(roi)} снимков)"
    )
    print(f"{ROI_FILE}: {len(roi)} позитивов", flush=True)

    negatives_pool = frame[(frame["label_hip_positioning"] == 0) & (frame["label_hip_roi"] == 0)]
    negatives = negatives_pool.sample(
        n=min(NEGATIVES_SAMPLE_N, len(negatives_pool)), random_state=cfg.SEED
    ).sort_values("dedup_group_id")
    render_grid(
        negatives,
        cfg.EDA_DIR / NEGATIVES_FILE,
        f"Контрольные негативы — бедро (n={len(negatives)}, seed={cfg.SEED})",
    )
    print(f"{NEGATIVES_FILE}: {len(negatives)} негативов", flush=True)


if __name__ == "__main__":
    main()
