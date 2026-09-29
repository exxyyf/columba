"""Этап 2, шаг 1: гриды позитивов позвоночника как файлы-артефакты.

Просмотр позитивов глазами делался вручную; здесь те же гриды сохраняются
воспроизводимо в `artifacts/eda/spine_positives_<критерий>.png`, чтобы
опора «как выглядит нарушение в этой выгрузке» существовала как артефакт,
а не только как чьи-то заметки.

Запуск: uv run python -m columba.spine_grids
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as cfg
from .dicom_io import read_dicom, normalize
from .inventory import load_manifest

CRITERIA_LABELS = {
    "label_spine_positioning": ("spine_positives_positioning.png", cfg.VIOLATION_POSITIONING),
    "label_spine_axis": ("spine_positives_axis.png", cfg.VIOLATION_SPINE_AXIS),
    "label_spine_objects": ("spine_positives_objects.png", cfg.VIOLATION_FOREIGN_OBJECTS),
}


def spine_targets_frame() -> pd.DataFrame:
    targets = pd.read_csv(cfg.TARGETS_CSV)
    manifest = load_manifest()
    paths = manifest[manifest["is_group_representative"].fillna(False)][
        ["dedup_group_id", "abs_path"]
    ]
    frame = targets[targets["region"] == cfg.REGION_SPINE].merge(paths, on="dedup_group_id")
    return frame.sort_values("dedup_group_id")


def render_grid(frame: pd.DataFrame, path: Path, title: str) -> None:
    count = len(frame)
    if not count:
        return
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.8 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, record in zip(axes.ravel(), frame.to_dict("records")):
        result = read_dicom(record["abs_path"])
        if not result.ok:
            continue
        axis.imshow(normalize(result.pixels, result.tags), cmap="gray", aspect=cfg.ANISOTROPY_Y_OVER_X)
        axis.set_title(f"{record['dedup_group_id']} ({record['split']})", fontsize=8)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    frame = spine_targets_frame()
    cfg.EDA_DIR.mkdir(parents=True, exist_ok=True)
    for column, (file_name, violation) in CRITERIA_LABELS.items():
        positives = frame[frame[column] == 1]
        path = cfg.EDA_DIR / file_name
        render_grid(positives, path, f"Позитивы «{violation}» ({len(positives)} снимков)")
        print(f"{path.name}: {len(positives)} позитивов", flush=True)


if __name__ == "__main__":
    main()
