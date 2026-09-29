"""Мост для TypeScript-интерфейса: классификация региона средствами пайплайна.

Вызывает `columba.inference.describe_inputs` (эвристика по ширине кадра +
CNN этапа 1 + арбитраж) для каждого переданного каталога и печатает JSON
в stdout. Никакой собственной логики классификации здесь нет — интерфейс
показывает ровно то, что решает пайплайн.

Использование: python bridge/describe.py <каталог> [<каталог> ...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from columba import config as cfg  # noqa: E402
from columba.inference import describe_inputs  # noqa: E402
from columba.region_cnn import load_region_predictor  # noqa: E402

# Колонки, которые нужны интерфейсу; остальное не сериализуем.
COLUMNS = [
    "file_id",
    "file_name",
    "relative_path",
    "abs_path",
    "study_folder",
    "read_status",
    "read_error",
    "rows",
    "cols",
    "region",
    "region_final",
    "hip_side",
    "hip_side_final",
    "hip_side_score",
    "hip_side_confident",
    "hip_side_method",
    "region_method",
    "region_disagreement",
    "cnn_label",
    "dedup_group_id",
    "dedup_group_size",
]


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):  # numpy-скаляры
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> None:
    roots = sys.argv[1:]
    if not roots:
        raise SystemExit("нужен хотя бы один каталог со снимками")

    cnn_available = load_region_predictor() is not None
    payload: dict = {
        "cnn_available": cnn_available,
        "region_labels": dict(cfg.REGION_OUTPUT_NAMES),
        "pixel_spacing_mm": {"y": cfg.PIXEL_SPACING_MM_Y, "x": cfg.PIXEL_SPACING_MM_X},
        "roots": [],
    }

    for root in roots:
        frame = describe_inputs(root)
        records = []
        if not frame.empty:
            present = [c for c in COLUMNS if c in frame.columns]
            for row in frame[present].to_dict(orient="records"):
                cleaned = {key: _clean(value) for key, value in row.items()}
                # pandas NA не проходит через math.isnan
                for key, value in cleaned.items():
                    if value is not None and str(value) == "<NA>":
                        cleaned[key] = None
                records.append(cleaned)
        payload["roots"].append({"root": str(Path(root)), "files": records})

    json.dump(payload, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
