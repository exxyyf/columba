"""Точка входа этапа 1: обучить CNN региона/стороны на артефактах этапа 0.

Запуск: `uv run python -m columba.train_region_cnn`
Требует artifacts/manifest.parquet и artifacts/split.json (создаются `main.py`).
"""

from __future__ import annotations

from . import config as cfg
from .inventory import load_manifest
from .region_cnn import build_region_dataset, train_region_cnn
from .split import load_split


def main() -> dict:
    manifest = load_manifest(cfg.MANIFEST_PARQUET)
    split_payload = load_split(cfg.SPLIT_JSON)
    dataset = build_region_dataset(manifest, split_payload)
    print(
        f"Датасет CNN: {len(dataset)} изображений "
        f"({dataset['cnn_split'].value_counts().to_dict()}), "
        f"классы: {dataset['cnn_label'].value_counts().to_dict()}",
        flush=True,
    )
    report = train_region_cnn(dataset)
    print(
        f"Готово: train acc={report['train_accuracy']}, val acc={report['val_accuracy']}; "
        f"веса: {cfg.REGION_CNN_WEIGHTS}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    main()
