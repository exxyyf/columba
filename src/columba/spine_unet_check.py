"""Этап 2, шаг 3: проверка маскирования для U-Net кейпоинтов.

Те же проверки, что у region_cnn на этапе 1, в применении к кейпоинт-сети:

* чувствительность к синтетическому маскированию углов кадра — предсказания
  до/после зануления углов сравниваются в миллиметрах;
* вместо Grad-CAM — карта салиенси (градиент суммы пиков хитмапы по входу):
  у полносвёрточной сети с хитмапой нет классификационной головы, и честный
  аналог «куда смотрит сеть» — входной градиент. Дополнительно считается доля
  массы салиенси, попадающая в занулённые углы, — прямой ответ на вопрос
  «не опирается ли сеть на края маскирования».

Снимки — ПОП с наибольшей площадью реального маскирования (has_mask_rect).
Запуск: uv run python -m columba.spine_unet_check
Артефакты: artifacts/models/spine_unet_masking_check.{png,json}.

U-Net в рантайм этапа 2 не включена (решает классический детектор,
см. decisions), поэтому проверка — санитарная, на случай подключения сети
на этапе 6.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import config as cfg
from .dicom_io import normalize, read_dicom_strict
from .inventory import load_manifest
from .spine_keypoint_net import (
    NetTransform,
    extract_peaks,
    letterbox_iso,
    load_spine_keypoint_predictor,
    match_points_mm,
)
from .spine_keypoints import mm_to_iso_px, resample_isotropic_np


def mask_corners(pixels_normalized: np.ndarray) -> np.ndarray:
    """Синтетическое маскирование углов — та же схема, что в юнит-тестах."""
    masked = pixels_normalized.copy()
    height, width = masked.shape
    masked[: height // 5, : width // 5] = 0.0
    masked[-height // 6 :, -width // 4 :] = 0.0
    return masked


def _predict_net_space(predictor, pixels_normalized: np.ndarray):
    """Хитмапа, пики в координатах сети и салиенси входного градиента."""
    iso = resample_isotropic_np(pixels_normalized)
    transform = NetTransform.for_iso_shape(iso.shape, int(pixels_normalized.shape[0]))
    image = letterbox_iso(iso, transform).unsqueeze(0).to(predictor.device)
    image.requires_grad_(True)
    heatmap_tensor = predictor.model(image)[0, 0]
    heatmap = heatmap_tensor.detach().cpu().numpy()
    min_spacing = mm_to_iso_px(cfg.SPINE_CENTER_MIN_SPACING_MM) * transform.scale
    peaks_net = extract_peaks(heatmap, min_spacing, cfg.SPINE_UNET_PEAK_THRESHOLD)

    saliency = np.zeros_like(heatmap)
    if len(peaks_net):
        ys = torch.as_tensor(peaks_net[:, 1].round().astype(int))
        xs = torch.as_tensor(peaks_net[:, 0].round().astype(int))
        heatmap_tensor[ys, xs].sum().backward()
        saliency = image.grad[0, 0].abs().cpu().numpy()
    return transform, image.detach()[0, 0].cpu().numpy(), heatmap, peaks_net, saliency


def _corner_saliency_fraction(saliency: np.ndarray) -> float:
    """Доля массы салиенси в углах, соответствующих маскированию."""
    total = float(saliency.sum())
    if total <= 0:
        return 0.0
    height, width = saliency.shape
    corners = float(saliency[: height // 5, : width // 5].sum()) + float(
        saliency[-height // 6 :, -width // 4 :].sum()
    )
    return corners / total


def pick_frames(n_images: int = cfg.SPINE_UNET_MASKING_N_IMAGES) -> pd.DataFrame:
    manifest = load_manifest()
    spine = manifest[
        manifest["is_group_representative"].fillna(False)
        & (manifest["region"] == cfg.REGION_SPINE)
        & manifest["has_mask_rect"].fillna(False)
    ]
    return spine.sort_values("mask_step_area_px", ascending=False).head(n_images)


def run_check(
    png_path: Path | str = cfg.SPINE_UNET_MASKING_PNG,
    json_path: Path | str = cfg.SPINE_UNET_MASKING_JSON,
    verbose: bool = True,
) -> dict:
    predictor = load_spine_keypoint_predictor()
    if predictor is None:
        raise FileNotFoundError(f"нет весов U-Net: {cfg.SPINE_UNET_WEIGHTS}")
    frames = pick_frames()

    figure, axes = plt.subplots(len(frames), 3, figsize=(9.5, 3.4 * len(frames)), squeeze=False)
    summaries = []
    for row_axes, row in zip(axes, frames.itertuples()):
        result = read_dicom_strict(row.abs_path)
        pixels = normalize(result.pixels, result.tags)
        base = predictor.predict(pixels)
        masked_pixels = mask_corners(pixels)
        masked = predictor.predict(masked_pixels)
        agreement = match_points_mm(masked.centers_raw_xy, base.centers_raw_xy)

        _, net_image, _, peaks, saliency = _predict_net_space(predictor, pixels)
        _, net_masked, _, peaks_masked, _ = _predict_net_space(predictor, masked_pixels)

        summaries.append(
            {
                "dedup_group_id": row.dedup_group_id,
                "mask_step_area_px": int(row.mask_step_area_px),
                "n_centers": base.n_centers,
                "n_centers_masked": masked.n_centers,
                "agreement_recall": round(agreement["recall"], 3),
                "mean_shift_mm": None
                if agreement["mean_error_mm"] != agreement["mean_error_mm"]
                else round(agreement["mean_error_mm"], 2),
                "corner_saliency_fraction": round(_corner_saliency_fraction(saliency), 4),
            }
        )

        for axis, (image, points, title, cmap) in zip(
            row_axes,
            [
                (net_image, peaks, f"{row.dedup_group_id}: вход + пики", "gray"),
                (saliency, peaks, "салиенси (|град. пиков по входу|)", "inferno"),
                (net_masked, peaks_masked, "углы занулены + пики", "gray"),
            ],
        ):
            axis.imshow(image, cmap=cmap)
            if len(points):
                axis.scatter(points[:, 0], points[:, 1], s=12, c="#4c72b0", marker="+")
            axis.set_title(title, fontsize=8)
            axis.axis("off")

    aggregate = {
        "images": summaries,
        "mean_agreement_recall": round(float(np.mean([s["agreement_recall"] for s in summaries])), 3),
        "max_corner_saliency_fraction": round(
            max(s["corner_saliency_fraction"] for s in summaries), 4
        ),
    }
    figure.suptitle("U-Net кейпоинтов: чувствительность к маскированию углов", fontsize=10, y=0.998)
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=110)
    plt.close(figure)
    Path(json_path).write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
        print(f"артефакты: {png_path}, {json_path}", flush=True)
    return aggregate


if __name__ == "__main__":
    run_check()
