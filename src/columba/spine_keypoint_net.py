"""Этап 2, шаг 3: нейросетевая модель кейпоинтов тел позвонков.

Лёгкая U-Net с одноканальной хитмапой «центр тела позвонка»: переменное число
позвонков решается пиками одной хитмапы, а не фиксированными головами.
Обучается на разметке шага 2 (train-фолд сплита этапа 0). В рантайме —
страховка/уточнение классического детектора; при отсутствии весов пайплайн
честно деградирует на классическую ветку.

Вход: изотропный кадр (растяжение Y в ANISOTROPY_Y_OVER_X), вписанный в
SPINE_UNET_INPUT_HEIGHT x SPINE_UNET_INPUT_WIDTH с сохранением пропорций.
Преобразование координат сеть <-> исходные пиксели — `NetTransform`,
инволюция закрыта юнит-тестом.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import config as cfg
from .dicom_io import normalize, read_dicom_strict
from .inventory import load_manifest
from .spine_keypoints import (
    STATUS_NOT_EVALUATED,
    STATUS_OK,
    SpineKeypoints,
    iso_to_raw,
    load_annotations,
    mm_to_iso_px,
    raw_to_iso,
    resample_isotropic_np,
)
from .spine_synth import rotate_isotropic, rotate_points_iso

# Аугментации обучения. Повороты шире, чем у CNN этапа 1: сеть обязана
# отслеживать наклонённые оси (синтетика 6-10 градусов — позитивы чекера).
AUG_MAX_ROTATION_DEG = 12.0
AUG_BRIGHTNESS = 0.10
AUG_CONTRAST = 0.10


# --------------------------------------------------------------------------- #
# Преобразование координат: исходный кадр <-> вход сети
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NetTransform:
    """Параметры вписывания изотропного кадра в фиксированный вход сети."""

    raw_height: int
    iso_shape: tuple[int, int]
    scale: float
    top: int
    left: int

    @classmethod
    def for_iso_shape(cls, iso_shape: tuple[int, int], raw_height: int) -> "NetTransform":
        height, width = iso_shape
        scale = min(cfg.SPINE_UNET_INPUT_HEIGHT / height, cfg.SPINE_UNET_INPUT_WIDTH / width)
        new_h = max(1, int(round(height * scale)))
        new_w = max(1, int(round(width * scale)))
        top = (cfg.SPINE_UNET_INPUT_HEIGHT - new_h) // 2
        left = (cfg.SPINE_UNET_INPUT_WIDTH - new_w) // 2
        return cls(raw_height=raw_height, iso_shape=iso_shape, scale=scale, top=top, left=left)

    def iso_to_net(self, points_iso: np.ndarray) -> np.ndarray:
        points = np.asarray(points_iso, dtype=np.float64).reshape(-1, 2).copy()
        points[:, 0] = (points[:, 0] + 0.5) * self.scale - 0.5 + self.left
        points[:, 1] = (points[:, 1] + 0.5) * self.scale - 0.5 + self.top
        return points

    def net_to_iso(self, points_net: np.ndarray) -> np.ndarray:
        points = np.asarray(points_net, dtype=np.float64).reshape(-1, 2).copy()
        points[:, 0] = (points[:, 0] - self.left + 0.5) / self.scale - 0.5
        points[:, 1] = (points[:, 1] - self.top + 0.5) / self.scale - 0.5
        return points

    def raw_to_net(self, points_raw: np.ndarray) -> np.ndarray:
        return self.iso_to_net(raw_to_iso(points_raw, self.raw_height))

    def net_to_raw(self, points_net: np.ndarray) -> np.ndarray:
        return iso_to_raw(self.net_to_iso(points_net), self.raw_height)


def letterbox_iso(iso_image: np.ndarray, transform: NetTransform) -> torch.Tensor:
    """Изотропный кадр -> тензор входа сети 1 x H x W."""
    tensor = torch.as_tensor(np.ascontiguousarray(iso_image), dtype=torch.float32)
    height, width = tensor.shape
    new_h = max(1, int(round(height * transform.scale)))
    new_w = max(1, int(round(width * transform.scale)))
    resized = F.interpolate(
        tensor.reshape(1, 1, height, width), size=(new_h, new_w), mode="bilinear", align_corners=False
    )[0, 0]
    canvas = torch.zeros(cfg.SPINE_UNET_INPUT_HEIGHT, cfg.SPINE_UNET_INPUT_WIDTH)
    canvas[transform.top : transform.top + new_h, transform.left : transform.left + new_w] = resized
    return canvas.unsqueeze(0)


def render_heatmap(points_net: np.ndarray, sigma_px: float) -> torch.Tensor:
    """Гауссианы в координатах сети -> хитмапа 1 x H x W с пиками 1.0."""
    heatmap = np.zeros((cfg.SPINE_UNET_INPUT_HEIGHT, cfg.SPINE_UNET_INPUT_WIDTH), dtype=np.float32)
    if len(points_net):
        yy, xx = np.mgrid[0 : heatmap.shape[0], 0 : heatmap.shape[1]].astype(np.float32)
        for x, y in points_net:
            gauss = np.exp(-((xx - float(x)) ** 2 + (yy - float(y)) ** 2) / (2 * sigma_px**2))
            heatmap = np.maximum(heatmap, gauss)
    return torch.from_numpy(heatmap.astype(np.float32)).unsqueeze(0)


def heatmap_sigma_net_px(transform: NetTransform) -> float:
    return mm_to_iso_px(cfg.SPINE_UNET_HEATMAP_SIGMA_MM) * transform.scale


# --------------------------------------------------------------------------- #
# Модель
# --------------------------------------------------------------------------- #


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SpineUNet(nn.Module):
    """U-Net глубины 3, базовые 16 каналов: ~0.5M параметров, CPU-дружелюбно."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.enc1 = _Block(1, base)
        self.enc2 = _Block(base, base * 2)
        self.enc3 = _Block(base * 2, base * 4)
        self.bottom = _Block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _Block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _Block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _Block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottom(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))


# --------------------------------------------------------------------------- #
# Датасет
# --------------------------------------------------------------------------- #


class SpineKeypointDataset(Dataset):
    """Изотропные кадры + центры из разметки; аугментации на лету (train)."""

    def __init__(self, frame: pd.DataFrame, annotations: dict, *, augment: bool, seed: int = cfg.SEED):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.rng = random.Random(seed)
        self.samples: list[tuple[np.ndarray, np.ndarray, int]] = []
        for row in self.frame.itertuples():
            result = read_dicom_strict(row.abs_path)
            pixels = normalize(result.pixels, result.tags)
            iso = resample_isotropic_np(pixels)
            entry = annotations[row.dedup_group_id]
            centers_raw = np.array(entry["centers_xy_raw"], dtype=np.float64)
            centers_iso = raw_to_iso(centers_raw, pixels.shape[0])
            self.samples.append((iso, centers_iso, pixels.shape[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        iso, centers_iso, raw_height = self.samples[index]
        if self.augment:
            angle = self.rng.uniform(-AUG_MAX_ROTATION_DEG, AUG_MAX_ROTATION_DEG)
            iso = rotate_isotropic(iso, angle)
            centers_iso = rotate_points_iso(centers_iso, iso.shape, angle)
            brightness = 1.0 + self.rng.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS)
            contrast = 1.0 + self.rng.uniform(-AUG_CONTRAST, AUG_CONTRAST)
            mean = float(iso.mean())
            iso = np.clip((iso - mean) * contrast + mean * brightness, 0.0, 1.0)
        transform = NetTransform.for_iso_shape(iso.shape, raw_height)
        inside = [
            (x, y)
            for x, y in centers_iso
            if 0 <= x < iso.shape[1] and 0 <= y < iso.shape[0]
        ]
        points_net = transform.iso_to_net(np.array(inside)) if inside else np.zeros((0, 2))
        image = letterbox_iso(iso, transform)
        heatmap = render_heatmap(points_net, heatmap_sigma_net_px(transform))
        return image, heatmap


# --------------------------------------------------------------------------- #
# Извлечение пиков
# --------------------------------------------------------------------------- #


def extract_peaks(heatmap: np.ndarray, min_spacing_px: float, threshold: float) -> np.ndarray:
    """Пики хитмапы: локальные максимумы выше порога с минимальным зазором."""
    tensor = torch.as_tensor(heatmap).reshape(1, 1, *heatmap.shape)
    pooled = F.max_pool2d(tensor, kernel_size=5, stride=1, padding=2)
    is_peak = (tensor == pooled) & (tensor >= threshold)
    ys, xs = np.nonzero(is_peak[0, 0].numpy())
    order = np.argsort(-heatmap[ys, xs])
    chosen: list[tuple[float, float]] = []
    for i in order:
        x, y = float(xs[i]), float(ys[i])
        if all((x - cx) ** 2 + (y - cy) ** 2 >= min_spacing_px**2 for cx, cy in chosen):
            chosen.append((x, y))
    chosen.sort(key=lambda p: p[1])  # сверху вниз
    return np.array(chosen, dtype=np.float64).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# Предиктор
# --------------------------------------------------------------------------- #


class SpineKeypointPredictor:
    """Загрузка весов U-Net и предсказание центров по нормализованным пикселям."""

    def __init__(self, weights_path: Path | str = cfg.SPINE_UNET_WEIGHTS, device: torch.device | None = None):
        from .region_cnn import default_device

        self.device = device or default_device()
        payload = torch.load(Path(weights_path), map_location=self.device, weights_only=True)
        self.model = SpineUNet()
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()

    def predict(self, pixels_normalized: np.ndarray) -> SpineKeypoints:
        if pixels_normalized.ndim != 2 or min(pixels_normalized.shape) < 8:
            return SpineKeypoints(status=STATUS_NOT_EVALUATED, reason="кадр вырожден")
        raw_height = int(pixels_normalized.shape[0])
        iso = resample_isotropic_np(pixels_normalized)
        transform = NetTransform.for_iso_shape(iso.shape, raw_height)
        image = letterbox_iso(iso, transform).unsqueeze(0).to(self.device)
        with torch.no_grad():
            heatmap = self.model(image)[0, 0].cpu().numpy()
        min_spacing = mm_to_iso_px(cfg.SPINE_CENTER_MIN_SPACING_MM) * transform.scale
        peaks_net = extract_peaks(heatmap, min_spacing, cfg.SPINE_UNET_PEAK_THRESHOLD)
        if len(peaks_net) < 1:
            return SpineKeypoints(status=STATUS_NOT_EVALUATED, reason="сеть не нашла ни одного центра")
        centers_iso = transform.net_to_iso(peaks_net)
        centers_raw = transform.net_to_raw(peaks_net)
        return SpineKeypoints(
            status=STATUS_OK,
            centers_raw_xy=centers_raw,
            centers_iso_xy=centers_iso,
        )


_PREDICTOR_CACHE: dict[str, SpineKeypointPredictor] = {}


def load_spine_keypoint_predictor(
    weights_path: Path | str = cfg.SPINE_UNET_WEIGHTS,
) -> SpineKeypointPredictor | None:
    """Предиктор, если веса на месте; иначе None (классическая ветка)."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        return None
    key = str(weights_path.resolve())
    if key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[key] = SpineKeypointPredictor(weights_path)
    return _PREDICTOR_CACHE[key]


# --------------------------------------------------------------------------- #
# Метрики качества точек
# --------------------------------------------------------------------------- #


def match_points_mm(
    predicted_raw: np.ndarray, annotated_raw: np.ndarray, tolerance_mm: float = cfg.SPINE_KEYPOINT_TOLERANCE_MM
) -> dict:
    """Жадный матчинг предсказанных и размеченных центров в миллиметрах."""
    predicted = np.asarray(predicted_raw, dtype=np.float64).reshape(-1, 2)
    annotated = np.asarray(annotated_raw, dtype=np.float64).reshape(-1, 2)
    spacing = np.array([cfg.PIXEL_SPACING_MM_X, cfg.PIXEL_SPACING_MM_Y])
    unmatched_pred = set(range(len(predicted)))
    errors = []
    matched = 0
    for ax, ay in annotated:
        best, best_dist = None, np.inf
        for j in unmatched_pred:
            d = float(np.linalg.norm((predicted[j] - (ax, ay)) * spacing))
            if d < best_dist:
                best, best_dist = j, d
        if best is not None and best_dist <= tolerance_mm:
            unmatched_pred.discard(best)
            errors.append(best_dist)
            matched += 1
    return {
        "n_annotated": int(len(annotated)),
        "n_predicted": int(len(predicted)),
        "n_matched": matched,
        "recall": matched / len(annotated) if len(annotated) else float("nan"),
        "extra_predictions": len(unmatched_pred),
        "mean_error_mm": float(np.mean(errors)) if errors else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Обучение
# --------------------------------------------------------------------------- #


def build_spine_dataset_frame() -> pd.DataFrame:
    """Снимки ПОП с разметкой и фолдом сплита этапа 0."""
    manifest = load_manifest()
    split_payload = json.loads(cfg.SPLIT_JSON.read_text(encoding="utf-8"))
    val_studies = set(split_payload["groups"]["val"])
    representatives = manifest[
        manifest["is_group_representative"].fillna(False) & (manifest["region"] == cfg.REGION_SPINE)
    ].copy()
    representatives["fold"] = [
        "val" if study in val_studies else "train" for study in representatives["study_folder"]
    ]
    return representatives[["dedup_group_id", "abs_path", "study_folder", "fold"]].sort_values("dedup_group_id")


def train_spine_unet(
    *,
    weights_path: Path | str = cfg.SPINE_UNET_WEIGHTS,
    report_path: Path | str | None = cfg.SPINE_UNET_REPORT,
    epochs: int = cfg.SPINE_UNET_EPOCHS,
    batch_size: int = cfg.SPINE_UNET_BATCH_SIZE,
    learning_rate: float = cfg.SPINE_UNET_LEARNING_RATE,
    seed: int = cfg.SEED,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    from .region_cnn import default_device

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = device or default_device()

    frame = build_spine_dataset_frame()
    annotations = load_annotations()
    train_frame = frame[frame["fold"] == "train"]
    val_frame = frame[frame["fold"] == "val"]
    train_data = SpineKeypointDataset(train_frame, annotations, augment=True, seed=seed)
    val_data = SpineKeypointDataset(val_frame, annotations, augment=False)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = SpineUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for images, heatmaps in train_loader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            predicted = model(images)
            # Взвешенный MSE: пики редкие, фон почти нулевой.
            weight = 1.0 + 20.0 * heatmaps
            loss = ((predicted - heatmaps) ** 2 * weight).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
        train_loss = total / len(train_data)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for images, heatmaps in val_loader:
                images, heatmaps = images.to(device), heatmaps.to(device)
                predicted = model(images)
                weight = 1.0 + 20.0 * heatmaps
                val_total += float((((predicted - heatmaps) ** 2) * weight).mean()) * len(images)
        val_loss = val_total / len(val_data)
        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6)})
        if verbose and (epoch + 1) % 5 == 0:
            print(f"эпоха {epoch + 1}/{epochs}: train={train_loss:.5f}, val={val_loss:.5f}", flush=True)

    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, weights_path)

    # Метрики точек по фолдам — предиктором с диска (той же веткой, что рантайм).
    predictor = SpineKeypointPredictor(weights_path, device=device)
    fold_metrics = evaluate_point_metrics(predictor, frame, annotations)
    report = {
        "seed": seed,
        "epochs": epochs,
        "train_images": len(train_data),
        "val_images": len(val_data),
        "history": history,
        "point_metrics": fold_metrics,
    }
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        for fold, metrics in fold_metrics.items():
            print(fold, metrics, flush=True)
    return report


def evaluate_point_metrics(predictor, frame: pd.DataFrame, annotations: dict) -> dict:
    """Метрики совпадения предсказанных точек с разметкой по фолдам."""
    folds: dict[str, list[dict]] = {}
    for row in frame.itertuples():
        result = read_dicom_strict(row.abs_path)
        pixels = normalize(result.pixels, result.tags)
        prediction = predictor.predict(pixels)
        entry = annotations[row.dedup_group_id]
        metrics = match_points_mm(prediction.centers_raw_xy, np.array(entry["centers_xy_raw"]))
        folds.setdefault(row.fold, []).append(metrics)
    summary = {}
    for fold, items in folds.items():
        recalls = [m["recall"] for m in items]
        errors = [m["mean_error_mm"] for m in items if m["n_matched"]]
        summary[fold] = {
            "images": len(items),
            "mean_recall": round(float(np.mean(recalls)), 4),
            "all_points_found_fraction": round(float(np.mean([r == 1.0 for r in recalls])), 4),
            "mean_error_mm": round(float(np.mean(errors)), 3),
            "mean_extra_predictions": round(float(np.mean([m["extra_predictions"] for m in items])), 3),
        }
    return summary


if __name__ == "__main__":
    train_spine_unet()
