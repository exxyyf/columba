"""Этап 2, шаг 6 (CNN-ветка): классификатор «посторонние предметы» на кадре ПОП.

Rule-based детектор упирается в AUC ~0.77 (тонкие дуги рёбер неотличимы от
цепочек простыми порогами), поэтому по плану шага 6 обучается маленький
классификатор целого кадра: ResNet18, бинарный выход, вероятность нарушения.

Данные: реальные метки `spine_artifacts` train-фолда + синтетические позитивы
шага 7 (вставка цепочек/клипс в качественные снимки train-фолда). Синтетика
в валидацию не попадает; метрики считаются только на реальных метках.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import config as cfg
from .dicom_io import normalize, read_dicom_strict
from .inventory import load_manifest
from .region_cnn import IMAGENET_MEAN, IMAGENET_STD, default_device
from .spine_keypoint_net import NetTransform
from .spine_keypoints import resample_isotropic_np
from .spine_synth import insert_synthetic_metal, rotate_isotropic

AUG_MAX_ROTATION_DEG = 5.0
AUG_BRIGHTNESS = 0.10
AUG_CONTRAST = 0.10


def _letterbox(iso: np.ndarray) -> torch.Tensor:
    """Изотропный кадр -> 3 x H x W с ImageNet-нормировкой."""
    import torch.nn.functional as F

    tensor = torch.as_tensor(np.ascontiguousarray(iso), dtype=torch.float32)
    height, width = tensor.shape
    scale = min(cfg.SPINE_OBJECTS_CNN_INPUT_HEIGHT / height, cfg.SPINE_OBJECTS_CNN_INPUT_WIDTH / width)
    new_h, new_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    resized = F.interpolate(
        tensor.reshape(1, 1, height, width), size=(new_h, new_w), mode="bilinear", align_corners=False
    )[0, 0]
    canvas = torch.zeros(cfg.SPINE_OBJECTS_CNN_INPUT_HEIGHT, cfg.SPINE_OBJECTS_CNN_INPUT_WIDTH)
    top = (cfg.SPINE_OBJECTS_CNN_INPUT_HEIGHT - new_h) // 2
    left = (cfg.SPINE_OBJECTS_CNN_INPUT_WIDTH - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    channels = canvas.expand(3, -1, -1).clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (channels - mean) / std


@dataclass(frozen=True)
class _Sample:
    dedup_group_id: str
    label: int
    synthetic: bool  # вставить синтетический металл (label принудительно 1)


class SpineObjectsDataset(Dataset):
    """Реальные кадры + синтетические позитивы (только train)."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        augment: bool,
        synthetic: bool,
        seed: int = cfg.SEED,
    ):
        self.augment = augment
        self.rng = random.Random(seed)
        self.raw: dict[str, np.ndarray] = {}
        samples: list[_Sample] = []
        for row in frame.itertuples():
            result = read_dicom_strict(row.abs_path)
            self.raw[row.dedup_group_id] = normalize(result.pixels, result.tags)
            samples.append(_Sample(row.dedup_group_id, int(row.label), synthetic=False))
        if synthetic:
            negatives = [row.dedup_group_id for row in frame.itertuples() if int(row.label) == 0]
            n_synth = int(len(samples) * cfg.SPINE_OBJECTS_SYNTH_FRACTION)
            for gid in (negatives * ((n_synth // max(1, len(negatives))) + 1))[:n_synth]:
                samples.append(_Sample(gid, 1, synthetic=True))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        pixels = self.raw[sample.dedup_group_id]
        if sample.synthetic:
            rng = np.random.default_rng(self.rng.randrange(2**31))
            pixels = insert_synthetic_metal(pixels, rng)
        iso = resample_isotropic_np(pixels)
        if self.augment:
            angle = self.rng.uniform(-AUG_MAX_ROTATION_DEG, AUG_MAX_ROTATION_DEG)
            iso = rotate_isotropic(iso, angle)
            if self.rng.random() < 0.5:
                iso = iso[:, ::-1].copy()
            brightness = 1.0 + self.rng.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS)
            contrast = 1.0 + self.rng.uniform(-AUG_CONTRAST, AUG_CONTRAST)
            mean = float(iso.mean())
            iso = np.clip((iso - mean) * contrast + mean * brightness, 0.0, 1.0)
        return _letterbox(iso), torch.tensor(float(sample.label), dtype=torch.float32)


def build_model(pretrained: bool = True) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def objects_dataset_frame() -> pd.DataFrame:
    """Снимки ПОП с меткой spine_artifacts и фолдом сплита этапа 0."""
    manifest = load_manifest()
    targets = pd.read_csv(cfg.TARGETS_CSV)
    split_payload = json.loads(cfg.SPLIT_JSON.read_text(encoding="utf-8"))
    val_studies = set(split_payload["groups"]["val"])
    spine = targets[(targets["region"] == cfg.REGION_SPINE) & targets["has_target"].fillna(False)]
    representatives = manifest[manifest["is_group_representative"].fillna(False)]
    frame = spine.merge(
        representatives[["dedup_group_id", "abs_path", "software_version"]],
        on="dedup_group_id",
    )  # study_folder уже есть в targets
    frame["label"] = frame["spine_artifacts"].astype(int)
    frame["fold"] = ["val" if s in val_studies else "train" for s in frame["study_folder"]]
    return frame[["dedup_group_id", "abs_path", "study_folder", "software_version", "label", "fold"]].sort_values(
        "dedup_group_id"
    )


def train_objects_cnn(
    train_frame: pd.DataFrame,
    *,
    weights_path: Path | str | None = cfg.SPINE_OBJECTS_CNN_WEIGHTS,
    epochs: int = cfg.SPINE_OBJECTS_CNN_EPOCHS,
    batch_size: int = cfg.SPINE_OBJECTS_CNN_BATCH_SIZE,
    learning_rate: float = cfg.SPINE_OBJECTS_CNN_LEARNING_RATE,
    seed: int = cfg.SEED,
    device: torch.device | None = None,
    verbose: bool = True,
) -> nn.Module:
    """Обучить классификатор на переданном train-фрейме; вернуть модель."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = device or default_device()

    data = SpineObjectsDataset(train_frame, augment=True, synthetic=True, seed=seed)
    labels = [s.label for s in data.samples]
    pos_weight = torch.tensor([(len(labels) - sum(labels)) / max(1, sum(labels))]).to(device)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))

    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(1), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(targets)
        if verbose and (epoch + 1) % 4 == 0:
            print(f"эпоха {epoch + 1}/{epochs}: loss={total / len(data):.4f}", flush=True)

    if weights_path is not None:
        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, weights_path)
    return model


class SpineObjectsPredictor:
    """Вероятность «на кадре посторонние предметы» по нормализованным пикселям."""

    def __init__(
        self,
        weights_path: Path | str = cfg.SPINE_OBJECTS_CNN_WEIGHTS,
        device: torch.device | None = None,
        model: nn.Module | None = None,
    ):
        self.device = device or default_device()
        if model is None:
            payload = torch.load(Path(weights_path), map_location=self.device, weights_only=True)
            model = build_model(pretrained=False)
            model.load_state_dict(payload["state_dict"])
        self.model = model.to(self.device).eval()

    def predict_proba(self, pixels_normalized: np.ndarray) -> float:
        tensor = _letterbox(resample_isotropic_np(pixels_normalized)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(torch.sigmoid(self.model(tensor)[0, 0]))


_PREDICTOR_CACHE: dict[str, SpineObjectsPredictor] = {}


def load_objects_predictor(
    weights_path: Path | str = cfg.SPINE_OBJECTS_CNN_WEIGHTS,
) -> SpineObjectsPredictor | None:
    """Предиктор, если веса на месте; иначе None (rule-based ветка)."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        return None
    key = str(weights_path.resolve())
    if key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[key] = SpineObjectsPredictor(weights_path)
    return _PREDICTOR_CACHE[key]


def main() -> None:
    frame = objects_dataset_frame()
    model = train_objects_cnn(frame[frame["fold"] == "train"])
    predictor = SpineObjectsPredictor(model=model)
    scores = {}
    for row in frame.itertuples():
        result = read_dicom_strict(row.abs_path)
        scores[row.dedup_group_id] = predictor.predict_proba(normalize(result.pixels, result.tags))
    frame = frame.assign(probability=[scores[g] for g in frame["dedup_group_id"]])
    report = {"folds": {}}
    for fold, chunk in frame.groupby("fold"):
        pos = chunk[chunk.label == 1]["probability"].values
        neg = chunk[chunk.label == 0]["probability"].values
        auc = float(np.mean([[(p > n) + 0.5 * (p == n) for n in neg] for p in pos])) if len(pos) and len(neg) else None
        report["folds"][fold] = {"n": len(chunk), "positives": int(chunk.label.sum()), "auc": auc}
    Path(cfg.SPINE_OBJECTS_CNN_REPORT).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
