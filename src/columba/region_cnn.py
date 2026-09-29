"""Этап 1: CNN-классификатор региона/стороны и арбитраж с эвристикой.

Три класса: ПОП / левое бедро / правое бедро. Сторона наружу не идёт
(в выходной таблице лево и право не различаются), но нужна модулю бедра
на этапе 4 для ориентации кропов.

CNN — страховка эвристики по ширине кадра, а не её замена: правило арбитража
(`arbitrate_region`) отдаёт приоритет эвристике на стандартных ширинах и CNN
на нестандартных. Классификатор смотрит только на пиксели: ни тегов, ни имён
файлов (в закрытом тесте их не будет).

Сетка анизотропная (1,05×0,6 мм), поэтому перед сетью кадр ресемплируется
к изотропному пикселю; коэффициент берётся из config, литералы запрещены
тестом единственности pixel spacing.
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
from .dicom_io import read_dicom, normalize
from .regions import SIDE_LEFT, SIDE_RIGHT, hip_side

# ImageNet-нормировка: сеть предобучена на таких статистиках.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_TO_INDEX = {name: index for index, name in enumerate(cfg.CNN_CLASSES)}

# Аугментации обучения. Повороты малые: наклон оси — признак нарушения для
# этапа 2, классификатору региона он безразличен, но раскачивать сильнее
# незачем. Горизонтальный флип меняет сторону бедра, поэтому выполняется
# ТОЛЬКО с одновременным переключением метки (см. RegionDataset).
AUG_MAX_ROTATION_DEG = 5.0
AUG_BRIGHTNESS = 0.10
AUG_CONTRAST = 0.10
AUG_FLIP_PROBABILITY = 0.5

FLIPPED_CLASS = {
    cfg.CNN_CLASS_SPINE: cfg.CNN_CLASS_SPINE,
    cfg.CNN_CLASS_HIP_LEFT: cfg.CNN_CLASS_HIP_RIGHT,
    cfg.CNN_CLASS_HIP_RIGHT: cfg.CNN_CLASS_HIP_LEFT,
}


def default_device() -> torch.device:
    """CUDA, если доступна (этап 7 — сервис на GPU, если есть; ранее эта
    функция проверяла только MPS, поэтому в Linux/Docker-окружении с GPU
    всегда уходила на CPU, даже когда видеокарта проброшена в контейнер),
    иначе MPS (Apple Silicon), иначе CPU — честная деградация, инференс
    и так работает на CPU у всех предикторов этого модуля."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Препроцессинг
# --------------------------------------------------------------------------- #


def resample_to_isotropic(pixels: np.ndarray) -> torch.Tensor:
    """Растянуть кадр по Y до изотропного пикселя (коэффициент из config)."""
    tensor = torch.as_tensor(np.ascontiguousarray(pixels), dtype=torch.float32)
    height, width = tensor.shape[-2], tensor.shape[-1]
    new_height = int(round(height * cfg.ANISOTROPY_Y_OVER_X))
    return F.interpolate(
        tensor.reshape(1, 1, height, width),
        size=(new_height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def letterbox_square(image: torch.Tensor, size: int = cfg.CNN_INPUT_SIZE) -> torch.Tensor:
    """Вписать кадр в квадрат size×size с сохранением пропорций (паддинг нулями)."""
    height, width = image.shape
    scale = size / max(height, width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    resized = F.interpolate(
        image.reshape(1, 1, height, width), size=(new_h, new_w), mode="bilinear", align_corners=False
    )[0, 0]
    canvas = torch.zeros(size, size, dtype=image.dtype)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def to_model_input(isotropic: torch.Tensor) -> torch.Tensor:
    """Изотропный кадр [0,1] -> тензор 3×N×N с ImageNet-нормировкой."""
    square = letterbox_square(isotropic)
    channels = square.expand(3, -1, -1).clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (channels - mean) / std


def prepare_pixels(pixels: np.ndarray, tags: dict | None = None) -> torch.Tensor:
    """Полный препроцессинг сырых пикселей DICOM до входа сети."""
    return to_model_input(resample_to_isotropic(normalize(pixels, tags)))


# --------------------------------------------------------------------------- #
# Датасет
# --------------------------------------------------------------------------- #


def build_region_dataset(manifest: pd.DataFrame, split_payload: dict) -> pd.DataFrame:
    """Собрать таблицу обучения CNN: уникальные изображения + метки региона/стороны.

    Метки — из эвристики этапа 0 (подтверждена организаторами и сверена
    глазами на спорных кейсах). Кадры 248 px (эндопротезы, регион `unknown`
    в манифесте) — это бёдра; сторона для них берётся из того же признака
    латеральности напрямую по пикселям.

    Сплит — тот же, что в этапе 0 (по исследованиям): исследования вне
    train/val (без таргета) уходят в train — таргета качества у них нет,
    а метка региона есть, и в val им появляться незачем.
    """
    representatives = manifest[manifest["is_group_representative"].fillna(False)].copy()

    labels: list[str | None] = []
    for row in representatives.itertuples():
        if row.region == cfg.REGION_SPINE:
            labels.append(cfg.CNN_CLASS_SPINE)
        elif row.region == cfg.REGION_HIP:
            side = row.hip_side if pd.notna(row.hip_side) else None
            labels.append(_hip_class(side, row.abs_path))
        else:
            # Нестандартная ширина: наблюдалась только у эндопротезов (бедро).
            labels.append(_hip_class(None, row.abs_path))
    representatives["cnn_label"] = labels
    representatives = representatives[representatives["cnn_label"].notna()].copy()

    val_studies = set(split_payload["groups"]["val"])
    representatives["cnn_split"] = [
        "val" if study in val_studies else "train" for study in representatives["study_folder"]
    ]
    return representatives[["abs_path", "study_folder", "dedup_group_id", "region", "cnn_label", "cnn_split"]]


def _hip_class(side: str | None, abs_path: str) -> str | None:
    if side is None:
        result = read_dicom(abs_path)
        if not result.ok:
            return None
        estimate = hip_side(normalize(result.pixels, result.tags))
        side = estimate.side
    if side == SIDE_LEFT:
        return cfg.CNN_CLASS_HIP_LEFT
    if side == SIDE_RIGHT:
        return cfg.CNN_CLASS_HIP_RIGHT
    return None


class RegionDataset(Dataset):
    """Изотропные кадры в памяти; аугментации — на лету, только для train."""

    def __init__(self, frame: pd.DataFrame, *, augment: bool, seed: int = cfg.CNN_SEED):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.rng = random.Random(seed)
        self.images: list[torch.Tensor] = []
        for row in self.frame.itertuples():
            result = read_dicom(row.abs_path)
            self.images.append(resample_to_isotropic(normalize(result.pixels, result.tags)))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        label = self.frame.loc[index, "cnn_label"]
        if self.augment:
            image, label = self._augment(image, label)
        return to_model_input(image), CLASS_TO_INDEX[label]

    def _augment(self, image: torch.Tensor, label: str) -> tuple[torch.Tensor, str]:
        # Флип меняет сторону бедра -> метка переключается синхронно.
        if self.rng.random() < AUG_FLIP_PROBABILITY:
            image = torch.flip(image, dims=[1])
            label = FLIPPED_CLASS[label]
        angle = self.rng.uniform(-AUG_MAX_ROTATION_DEG, AUG_MAX_ROTATION_DEG)
        image = _rotate(image, angle)
        brightness = 1.0 + self.rng.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS)
        contrast = 1.0 + self.rng.uniform(-AUG_CONTRAST, AUG_CONTRAST)
        mean = image.mean()
        image = torch.clamp((image - mean) * contrast + mean * brightness, 0.0, 1.0)
        return image, label


def _rotate(image: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Повернуть изотропный кадр (после ресемплинга поворот честный)."""
    theta_rad = float(np.deg2rad(angle_deg))
    cos, sin = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, (1, 1, *image.shape), align_corners=False)
    return F.grid_sample(
        image.reshape(1, 1, *image.shape), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0, 0]


# --------------------------------------------------------------------------- #
# Модель и обучение
# --------------------------------------------------------------------------- #


def build_model(pretrained: bool = True) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(cfg.CNN_CLASSES))
    return model


def train_region_cnn(
    dataset_frame: pd.DataFrame,
    *,
    weights_path: Path | str = cfg.REGION_CNN_WEIGHTS,
    report_path: Path | str | None = cfg.REGION_CNN_REPORT,
    epochs: int = cfg.CNN_EPOCHS,
    batch_size: int = cfg.CNN_BATCH_SIZE,
    learning_rate: float = cfg.CNN_LEARNING_RATE,
    seed: int = cfg.CNN_SEED,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Обучить ResNet18 и сохранить веса + отчёт. Возвращает отчёт."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = device or default_device()

    train_frame = dataset_frame[dataset_frame["cnn_split"] == "train"]
    val_frame = dataset_frame[dataset_frame["cnn_split"] == "val"]
    train_data = RegionDataset(train_frame, augment=True, seed=seed)
    val_data = RegionDataset(val_frame, augment=False)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
        train_loss = total_loss / len(train_data)
        val_accuracy, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4), "val_accuracy": round(val_accuracy, 4)})
        if verbose:
            print(f"эпоха {epoch + 1}/{epochs}: loss={train_loss:.4f}, val acc={val_accuracy:.4f}", flush=True)

    train_eval_loader = DataLoader(RegionDataset(train_frame, augment=False), batch_size=batch_size)
    train_accuracy, train_confusion = evaluate(model, train_eval_loader, device)
    val_accuracy, val_confusion = evaluate(model, val_loader, device)

    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": list(cfg.CNN_CLASSES)}, weights_path)

    report = {
        "classes": list(cfg.CNN_CLASSES),
        "seed": seed,
        "epochs": epochs,
        "train_images": len(train_data),
        "val_images": len(val_data),
        "train_accuracy": round(train_accuracy, 4),
        "val_accuracy": round(val_accuracy, 4),
        "train_confusion": train_confusion,
        "val_confusion": val_confusion,
        "history": history,
    }
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    correct, total = 0, 0
    confusion: dict[str, dict[str, int]] = {
        true: {pred: 0 for pred in cfg.CNN_CLASSES} for true in cfg.CNN_CLASSES
    }
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            predicted = logits.argmax(dim=1).cpu()
            for true_idx, pred_idx in zip(labels.tolist(), predicted.tolist()):
                confusion[cfg.CNN_CLASSES[true_idx]][cfg.CNN_CLASSES[pred_idx]] += 1
            correct += int((predicted == labels).sum())
            total += len(labels)
    return (correct / total if total else float("nan")), confusion


# --------------------------------------------------------------------------- #
# Инференс
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CnnPrediction:
    label: str  # один из CNN_CLASSES
    probs: dict[str, float]

    @property
    def region(self) -> str:
        return cfg.REGION_SPINE if self.label == cfg.CNN_CLASS_SPINE else cfg.REGION_HIP

    @property
    def side(self) -> str | None:
        if self.label == cfg.CNN_CLASS_HIP_LEFT:
            return SIDE_LEFT
        if self.label == cfg.CNN_CLASS_HIP_RIGHT:
            return SIDE_RIGHT
        return None


class RegionCnnPredictor:
    """Загрузка весов и предсказание по сырым пикселям DICOM."""

    def __init__(self, weights_path: Path | str = cfg.REGION_CNN_WEIGHTS, device: torch.device | None = None):
        self.device = device or default_device()
        payload = torch.load(Path(weights_path), map_location=self.device, weights_only=True)
        if payload["classes"] != list(cfg.CNN_CLASSES):
            raise ValueError(f"веса обучены на классах {payload['classes']}, ожидались {list(cfg.CNN_CLASSES)}")
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()

    def predict_pixels(self, pixels: np.ndarray, tags: dict | None = None) -> CnnPrediction:
        tensor = prepare_pixels(pixels, tags).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu()
        label = cfg.CNN_CLASSES[int(probs.argmax())]
        return CnnPrediction(label=label, probs={name: float(p) for name, p in zip(cfg.CNN_CLASSES, probs)})


_PREDICTOR_CACHE: dict[str, "RegionCnnPredictor"] = {}


def load_region_predictor(
    weights_path: Path | str = cfg.REGION_CNN_WEIGHTS,
) -> RegionCnnPredictor | None:
    """Предиктор, если веса на месте; иначе None (ветка «только эвристика»).

    Загруженные веса кэшируются по пути: инференс зовёт эту функцию на каждый
    входной каталог, а модель одна.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        return None
    key = str(weights_path.resolve())
    if key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[key] = RegionCnnPredictor(weights_path)
    return _PREDICTOR_CACHE[key]


# --------------------------------------------------------------------------- #
# Арбитраж эвристика <-> CNN (шаг 5, правило зафиксировано в decisions)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArbitratedRegion:
    region: str  # spine | hip — итоговый регион
    side: str | None  # сторона бедра (итоговая)
    method: str  # REGION_METHOD_HEURISTIC | REGION_METHOD_CNN
    cnn_label: str | None
    disagreement: bool  # эвристика и CNN разошлись (лог с предупреждением)


def arbitrate_region(
    width: int | None,
    heuristic_region: str,
    heuristic_side: str | None,
    cnn: CnnPrediction | None,
) -> ArbitratedRegion:
    """Свести ответы эвристики и CNN в итоговый регион/сторону.

    Правило (шаг 5 этапа 1):
    * стандартная ширина (300/280/248 px) — эвристика первична, CNN
      подтверждает; при расхождении приоритет у эвристики (она подтверждена
      организаторами), а расхождение поднимается флагом `disagreement`;
      если эвристика сторону бедра не дала (248 px: сырой регион `unknown`,
      признак латеральности не считался), сторона берётся из CNN — это
      заполнение вакуума, а не конфликт веток, `disagreement` не поднимается;
    * нестандартная ширина — эвристика не определена, приоритет у CNN;
    * CNN недоступна (нет весов) — чистая эвристика с фолбэком региона.
    """
    standard = width is not None and int(width) in cfg.REGION_BY_STANDARD_WIDTH
    if standard:
        region = cfg.REGION_BY_STANDARD_WIDTH[int(width)]
        side = heuristic_side if region == cfg.REGION_HIP else None
        side_backfilled = False
        if region == cfg.REGION_HIP and side is None and cnn is not None:
            side = cnn.side
            side_backfilled = True
        disagreement = cnn is not None and (
            cnn.region != region
            or (
                region == cfg.REGION_HIP
                and not side_backfilled
                and side is not None
                and cnn.side != side
            )
        )
        return ArbitratedRegion(
            region=region,
            side=side,
            method=cfg.REGION_METHOD_HEURISTIC,
            cnn_label=cnn.label if cnn else None,
            disagreement=bool(disagreement),
        )
    if cnn is not None:
        return ArbitratedRegion(
            region=cnn.region,
            side=cnn.side,
            method=cfg.REGION_METHOD_CNN,
            cnn_label=cnn.label,
            disagreement=heuristic_region not in (cfg.REGION_UNKNOWN, cnn.region),
        )
    # Ни стандартной ширины, ни CNN: фолбэк региона из этапа 0.
    region = heuristic_region if heuristic_region != cfg.REGION_UNKNOWN else cfg.REGION_SUBMISSION_FALLBACK
    return ArbitratedRegion(
        region=region,
        side=heuristic_side if region == cfg.REGION_HIP else None,
        method=cfg.REGION_METHOD_HEURISTIC,
        cnn_label=None,
        disagreement=False,
    )
