"""Этап 4, шаг 6: локальный рантайм CNN-классификатора укладки бедра.

Обучение — только в Google Colab (`notebooks/hip_positioning_cnn.ipynb`,
данные — `artifacts/colab/hip_positioning_dataset.zip` из `hip_export.py`).
Сюда веса только приходят: `load_hip_positioning_predictor()` читает
`artifacts/models/hip_positioning_cnn.pt`, если файла нет — возвращает `None`
без исключений (та же схема честной деградации, что у
`region_cnn.load_region_predictor`/`spine_objects_cnn.load_objects_predictor`).

Препроцессинг переиспользован из `hip_export.preprocess_for_cnn` — тот же код,
что использовался при экспорте датасета в Colab (не дублируется).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import config as cfg
from .hip_export import preprocess_for_cnn
from .region_cnn import IMAGENET_MEAN, IMAGENET_STD, default_device, letterbox_square


def build_model(pretrained: bool = False) -> nn.Module:
    """ResNet18, 1 -> 3 канала на входе (letterbox), 1 выход (логит нарушения)."""
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def to_model_input(iso: np.ndarray, input_size: int = cfg.HIP_CNN_INPUT_SIZE) -> torch.Tensor:
    """Изотропный одноканальный кадр -> тензор 3xNxN с ImageNet-нормировкой."""
    tensor = torch.as_tensor(np.ascontiguousarray(iso), dtype=torch.float32)
    square = letterbox_square(tensor, size=input_size)
    channels = square.expand(3, -1, -1).clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (channels - mean) / std


class HipPositioningPredictor:
    """Вероятность «Некорректная укладка» бедра по нормализованным пикселям."""

    def __init__(
        self,
        weights_path: Path | str = cfg.HIP_POSITIONING_CNN_WEIGHTS,
        device: torch.device | None = None,
        model: nn.Module | None = None,
        input_size: int = cfg.HIP_CNN_INPUT_SIZE,
    ):
        self.device = device or default_device()
        if model is None:
            payload = torch.load(Path(weights_path), map_location=self.device, weights_only=True)
            input_size = int(payload.get("input_size", input_size))
            model = build_model(pretrained=False)
            model.load_state_dict(payload["state_dict"])
        self.input_size = input_size
        self.model = model.to(self.device).eval()

    def predict_pixels(self, pixels_normalized: np.ndarray, side: str | None) -> float | None:
        """Вероятность нарушения укладки; `None`, если препроцессинг упал.

        Без исключений наружу (инвариант 5 этапа 4: честная деградация вместо
        падения инференса на одном файле).
        """
        try:
            iso = preprocess_for_cnn(pixels_normalized, side)
            tensor = to_model_input(iso, self.input_size).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return float(torch.sigmoid(self.model(tensor)[0, 0]))
        except Exception:  # noqa: BLE001 — честная деградация, не падение инференса
            return None


_PREDICTOR_CACHE: dict[str, HipPositioningPredictor] = {}


def load_hip_positioning_predictor(
    weights_path: Path | str = cfg.HIP_POSITIONING_CNN_WEIGHTS,
) -> HipPositioningPredictor | None:
    """Предиктор, если веса на месте; иначе `None` (CNN-ветка не подключена).

    Веса обучаются только в Colab-ноутбуке и кладутся в `artifacts/models/`
    вручную — локального обучения здесь нет и не будет.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        return None
    key = str(weights_path.resolve())
    if key not in _PREDICTOR_CACHE:
        try:
            _PREDICTOR_CACHE[key] = HipPositioningPredictor(weights_path)
        except Exception:  # noqa: BLE001 — битые/несовместимые веса не должны падать рантайм
            return None
    return _PREDICTOR_CACHE[key]
