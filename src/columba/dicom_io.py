"""Шаг 2: чтение DICOM и нормализация интенсивностей.

Ключевое требование ТЗ: битые/нечитаемые файлы не выбрасываются, а получают
статус Failure и остаются строкой в манифесте. Поэтому `read_dicom` никогда
не бросает исключение наружу.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

STATUS_SUCCESS = "Success"
STATUS_FAILURE = "Failure"

# Теги, которые реально присутствуют в этой выгрузке и полезны дальше.
TAGS_OF_INTEREST: tuple[str, ...] = (
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesNumber",
    "InstanceNumber",
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "StudyDescription",
    "SeriesDescription",
    "BodyPartExamined",
    "ViewPosition",
    "Laterality",
    "PatientOrientation",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "PixelRepresentation",
    "PhotometricInterpretation",
)


@dataclass
class DicomRead:
    """Результат чтения одного файла."""

    path: Path
    status: str
    error: str = ""
    pixels: np.ndarray | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_SUCCESS


def _tag_value(dataset: pydicom.Dataset, name: str) -> Any:
    value = dataset.get(name, None)
    if value is None:
        return None
    if isinstance(value, pydicom.multival.MultiValue):
        return "\\".join(str(v) for v in value)
    if isinstance(value, (int, float)):
        return value
    return str(value)


def read_dicom(path: str | Path, *, load_pixels: bool = True) -> DicomRead:
    """Прочитать файл. Любая ошибка -> статус Failure, а не исключение.

    Файл без 128-байтного преамбула/метаинформации (такое случается при
    небрежной выгрузке из PACS) не проходит `force=False`; тогда делается
    вторая попытка с `force=True`, и её результат принимается только если
    пиксели реально декодировались — иначе force прочитал бы и мусор.
    """
    path = Path(path)
    try:
        return _read_attempt(path, force=False, load_pixels=load_pixels)
    except Exception as first_exc:  # noqa: BLE001 — по требованию ТЗ ловим всё
        try:
            return _read_attempt(path, force=True, load_pixels=load_pixels)
        except Exception:  # noqa: BLE001
            # В отчёт идёт исходная ошибка: она описывает файл честнее,
            # чем каскад от force-попытки.
            return DicomRead(
                path=path,
                status=STATUS_FAILURE,
                error=f"{type(first_exc).__name__}: {first_exc}",
            )


def _read_attempt(path: Path, *, force: bool, load_pixels: bool) -> DicomRead:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = pydicom.dcmread(path, force=force)
        tags = {name: _tag_value(dataset, name) for name in TAGS_OF_INTEREST}
        # При force=True пиксели декодируются всегда, даже когда наружу не
        # нужны: успешный decode — единственное доказательство, что force
        # прочитал DICOM, а не произвольные байты.
        pixels = None
        if load_pixels or force:
            pixels = np.asarray(dataset.pixel_array)
        if not load_pixels:
            pixels = None
    return DicomRead(path=path, status=STATUS_SUCCESS, pixels=pixels, tags=tags)


def read_dicom_strict(path: str | Path, *, load_pixels: bool = True) -> DicomRead:
    """Как `read_dicom`, но бросает исключение при статусе Failure.

    Только для eval/обучающих циклов, которые сами не готовы обрабатывать
    Failure (в отличие от `pipeline.run_stage0`, где Failure — законная строка
    манифеста, а не повод падать): там молчаливый `AttributeError` в
    `normalize(None, ...)` хуже честного падения с путём и причиной — метрики
    на тихо укороченной выборке вводили бы в заблуждение сильнее.
    """
    result = read_dicom(path, load_pixels=load_pixels)
    if not result.ok:
        raise RuntimeError(f"read_dicom: не удалось прочитать {result.path}: {result.error}")
    return result


def normalize(pixels: np.ndarray, tags: dict[str, Any] | None = None) -> np.ndarray:
    """Привести интенсивности к float32 в [0, 1].

    MONOCHROME1 инвертируется, чтобы «ярче = плотнее» во всех снимках.
    Масштаб берётся от разрядности (BitsStored), а не от max по кадру, —
    иначе одинаковые ткани на разных снимках получат разную яркость.
    """
    tags = tags or {}
    array = np.asarray(pixels).astype(np.float32)

    bits_stored = tags.get("BitsStored")
    if isinstance(bits_stored, (int, float)) and bits_stored > 0:
        scale = float(2 ** int(bits_stored) - 1)
    else:
        scale = float(np.iinfo(pixels.dtype).max) if np.issubdtype(pixels.dtype, np.integer) else 1.0

    array = array / scale
    if str(tags.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        array = 1.0 - array
    return np.clip(array, 0.0, 1.0, out=array)
