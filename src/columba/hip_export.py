"""Этап 4, шаг 6: экспорт датасета бедра для обучения CNN в Google Colab.

Локально ничего не обучается (см. `stages/stage_4.md`: обучение — только в
Google Colab): этот модуль только готовит данные — по одному PNG
на уникальное размеченное изображение бедра, тот же препроцессинг, что у
ResNet18-пайплайна этапа 1 (`region_cnn.resample_to_isotropic`), плюс
отражение правого бедра к виду левого (тот же приём, что `hip_grids.py`
и `hip_landmarks.compute_hip_signals`).

Обезличивание: `study_folder` (UID папки выгрузки PACS) в архив не попадает —
только стабильный локальный id вида `s000`. Связь `study_folder -> anon_id`
хранится ТОЛЬКО в `artifacts/colab/study_map.csv`, вне архива. Никаких
DICOM-тегов, ФИО или UID исследований в архиве нет вообще — таблица `labels.csv`
собрана из уже посчитанных `targets.csv`/`manifest.csv`, а не из тегов файла.

Запуск: `uv run python -m columba.hip_export`
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config as cfg
from .dicom_io import normalize, read_dicom
from .inventory import load_manifest
from .regions import SIDE_RIGHT
from .spine_keypoints import resample_isotropic_np

# Колонки итогового labels.csv. Намеренно без UID/тегов DICOM (проверяется
# тестом test_hip_export.py::test_labels_csv_has_no_identifying_columns).
LABELS_COLUMNS: tuple[str, ...] = (
    "dedup_group_id",
    "study_folder",  # обезличенный id (s000...), НЕ реальный UID папки
    "split",
    "software_version",
    "hip_side",
    "label_hip_positioning",
    "label_hip_roi",
    "has_mask_rect",
)


# --------------------------------------------------------------------------- #
# Препроцессинг — общая функция для экспорта И локального рантайма (hip_cnn.py)
# --------------------------------------------------------------------------- #


def preprocess_for_cnn(pixels_normalized: np.ndarray, side: str | None) -> np.ndarray:
    """Нормализованные пиксели [0,1] -> изотропный кадр, правое бедро отражено.

    Единственная точка препроцессинга для CNN-ветки бедра: тот же изотропный
    ресемплинг, что у `region_cnn`/`spine_keypoints` (растяжение Y на
    `ANISOTROPY_Y_OVER_X`), затем зеркалирование правого бедра к виду левого
    (как `hip_grids.render_grid`/`hip_landmarks`). Используется и при экспорте
    в PNG для Colab, и в `hip_cnn.HipPositioningPredictor` — не дублировать.
    Детерминирована: одинаковый вход всегда даёт одинаковый выход.
    """
    iso = resample_isotropic_np(np.asarray(pixels_normalized, dtype=np.float32))
    if side == SIDE_RIGHT:
        iso = np.fliplr(iso)
    return np.ascontiguousarray(iso, dtype=np.float32)


def to_uint8_png_array(iso: np.ndarray) -> np.ndarray:
    """Изотропный кадр [0,1] -> 8-битный массив для сохранения в PNG."""
    clipped = np.clip(iso, 0.0, 1.0)
    return np.round(clipped * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Сбор таблицы: уникальные размеченные изображения бедра + обезличивание
# --------------------------------------------------------------------------- #


def hip_export_frame() -> pd.DataFrame:
    """Одна строка на уникальное размеченное изображение бедра (has_target).

    `abs_path` идёт из общего загрузчика `inventory.load_manifest` — он
    пересчитывает её из `relative_path` + `STUDIES_DIR` текущей машины, а не
    берёт из файла манифеста (записан на машине сборки выгрузки, там же и
    может быть битым; тот же приём, что `hip_grids.hip_targets_frame`).
    """
    targets = pd.read_csv(cfg.TARGETS_CSV)
    manifest = load_manifest()
    representatives = manifest[manifest["is_group_representative"].fillna(False)][
        ["dedup_group_id", "abs_path", "software_version", "has_mask_rect"]
    ]
    frame = targets[(targets["region"] == cfg.REGION_HIP) & targets["has_target"].fillna(False)].merge(
        representatives, on="dedup_group_id", how="left"
    )
    return frame.sort_values("dedup_group_id").reset_index(drop=True)


def anonymize_study_folders(study_folders: pd.Series) -> dict[str, str]:
    """Стабильная детерминированная карта `study_folder -> sNNN` (сортировка по UID)."""
    unique = sorted(study_folders.dropna().unique())
    width = max(3, len(str(max(0, len(unique) - 1))))
    return {study: f"s{index:0{width}d}" for index, study in enumerate(unique)}


def build_labels_frame(frame: pd.DataFrame, study_map: dict[str, str]) -> pd.DataFrame:
    labels = pd.DataFrame({column: frame.get(column) for column in LABELS_COLUMNS})
    labels["study_folder"] = frame["study_folder"].map(study_map)
    labels["has_mask_rect"] = frame["has_mask_rect"].fillna(False).astype(bool)
    labels["label_hip_positioning"] = frame["label_hip_positioning"].astype(int)
    labels["label_hip_roi"] = frame["label_hip_roi"].astype(int)
    return labels[list(LABELS_COLUMNS)]


# --------------------------------------------------------------------------- #
# Экспорт
# --------------------------------------------------------------------------- #


def export_dataset(
    *,
    zip_path: Path | str = cfg.HIP_EXPORT_ZIP,
    study_map_path: Path | str = cfg.HIP_EXPORT_STUDY_MAP_CSV,
    verbose: bool = True,
) -> Path:
    """Собрать `hip_positioning_dataset.zip`: PNG на снимок + labels.csv.

    `study_map.csv` (study_folder -> anon_id) пишется РЯДОМ с архивом, но
    НЕ внутрь него — связь с реальными UID папок остаётся только локально.
    """
    frame = hip_export_frame()
    study_map = anonymize_study_folders(frame["study_folder"])
    labels = build_labels_frame(frame, study_map)

    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    study_map_path = Path(study_map_path)
    study_map_path.parent.mkdir(parents=True, exist_ok=True)

    with study_map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_folder", "anon_id"])
        for study, anon_id in sorted(study_map.items(), key=lambda kv: kv[1]):
            writer.writerow([study, anon_id])

    n_written = 0
    n_failed = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(cfg.HIP_EXPORT_LABELS_CSV_NAME, labels.to_csv(index=False))
        for row in frame.itertuples():
            result = read_dicom(row.abs_path)
            if not result.ok:
                n_failed += 1
                continue
            pixels = normalize(result.pixels, result.tags)
            iso = preprocess_for_cnn(pixels, row.hip_side if pd.notna(row.hip_side) else None)
            png_array = to_uint8_png_array(iso)
            image = Image.fromarray(png_array, mode="L")
            buffer_path = f"{cfg.HIP_EXPORT_IMAGES_DIRNAME}/{row.dedup_group_id}.png"
            import io

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            archive.writestr(buffer_path, buffer.getvalue())
            n_written += 1

    size_bytes = zip_path.stat().st_size
    if verbose:
        print(
            f"{zip_path}: {n_written} PNG, {n_failed} не прочитано, "
            f"{size_bytes / 1024 / 1024:.2f} МБ; study_map -> {study_map_path} ({len(study_map)} исследований)",
            flush=True,
        )
    return zip_path


def main() -> None:
    export_dataset()


if __name__ == "__main__":
    main()
