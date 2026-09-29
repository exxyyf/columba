"""Этап 4, шаг 6: экспорт датасета бедра для Colab и локальный CNN-рантайм.

Обучение здесь не проверяется (его нет — обучение только в Colab): тесты
покрывают детерминизм препроцессинга, зеркалирование правого бедра,
отсутствие UID/тегов в выгружаемых метках и честную деградацию рантайма
без весов.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from columba import config as cfg
from columba.hip_cnn import load_hip_positioning_predictor
from columba.hip_export import (
    LABELS_COLUMNS,
    anonymize_study_folders,
    build_labels_frame,
    hip_export_frame,
    preprocess_for_cnn,
    to_uint8_png_array,
)
from columba.regions import SIDE_LEFT, SIDE_RIGHT

from conftest import requires_data

TEST_HEIGHT = 320
TEST_WIDTH = 280


def _synthetic_hip(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.0, 0.2, size=(TEST_HEIGHT, TEST_WIDTH)).astype(np.float32)
    # асимметричная яркая полоса — чтобы зеркалирование было проверяемо
    image[40:260, 60:90] = 0.8
    image[40:260, 60:65] = 0.95
    return image


# --------------------------------------------------------------------------- #
# Препроцессинг: детерминизм и зеркалирование
# --------------------------------------------------------------------------- #


def test_preprocess_is_deterministic():
    image = _synthetic_hip()
    first = preprocess_for_cnn(image, SIDE_LEFT)
    second = preprocess_for_cnn(image, SIDE_LEFT)
    np.testing.assert_array_equal(first, second)


def test_preprocess_mirrors_right_side_only():
    image = _synthetic_hip()
    left = preprocess_for_cnn(image, SIDE_LEFT)
    right = preprocess_for_cnn(image, SIDE_RIGHT)
    none_side = preprocess_for_cnn(image, None)

    # Правое бедро — горизонтальное зеркало левого (тот же изотропный кадр).
    np.testing.assert_allclose(right, np.fliplr(left), atol=1e-5)
    # Без указанной стороны кадр не отражается (как left).
    np.testing.assert_allclose(none_side, left, atol=1e-5)
    # Само отражение реально меняет пиксели (не тождественная операция).
    assert not np.allclose(left, right)


def test_preprocess_output_is_isotropic_shape():
    image = _synthetic_hip()
    iso = preprocess_for_cnn(image, SIDE_LEFT)
    expected_height = int(round(TEST_HEIGHT * cfg.ANISOTROPY_Y_OVER_X))
    assert iso.shape == (expected_height, TEST_WIDTH)
    assert iso.dtype == np.float32


def test_to_uint8_png_array_range():
    image = _synthetic_hip()
    iso = preprocess_for_cnn(image, SIDE_LEFT)
    png = to_uint8_png_array(iso)
    assert png.dtype == np.uint8
    assert png.min() >= 0 and png.max() <= 255


# --------------------------------------------------------------------------- #
# labels.csv: нет UID/тегов DICOM
# --------------------------------------------------------------------------- #

_FORBIDDEN_SUBSTRINGS = ("uid", "sop", "instance", "patient", "tag", "path", "file_name")


def test_labels_csv_has_no_identifying_columns():
    for column in LABELS_COLUMNS:
        lowered = column.lower()
        for forbidden in _FORBIDDEN_SUBSTRINGS:
            assert forbidden not in lowered, f"{column} похоже на идентифицирующее поле"


def test_anonymize_study_folders_is_stable_and_injective():
    studies = pd.Series(
        [
            "2.25.000000000000000000000000000000000001",
            "1.2.840.113619.2.9",
            "2.25.000000000000000000000000000000000001",
            None,
        ]
    )
    mapping = anonymize_study_folders(studies)
    assert set(mapping) == {
        "2.25.000000000000000000000000000000000001",
        "1.2.840.113619.2.9",
    }
    # id-ы уникальны и имеют устойчивый читаемый формат s###...
    assert len(set(mapping.values())) == len(mapping)
    assert all(v.startswith("s") for v in mapping.values())

    # тот же вход -> тот же результат (детерминизм сортировки по исходному UID)
    again = anonymize_study_folders(studies)
    assert mapping == again


def test_build_labels_frame_replaces_study_folder_with_anon_id():
    frame = pd.DataFrame(
        {
            "dedup_group_id": ["g0000", "g0001"],
            "study_folder": ["study_a", "study_b"],
            "split": ["train", "val"],
            "software_version": ["18.41.005", "18.50.082"],
            "hip_side": ["left", "right"],
            "label_hip_positioning": [0, 1],
            "label_hip_roi": [0, 0],
            "has_mask_rect": [True, np.nan],
        }
    )
    study_map = {"study_a": "s000", "study_b": "s001"}
    labels = build_labels_frame(frame, study_map)
    assert list(labels.columns) == list(LABELS_COLUMNS)
    assert list(labels["study_folder"]) == ["s000", "s001"]
    assert "study_a" not in labels.to_csv()
    assert "study_b" not in labels.to_csv()
    assert labels["has_mask_rect"].tolist() == [True, False]
    assert labels["label_hip_positioning"].dtype == np.int64 or labels["label_hip_positioning"].dtype == int


# --------------------------------------------------------------------------- #
# Локальный рантайм без весов: честная деградация
# --------------------------------------------------------------------------- #


def test_loader_returns_none_without_weights(tmp_path):
    missing = tmp_path / "hip_positioning_cnn.pt"
    assert not missing.exists()
    predictor = load_hip_positioning_predictor(missing)
    assert predictor is None


# --------------------------------------------------------------------------- #
# Экспорт на реальных данных (требует выгрузку)
# --------------------------------------------------------------------------- #


@requires_data
def test_export_frame_has_one_row_per_unique_labeled_hip_image():
    frame = hip_export_frame()
    assert len(frame) == 150  # stages/stage_4.md: 151 уникальное изображение бедра, has_target=True у 150
    assert frame["dedup_group_id"].is_unique


@requires_data
def test_export_dataset_writes_zip_with_one_png_per_row(tmp_path):
    from columba.hip_export import export_dataset

    zip_path = tmp_path / "hip_positioning_dataset.zip"
    study_map_path = tmp_path / "study_map.csv"
    export_dataset(zip_path=zip_path, study_map_path=study_map_path, verbose=False)

    assert zip_path.exists()
    assert study_map_path.exists()

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        assert cfg.HIP_EXPORT_LABELS_CSV_NAME in names
        png_names = [n for n in names if n.endswith(".png")]
        labels_csv = archive.read(cfg.HIP_EXPORT_LABELS_CSV_NAME).decode("utf-8")

        for forbidden in ("2.25.", "1.2.840.113619"):
            assert forbidden not in labels_csv

    labels = pd.read_csv(study_map_path)  # sanity: study_map читается отдельно
    assert set(labels.columns) == {"study_folder", "anon_id"}
    assert len(png_names) == 150
