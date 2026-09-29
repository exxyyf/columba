"""Шаги 1-3: инвентаризация, чтение, дедупликация."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from columba import config as cfg
from columba.dedup import pixel_hash
from columba.dicom_io import STATUS_FAILURE, STATUS_SUCCESS, normalize, read_dicom
from columba.inventory import count_all_dicom_under, load_manifest

from conftest import requires_data


@requires_data
def test_file_counts_reconciled():
    """502 против 499: разница — ровно тестовый набор."""
    assert count_all_dicom_under(cfg.STUDIES_DIR) == cfg.EXPECTED_TRAIN_FILES
    assert count_all_dicom_under(cfg.TEST_DIR) == cfg.EXPECTED_TEST_FILES
    assert count_all_dicom_under(cfg.DATA_DIR) == cfg.EXPECTED_TOTAL_DICOM


@requires_data
def test_manifest_has_row_per_file(manifest):
    assert len(manifest) == cfg.EXPECTED_TRAIN_FILES
    assert manifest["file_id"].is_unique
    assert manifest["relative_path"].is_unique
    assert manifest["study_folder"].nunique() == cfg.EXPECTED_STUDIES
    assert set(manifest["read_status"]) <= {STATUS_SUCCESS, STATUS_FAILURE}


@requires_data
def test_dedup_group_count(manifest):
    assert manifest["dedup_group_id"].nunique() == cfg.EXPECTED_DEDUP_GROUPS


@requires_data
def test_dedup_groups_are_pixel_identical(manifest):
    """Каждая группа дедупа — попиксельные копии, а не просто совпавший хэш."""
    multi = manifest[manifest["dedup_group_size"] > 1]
    assert len(multi) > 0
    for _, chunk in multi.groupby("dedup_group_id"):
        reference = read_dicom(chunk.iloc[0]["abs_path"]).pixels
        for path in chunk["abs_path"].iloc[1:]:
            other = read_dicom(path).pixels
            assert np.array_equal(reference, other)


@requires_data
def test_dedup_does_not_drop_files(manifest):
    """Дедупликация — атрибут, а не удаление: все файлы на месте."""
    readable = manifest[manifest["read_status"] == STATUS_SUCCESS]
    assert int(readable["dedup_group_size"].groupby(readable["dedup_group_id"]).first().sum()) == len(readable)
    assert int(manifest["is_group_representative"].sum()) == manifest["dedup_group_id"].nunique()


@requires_data
def test_every_dedup_group_lives_in_one_study(manifest):
    per_group = manifest.groupby("dedup_group_id")["study_folder"].nunique()
    assert per_group.max() == 1


def test_unreadable_file_gets_failure_status(tmp_path):
    """Битый файл не роняет ингест, а получает статус Failure."""
    broken = tmp_path / "broken.dcm"
    broken.write_bytes(b"not a dicom at all")
    result = read_dicom(broken)
    assert result.status == STATUS_FAILURE
    assert result.pixels is None
    assert result.error


def test_pixel_hash_is_content_based():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    b = a.copy()
    c = a.copy()
    c[0, 0] += 1
    assert pixel_hash(a) == pixel_hash(b)
    assert pixel_hash(a) != pixel_hash(c)
    assert pixel_hash(a) != pixel_hash(a.reshape(4, 3))


def test_load_manifest_recomputes_foreign_abs_path(tmp_path):
    """Манифест с `abs_path` чужой машины (этап 9, п. 9.1): загрузчик обязан
    пересчитать `abs_path` из `relative_path` + `studies_dir` ЭТОЙ машины,
    а не доверять тому, что записано в файле."""
    studies_dir = tmp_path / "studies"
    (studies_dir / "study1").mkdir(parents=True)
    (studies_dir / "study1" / "file.dcm").write_bytes(b"stub")

    manifest = pd.DataFrame(
        {
            "dedup_group_id": ["g0001"],
            "relative_path": ["study1/file.dcm"],
            "abs_path": [
                "/Users/exxyyf/Documents/projects/columba/data/НД_для_обучения/"
                "Исследования/study1/file.dcm"
            ],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    loaded = load_manifest(manifest_path, studies_dir=studies_dir)
    expected = str(studies_dir / "study1" / "file.dcm")
    assert loaded.loc[0, "abs_path"] == expected
    assert (studies_dir / "study1" / "file.dcm").exists()  # пересчитанный путь реально существует


def test_load_manifest_defaults_are_late_bound_to_config(tmp_path, monkeypatch):
    """Аргументы по умолчанию читают `cfg.MANIFEST_CSV`/`cfg.STUDIES_DIR` в
    момент ВЫЗОВА, а не при импорте модуля — иначе monkeypatch в тестах не
    действовал бы (тот же приём, что уже используют читатели манифеста)."""
    studies_dir = tmp_path / "studies"
    (studies_dir / "s1").mkdir(parents=True)
    (studies_dir / "s1" / "a.dcm").write_bytes(b"x")
    manifest = pd.DataFrame({"relative_path": ["s1/a.dcm"], "abs_path": ["/foreign/a.dcm"]})
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    monkeypatch.setattr(cfg, "MANIFEST_CSV", manifest_path)
    monkeypatch.setattr(cfg, "STUDIES_DIR", studies_dir)

    loaded = load_manifest()
    assert loaded.loc[0, "abs_path"] == str(studies_dir / "s1" / "a.dcm")


def test_load_manifest_reads_parquet_by_suffix(tmp_path):
    studies_dir = tmp_path / "studies"
    (studies_dir / "s1").mkdir(parents=True)
    (studies_dir / "s1" / "a.dcm").write_bytes(b"x")
    manifest = pd.DataFrame({"relative_path": ["s1/a.dcm"], "abs_path": ["/foreign/a.dcm"]})
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    loaded = load_manifest(manifest_path, studies_dir=studies_dir)
    assert loaded.loc[0, "abs_path"] == str(studies_dir / "s1" / "a.dcm")


def test_load_manifest_without_relative_path_column_passes_through(tmp_path):
    """Манифест без `relative_path` (например урезанная синтетика в тестах
    calibrate.py) не должен падать — просто нечего пересчитывать."""
    manifest = pd.DataFrame({"dedup_group_id": ["g0001"], "markup_comment": ["x"]})
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    loaded = load_manifest(manifest_path, studies_dir=tmp_path)
    assert "abs_path" not in loaded.columns
    assert list(loaded["dedup_group_id"]) == ["g0001"]


def test_normalize_uses_bits_stored():
    pixels = np.array([[0, 128, 255]], dtype=np.uint8)
    out = normalize(pixels, {"BitsStored": 8, "PhotometricInterpretation": "MONOCHROME2"})
    assert out.dtype == np.float32
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    inverted = normalize(pixels, {"BitsStored": 8, "PhotometricInterpretation": "MONOCHROME1"})
    assert inverted[0, 0] == pytest.approx(1.0)
