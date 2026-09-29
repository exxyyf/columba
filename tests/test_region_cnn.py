"""Этап 1: CNN региона/стороны, арбитраж с эвристикой, критерий готовности.

Критерий готовности этапа: безошибочное определение ПОП/бедро на всём трейне —
для обеих веток (эвристика и CNN) и для арбитража. Плюс: латеральность на
именованных файлах «Для теста», кадры 248 px, независимость от тегов и имён
файлов, устойчивость к маскированию, дубликаты обсчитываются один раз.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from columba import config as cfg
from columba.dicom_io import read_dicom
from columba.inference import describe_inputs
from columba.regions import SIDE_LEFT, SIDE_RIGHT

from conftest import requires_data

requires_weights = pytest.mark.skipif(
    not cfg.REGION_CNN_WEIGHTS.exists(),
    reason="нет весов CNN (uv run python -m columba.train_region_cnn)",
)


# --------------------------------------------------------------------------- #
# default_device: этап 7 — GPU в Docker-контейнере, если проброшена
# --------------------------------------------------------------------------- #


def test_default_device_prefers_cuda_over_mps_and_cpu(monkeypatch):
    import torch

    from columba.region_cnn import default_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_device().type == "cuda"


def test_default_device_falls_back_to_mps_without_cuda(monkeypatch):
    import torch

    from columba.region_cnn import default_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_device().type == "mps"


def test_default_device_falls_back_to_cpu_without_gpu(monkeypatch):
    import torch

    from columba.region_cnn import default_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert default_device().type == "cpu"


@pytest.fixture(scope="session")
def predictor():
    if not cfg.REGION_CNN_WEIGHTS.exists():
        pytest.skip("нет весов CNN")
    from columba.region_cnn import load_region_predictor

    return load_region_predictor()


@pytest.fixture(scope="session")
def cnn_by_hash(manifest, predictor):
    """CNN-предсказания для всех уникальных изображений выгрузки."""
    cache = {}
    readable = manifest[manifest["read_status"] == "Success"]
    for row in readable[readable["is_group_representative"].fillna(False)].itertuples():
        result = read_dicom(row.abs_path)
        cache[row.pixel_hash] = predictor.predict_pixels(result.pixels, result.tags)
    return cache


# --------------------------------------------------------------------------- #
# Критерий готовности: 100% на всём трейне (обе ветки + арбитраж)
# --------------------------------------------------------------------------- #


@requires_data
def test_heuristic_region_is_correct_on_every_train_file(manifest):
    """Ветка «эвристика»: регион на всех 499 файлах (248 px -> бедро фолбэком)."""
    readable = manifest[manifest["read_status"] == "Success"]
    assert len(readable) == cfg.EXPECTED_TRAIN_FILES
    resolved = readable["region"].replace(cfg.REGION_UNKNOWN, cfg.REGION_SUBMISSION_FALLBACK)
    # Истина: зона из таблицы «Калибровка» (spine / hip_*), где она размечена.
    annotated = readable[readable["zone_key"].notna()]
    expected = annotated["zone_key"].map(cfg.ZONE_TO_REGION)
    assert (resolved.loc[annotated.index] == expected).all()


@requires_data
@requires_weights
def test_cnn_region_matches_on_all_unique_images_and_all_files(manifest, cnn_by_hash):
    """Ветка «CNN»: регион на всех 252 уникальных и, через хэш, на всех 499 файлах."""
    readable = manifest[manifest["read_status"] == "Success"]
    assert readable["pixel_hash"].isin(cnn_by_hash).all()  # дубликаты покрыты
    assert len(cnn_by_hash) == cfg.EXPECTED_DEDUP_GROUPS
    for row in readable.itertuples():
        expected = row.region if row.region != cfg.REGION_UNKNOWN else cfg.REGION_HIP
        assert cnn_by_hash[row.pixel_hash].region == expected, row.relative_path


@requires_data
@requires_weights
def test_cnn_side_matches_heuristic_on_all_hips(manifest, cnn_by_hash):
    hips = manifest[
        (manifest["region"] == cfg.REGION_HIP) & manifest["is_group_representative"].fillna(False)
    ]
    for row in hips.itertuples():
        if pd.isna(row.hip_side):
            continue
        assert cnn_by_hash[row.pixel_hash].side == row.hip_side, row.relative_path


@requires_data
@requires_weights
def test_arbitration_has_no_disagreements_on_train(manifest, cnn_by_hash):
    """Арбитраж: итог верен и ни одного расхождения веток на всём трейне."""
    from columba.region_cnn import arbitrate_region

    readable = manifest[manifest["read_status"] == "Success"]
    for row in readable.itertuples():
        side = row.hip_side if pd.notna(row.hip_side) else None
        verdict = arbitrate_region(int(row.cols), row.region, side, cnn_by_hash[row.pixel_hash])
        expected = row.region if row.region != cfg.REGION_UNKNOWN else cfg.REGION_HIP
        assert verdict.region == expected, row.relative_path
        assert not verdict.disagreement, row.relative_path


# --------------------------------------------------------------------------- #
# Кадры 248 px и латеральность на файлах «Для теста»
# --------------------------------------------------------------------------- #


@requires_data
@requires_weights
def test_248px_endoprosthesis_frames_are_hips_in_both_branches(manifest, cnn_by_hash):
    """248 px — протокол эндопротеза: бедро и по CNN, и по таблице ширин.

    Стороны сверены глазами на этапе 1: CR000001 — левое, CR000002 — правое.
    """
    odd = manifest[
        (manifest["region"] == cfg.REGION_UNKNOWN) & (manifest["read_status"] == "Success")
    ]
    assert len(odd) == 2
    assert set(odd["cols_from_pixels"]) == {248}
    assert all(int(w) in cfg.REGION_BY_STANDARD_WIDTH for w in odd["cols_from_pixels"])
    sides = {}
    for row in odd.itertuples():
        prediction = cnn_by_hash[row.pixel_hash]
        assert prediction.region == cfg.REGION_HIP
        sides[row.file_name] = prediction.side
    assert sides == {"CR000001.dcm": SIDE_LEFT, "CR000002.dcm": SIDE_RIGHT}


@requires_data
@requires_weights
def test_248px_frames_get_hip_side_final_from_cnn_backfill(manifest, tmp_path):
    """Сквозной инференс: у 248 px кадров `hip_side_final` заполнена из CNN.

    Эвристика для них сторону не считает (сырой регион `unknown`), поэтому
    без backfill сторона терялась бы — а она нужна модулю бедра (этап 4).
    """
    import shutil

    odd = manifest[
        (manifest["region"] == cfg.REGION_UNKNOWN) & (manifest["read_status"] == "Success")
    ]
    for row in odd.itertuples():
        shutil.copy(row.abs_path, tmp_path / row.file_name)

    frame = describe_inputs(tmp_path)  # AUTO: веса есть, CNN-ветка активна
    assert set(frame["region_final"]) == {cfg.REGION_HIP}
    assert set(frame["region_method"]) == {cfg.REGION_METHOD_HEURISTIC}
    assert not frame["region_disagreement"].any()
    sides = dict(zip(frame["file_name"], frame["hip_side_final"]))
    assert sides == {"CR000001.dcm": SIDE_LEFT, "CR000002.dcm": SIDE_RIGHT}
    # Сырая эвристическая сторона при этом честно пуста.
    assert frame["hip_side"].isna().all()


@requires_data
@requires_weights
def test_laterality_on_named_test_files(predictor):
    """Именованные организаторами файлы «Для теста»: ПОП / ППОБ / ЛПОБ."""
    expected = {
        "CR000000_ПОП.dcm": (cfg.REGION_SPINE, None),
        "CR000000_ППОБ.dcm": (cfg.REGION_HIP, SIDE_RIGHT),
        "CR000001_ЛПОБ.dcm": (cfg.REGION_HIP, SIDE_LEFT),
    }
    for name, (region, side) in expected.items():
        result = read_dicom(cfg.TEST_DIR / name)
        prediction = predictor.predict_pixels(result.pixels, result.tags)
        assert prediction.region == region, name
        assert prediction.side == side, name


# --------------------------------------------------------------------------- #
# Классификатор не читает теги и имена файлов
# --------------------------------------------------------------------------- #


@requires_data
@requires_weights
def test_cnn_branch_survives_stripped_tags_and_renamed_files(tmp_path):
    """region_final не меняется после удаления тегов и переименования файлов."""
    import pydicom

    expected = {}
    for index, (name, answer) in enumerate(
        sorted(
            {
                "CR000000_ПОП.dcm": cfg.REGION_SPINE,
                "CR000000_ППОБ.dcm": cfg.REGION_HIP,
                "CR000001_ЛПОБ.dcm": cfg.REGION_HIP,
            }.items()
        )
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = pydicom.dcmread(cfg.TEST_DIR / name)
        for tag in (
            "BodyPartExamined", "ViewPosition", "Laterality", "SeriesDescription",
            "StudyDescription", "SoftwareVersions", "PatientOrientation",
        ):
            if tag in dataset:
                delattr(dataset, tag)
        neutral = f"{index:06d}.dcm"
        expected[neutral] = answer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset.save_as(tmp_path / neutral, enforce_file_format=False)

    frame = describe_inputs(tmp_path)  # AUTO: CNN-ветка активна, веса есть
    assert set(frame["region_method"]) == {cfg.REGION_METHOD_HEURISTIC}
    for row in frame.itertuples():
        assert row.region_final == expected[row.file_name], row.file_name
        assert not row.region_disagreement, row.file_name


# --------------------------------------------------------------------------- #
# Маскирование: чёрные прямоугольники не должны менять ответ
# --------------------------------------------------------------------------- #


@requires_data
@requires_weights
def test_cnn_ignores_masking_rectangles(predictor):
    """Синтетическое маскирование краёв не меняет ни регион, ни сторону."""
    result = read_dicom(cfg.TEST_DIR / "CR000000_ППОБ.dcm")
    baseline = predictor.predict_pixels(result.pixels, result.tags)

    masked = np.array(result.pixels).copy()
    height, width = masked.shape[:2]
    masked[: height // 4, : width // 5] = 0  # верхний левый угол
    masked[-height // 5 :, -width // 4 :] = 0  # нижний правый угол
    prediction = predictor.predict_pixels(masked, result.tags)
    assert prediction.label == baseline.label


# --------------------------------------------------------------------------- #
# Правило арбитража (юнит, без данных и весов)
# --------------------------------------------------------------------------- #


def _cnn(label: str):
    from columba.region_cnn import CnnPrediction

    probs = {name: 0.05 for name in cfg.CNN_CLASSES}
    probs[label] = 0.9
    return CnnPrediction(label=label, probs=probs)


def test_arbitration_standard_width_prefers_heuristic_and_flags_disagreement():
    from columba.region_cnn import arbitrate_region

    verdict = arbitrate_region(300, cfg.REGION_SPINE, None, _cnn(cfg.CNN_CLASS_HIP_LEFT))
    assert verdict.region == cfg.REGION_SPINE
    assert verdict.method == cfg.REGION_METHOD_HEURISTIC
    assert verdict.disagreement

    agreed = arbitrate_region(280, cfg.REGION_HIP, SIDE_LEFT, _cnn(cfg.CNN_CLASS_HIP_LEFT))
    assert agreed.region == cfg.REGION_HIP and agreed.side == SIDE_LEFT
    assert not agreed.disagreement

    side_conflict = arbitrate_region(280, cfg.REGION_HIP, SIDE_LEFT, _cnn(cfg.CNN_CLASS_HIP_RIGHT))
    assert side_conflict.side == SIDE_LEFT  # эвристика первична
    assert side_conflict.disagreement


def test_arbitration_backfills_side_from_cnn_when_heuristic_has_none():
    """248 px: эвристика стороны не дала — сторона берётся из CNN без флага.

    Backfill работает только в вакуум: заданную эвристическую сторону CNN
    не перебивает (это конфликт веток, он флагуется отдельным тестом выше).
    """
    from columba.region_cnn import arbitrate_region

    verdict = arbitrate_region(248, cfg.REGION_UNKNOWN, None, _cnn(cfg.CNN_CLASS_HIP_LEFT))
    assert verdict.region == cfg.REGION_HIP
    assert verdict.side == SIDE_LEFT
    assert verdict.method == cfg.REGION_METHOD_HEURISTIC
    assert not verdict.disagreement

    # Без CNN заполнять нечем: сторона честно None.
    empty = arbitrate_region(248, cfg.REGION_UNKNOWN, None, None)
    assert empty.region == cfg.REGION_HIP
    assert empty.side is None
    assert not empty.disagreement


def test_arbitration_nonstandard_width_prefers_cnn():
    from columba.region_cnn import arbitrate_region

    verdict = arbitrate_region(500, cfg.REGION_UNKNOWN, None, _cnn(cfg.CNN_CLASS_SPINE))
    assert verdict.region == cfg.REGION_SPINE
    assert verdict.method == cfg.REGION_METHOD_CNN
    assert not verdict.disagreement


def test_arbitration_without_cnn_falls_back_to_heuristic():
    from columba.region_cnn import arbitrate_region

    verdict = arbitrate_region(500, cfg.REGION_UNKNOWN, None, None)
    assert verdict.region == cfg.REGION_SUBMISSION_FALLBACK
    assert verdict.method == cfg.REGION_METHOD_HEURISTIC

    spine = arbitrate_region(300, cfg.REGION_SPINE, None, None)
    assert spine.region == cfg.REGION_SPINE
    assert not spine.disagreement


def test_flip_swaps_hip_side_and_keeps_spine():
    """Горизонтальный флип в аугментациях обязан переключать метку стороны."""
    from columba.region_cnn import FLIPPED_CLASS

    assert FLIPPED_CLASS[cfg.CNN_CLASS_HIP_LEFT] == cfg.CNN_CLASS_HIP_RIGHT
    assert FLIPPED_CLASS[cfg.CNN_CLASS_HIP_RIGHT] == cfg.CNN_CLASS_HIP_LEFT
    assert FLIPPED_CLASS[cfg.CNN_CLASS_SPINE] == cfg.CNN_CLASS_SPINE
    # Инволюция: двойной флип возвращает исходную метку.
    assert all(FLIPPED_CLASS[FLIPPED_CLASS[c]] == c for c in cfg.CNN_CLASSES)


# --------------------------------------------------------------------------- #
# Дубликаты обсчитываются CNN один раз
# --------------------------------------------------------------------------- #


@requires_data
def test_duplicates_hit_the_cnn_once(tmp_path):
    """Три попиксельных копии — одно CNN-предсказание, три строки в манифесте."""
    import shutil

    class CountingStub:
        def __init__(self):
            self.calls = 0

        def predict_pixels(self, pixels, tags=None):
            from columba.region_cnn import CnnPrediction

            self.calls += 1
            return CnnPrediction(
                label=cfg.CNN_CLASS_HIP_RIGHT,
                probs={name: 1.0 if name == cfg.CNN_CLASS_HIP_RIGHT else 0.0 for name in cfg.CNN_CLASSES},
            )

    stub = CountingStub()
    source = cfg.TEST_DIR / "CR000000_ППОБ.dcm"
    for index in range(3):
        shutil.copy(source, tmp_path / f"copy_{index}.dcm")

    frame = describe_inputs(tmp_path, region_predictor=stub)
    assert len(frame) == 3
    assert stub.calls == 1
    assert set(frame["region_final"]) == {cfg.REGION_HIP}


# --------------------------------------------------------------------------- #
# Изотропный ресемплинг — из констант config
# --------------------------------------------------------------------------- #


def test_isotropic_resampling_uses_config_anisotropy():
    from columba.region_cnn import resample_to_isotropic

    image = np.random.default_rng(0).random((100, 60)).astype("float32")
    resampled = resample_to_isotropic(image)
    assert resampled.shape[1] == 60  # ширина не меняется
    assert resampled.shape[0] == round(100 * cfg.ANISOTROPY_Y_OVER_X)
