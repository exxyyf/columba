"""Этап 1, шаг 1: хвосты ревью этапа 0.

1. `read_dicom` дочитывает файлы без 128-байтного преамбула второй попыткой
   с force=True, но мусор по-прежнему уходит в Failure.
2. В плоском каталоге бедра разных пациентов не разводятся принудительно
   в пару право/лево — метод честно пишется как `content_only`.
3. `stratified_by` в split.json описывает фактические токены стратификации.
"""

from __future__ import annotations

import numpy as np
import pytest

from columba import config as cfg
from columba.dicom_io import STATUS_FAILURE, STATUS_SUCCESS, read_dicom
from columba.inference import (
    HIP_SIDE_METHOD_CONTENT_ONLY,
    HIP_SIDE_METHOD_PAIRED,
    describe_inputs,
)
from columba.split import build_split

from conftest import requires_data

REFERENCE_HIP = cfg.TEST_DIR / "CR000000_ППОБ.dcm"
REFERENCE_HIP_LEFT = cfg.TEST_DIR / "CR000001_ЛПОБ.dcm"

PREAMBLE_AND_MAGIC = 132  # 128 байт преамбула + b"DICM"


# --------------------------------------------------------------------------- #
# 1. read_dicom: force=True как вторая попытка
# --------------------------------------------------------------------------- #


@requires_data
def test_dicom_without_preamble_is_read_on_second_attempt(tmp_path):
    """Файл со срезанным преамбулом должен читаться (вторая попытка, force=True)."""
    raw = REFERENCE_HIP.read_bytes()
    assert raw[128:132] == b"DICM"
    stripped = tmp_path / "no_preamble.dcm"
    stripped.write_bytes(raw[PREAMBLE_AND_MAGIC:])

    reference = read_dicom(REFERENCE_HIP)
    result = read_dicom(stripped)
    assert result.status == STATUS_SUCCESS, result.error
    assert result.pixels is not None
    assert np.array_equal(result.pixels, reference.pixels)


@requires_data
def test_dicom_without_preamble_reads_without_pixels_too(tmp_path):
    raw = REFERENCE_HIP.read_bytes()
    stripped = tmp_path / "no_preamble.dcm"
    stripped.write_bytes(raw[PREAMBLE_AND_MAGIC:])

    result = read_dicom(stripped, load_pixels=False)
    assert result.status == STATUS_SUCCESS, result.error
    assert result.pixels is None
    assert result.tags.get("Rows")


def test_garbage_file_still_fails(tmp_path):
    """force=True не должен «читать» произвольные байты как DICOM."""
    garbage = tmp_path / "garbage.dcm"
    garbage.write_bytes(b"\x13\x37" * 4096)
    result = read_dicom(garbage)
    assert result.status == STATUS_FAILURE
    assert result.error
    assert result.pixels is None


def test_empty_file_still_fails(tmp_path):
    empty = tmp_path / "empty.dcm"
    empty.write_bytes(b"")
    assert read_dicom(empty).status == STATUS_FAILURE


# --------------------------------------------------------------------------- #
# 2. Плоский каталог: пары не строятся
# --------------------------------------------------------------------------- #


@requires_data
def test_flat_directory_hips_are_not_forcibly_paired(tmp_path):
    """Два бедра разных пациентов в плоском каталоге не разводятся в пару.

    Оба эталонных снимка кладутся в один плоский каталог. Раньше их
    принудительно спаривали как «правое+левое одного пациента»; теперь
    сторона берётся из абсолютного знака признака по каждому снимку,
    а метод честно называется content_only.
    """
    import shutil

    shutil.copy(REFERENCE_HIP, tmp_path / "a.dcm")
    shutil.copy(REFERENCE_HIP_LEFT, tmp_path / "b.dcm")

    frame = describe_inputs(tmp_path)
    hips = frame[frame["region"] == cfg.REGION_HIP]
    assert len(hips) == 2
    assert set(hips["hip_side_method"]) == {HIP_SIDE_METHOD_CONTENT_ONLY}
    # Абсолютный знак согласован с парным разведением на всей выгрузке
    # (решения этапа 0, п. 5), поэтому ответ остаётся правильным.
    sides = dict(zip(hips["file_name"], hips["hip_side"]))
    assert sides == {"a.dcm": "right", "b.dcm": "left"}


@requires_data
def test_flat_directory_two_same_side_hips_can_agree(tmp_path):
    """Два правых бедра (копии) в плоском каталоге оба остаются правыми."""
    import shutil

    shutil.copy(REFERENCE_HIP, tmp_path / "p1.dcm")
    raw = REFERENCE_HIP.read_bytes()
    (tmp_path / "p2.dcm").write_bytes(raw)  # попиксельный дубликат

    frame = describe_inputs(tmp_path)
    hips = frame[frame["region"] == cfg.REGION_HIP]
    assert set(hips["hip_side"]) == {"right"}
    assert set(hips["hip_side_method"]) == {HIP_SIDE_METHOD_CONTENT_ONLY}


@requires_data
def test_study_folders_still_use_paired_resolution(tmp_path):
    """Внутри папки исследования парное разведение сохраняется."""
    import shutil

    study = tmp_path / "study_x"
    study.mkdir()
    shutil.copy(REFERENCE_HIP, study / "a.dcm")
    shutil.copy(REFERENCE_HIP_LEFT, study / "b.dcm")

    frame = describe_inputs(tmp_path)
    hips = frame[frame["region"] == cfg.REGION_HIP]
    assert set(hips["hip_side_method"]) == {HIP_SIDE_METHOD_PAIRED}
    assert set(hips["hip_side"]) == {"right", "left"}


# --------------------------------------------------------------------------- #
# 3. stratified_by соответствует фактическим токенам
# --------------------------------------------------------------------------- #


@requires_data
def test_split_stratified_by_matches_actual_tokens(targets):
    payload = build_split(targets)
    declared = set(payload["stratified_by"])
    # Токены из _group_label_sets: "<region>:present", ключи меток, "<region>:clean".
    assert declared == {"<region>:present", "<region>:clean", *cfg.OUTPUT_LABEL_KEYS}
    assert "quality_class" not in declared
