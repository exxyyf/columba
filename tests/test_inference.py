"""Прогон по выборке, устроенной как закрытый тест.

Организаторы подтвердили: те же аппараты и типы исследований, другие пациенты,
никаких дополнительных тегов, меток зоны или суффиксов региона в именах.
Эти тесты проверяют, что пайплайн на такой вход опирается только на пиксели.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pydicom
import pytest

from columba import config as cfg
from columba.inference import describe_inputs, run_inference
from columba.regions import classify_region, classify_region_from_pixels
from columba.submission import validate_submission
from columba.targets import violation_string

from conftest import requires_data

# Имена, которые организаторы дали эталонным файлам. В самом пайплайне они
# не используются — только здесь, как ожидаемый ответ.
EXPECTED = {
    "CR000000_ПОП.dcm": (cfg.REGION_SPINE, None),
    "CR000000_ППОБ.dcm": (cfg.REGION_HIP, "right"),
    "CR000001_ЛПОБ.dcm": (cfg.REGION_HIP, "left"),
}


def test_region_comes_from_the_array_not_a_tag():
    import numpy as np

    spine = np.zeros((317, 300), dtype=np.uint8)
    hip = np.zeros((291, 280), dtype=np.uint8)
    odd = np.zeros((401, 248), dtype=np.uint8)
    assert classify_region_from_pixels(spine) == cfg.REGION_SPINE
    assert classify_region_from_pixels(hip) == cfg.REGION_HIP
    assert classify_region_from_pixels(odd) == cfg.REGION_UNKNOWN
    # Высота кадра не участвует: она гуляет от 180 до 405 px.
    assert classify_region_from_pixels(np.zeros((180, 300), dtype=np.uint8)) == cfg.REGION_SPINE
    assert classify_region_from_pixels(np.zeros((405, 300), dtype=np.uint8)) == cfg.REGION_SPINE


@requires_data
def test_tag_columns_agrees_with_array_width(manifest):
    readable = manifest[manifest["read_status"] == "Success"]
    assert (readable["cols"] == readable["cols_from_pixels"]).all()


@requires_data
def test_flat_directory_is_processed_like_the_closed_test():
    """Плоский каталог без папок исследований — как придёт закрытый тест."""
    frame = describe_inputs(cfg.TEST_DIR)
    assert len(frame) == cfg.EXPECTED_TEST_FILES
    for row in frame.itertuples():
        expected_region, expected_side = EXPECTED[row.file_name]
        assert row.region == expected_region, row.file_name
        if expected_side is not None:
            assert row.hip_side == expected_side, row.file_name


@requires_data
def test_region_and_side_survive_stripped_tags_and_renamed_files(tmp_path):
    """Ни теги зоны, ни имя файла в закрытом тесте не помогут — и не нужны."""
    renamed = {}
    for index, (name, expected) in enumerate(sorted(EXPECTED.items())):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = pydicom.dcmread(cfg.TEST_DIR / name)
        for tag in ("BodyPartExamined", "ViewPosition", "Laterality", "SeriesDescription",
                    "StudyDescription", "SoftwareVersions", "PatientOrientation"):
            if tag in dataset:
                delattr(dataset, tag)
        neutral = f"{index:06d}.dcm"
        renamed[neutral] = expected
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset.save_as(tmp_path / neutral, enforce_file_format=False)

    frame = describe_inputs(tmp_path)
    assert len(frame) == len(renamed)
    for row in frame.itertuples():
        expected_region, expected_side = renamed[row.file_name]
        assert row.region == expected_region, row.file_name
        if expected_side is not None:
            assert row.hip_side == expected_side, row.file_name


@requires_data
def test_inference_produces_a_valid_submission():
    submission = run_inference(cfg.TEST_DIR)
    validate_submission(submission, expected_rows=cfg.EXPECTED_TEST_FILES)
    assert set(submission["anatomical_region"]) == {
        cfg.REGION_OUTPUT_NAMES[cfg.REGION_SPINE],
        cfg.REGION_OUTPUT_NAMES[cfg.REGION_HIP],
    }


@requires_data
def test_inference_accepts_a_predictor_and_keeps_the_format():
    """Заглушка предсказателя: формат и полярность держатся и с нулями, и с единицами."""

    def predictor(manifest: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in manifest.itertuples():
            if row.region == cfg.REGION_HIP:
                rows.append(
                    {
                        "file_id": row.file_id,
                        "quality_class": cfg.QUALITY_CLASS_VIOLATION,
                        "quality_prob": 0.87,
                        "violation_type": violation_string(["hip_positioning", "hip_roi"]),
                    }
                )
            else:
                rows.append(
                    {
                        "file_id": row.file_id,
                        "quality_class": cfg.QUALITY_CLASS_OK,
                        "quality_prob": 0.03,
                        "violation_type": "",
                    }
                )
        return pd.DataFrame(rows)

    submission = run_inference(cfg.TEST_DIR, predictor)
    validate_submission(submission, expected_rows=cfg.EXPECTED_TEST_FILES)
    hips = submission[submission["anatomical_region"] == cfg.REGION_OUTPUT_NAMES[cfg.REGION_HIP]]
    assert set(hips["quality_class"]) == {1}
    assert set(hips["violation_type"]) == {
        f"{cfg.VIOLATION_POSITIONING}{cfg.VIOLATION_TYPE_SEPARATOR}{cfg.VIOLATION_ROI}"
    }


@requires_data
def test_duplicate_files_each_get_their_own_row(tmp_path):
    """Попиксельные дубликаты возможны и в закрытом тесте — строк должно быть столько же."""
    import shutil

    source = cfg.TEST_DIR / "CR000000_ППОБ.dcm"
    for index in range(3):
        shutil.copy(source, tmp_path / f"copy_{index}.dcm")
    submission = run_inference(tmp_path)
    assert len(submission) == 3
    assert submission["dedup_group_id"].nunique() == 1
    validate_submission(submission, expected_rows=3)


# --------------------------------------------------------------------------- #
# Задача 9.3: ключ строки сабмита и изоляция ошибок чекеров
# --------------------------------------------------------------------------- #


@requires_data
def test_file_name_equals_relative_path_with_posix_separators(tmp_path):
    """`file_name` — относительный путь от корня входа, а не голое имя файла.

    Голое имя файла коллидирует в закрытом тесте, где организаторы снимают
    суффиксы региона («CR000000_ПОП.dcm» и «CR000000_ППОБ.dcm» оба стали бы
    «CR000000.dcm»); относительный путь остаётся уникальным и в папочном, и
    в плоском входе.
    """
    import shutil

    study_dir = tmp_path / "study_001"
    study_dir.mkdir()
    shutil.copy(cfg.TEST_DIR / "CR000000_ПОП.dcm", study_dir / "a.dcm")

    frame = describe_inputs(tmp_path)
    assert len(frame) == 1
    row = frame.iloc[0]
    # file_name и relative_path согласованы — одна и та же строка.
    assert row["file_name"] == row["relative_path"]
    assert row["file_name"] == "study_001/a.dcm"
    assert "\\" not in row["file_name"]


@requires_data
def test_file_name_is_bare_name_for_a_flat_directory():
    """Плоский каталог (как придёт закрытый тест) — file_name без папок."""
    frame = describe_inputs(cfg.TEST_DIR)
    assert set(frame["file_name"]) == set(EXPECTED.keys())
    assert (frame["file_name"] == frame["relative_path"]).all()


@requires_data
def test_sop_instance_uid_is_read_as_an_extra_debug_column():
    """`sop_instance_uid` — дополнительная колонка (не в SUBMISSION_COLUMNS),
    читается из тега DICOM для каждого читаемого файла."""
    frame = describe_inputs(cfg.TEST_DIR)
    readable = frame[frame["read_status"] == "Success"]
    assert readable["sop_instance_uid"].notna().all()
    # Разные файлы — разные UID (это не константа-заглушка).
    assert readable["sop_instance_uid"].nunique() == len(readable)


@requires_data
def test_hip_checker_exception_on_one_file_does_not_fail_the_batch(monkeypatch):
    """Исключение чекера бедра на одном файле не должно ронять весь батч.

    Батч из `cfg.TEST_DIR` содержит две разные картинки бедра (разные
    pixel_hash) — первый вызов `run_hip_checkers` кидает исключение
    (имитация вырожденного кейпоинта), второй отрабатывает нормально.
    Ожидается: строка на каждый файл (в т.ч. упавший), у упавшего —
    `status=not_evaluated` по обоим чекерам бедра, у второго — нормальный
    результат, спина не затронута вовсе.
    """
    import columba.hip_checkers as hip_checkers_module
    from columba.inference import submission_from_manifest

    original = hip_checkers_module.run_hip_checkers
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("вырожденный кадр (тест изоляции ошибок)")
        return original(*args, **kwargs)

    monkeypatch.setattr(hip_checkers_module, "run_hip_checkers", flaky)

    frame = describe_inputs(cfg.TEST_DIR)
    assert len(frame) == cfg.EXPECTED_TEST_FILES  # строка на каждый файл сохранилась
    assert calls["n"] == 2  # оба уникальных бедра действительно дошли до чекера

    hips = frame[frame["region_final"] == cfg.REGION_HIP]
    assert len(hips) == 2
    # `hip_positioning` может быть not_evaluated и без ошибки (например, нет
    # CNN-весов на машине, где запущен тест) — неразличимый сигнал сам по
    # себе. `hip_roi` же rule-based и не зависит от CNN, поэтому он надёжно
    # отличает упавший файл от нормально обработанного независимо от весов.
    roi_statuses = hips["hip_roi_status"].tolist()
    assert roi_statuses.count("not_evaluated") == 1  # ровно один файл упал
    assert "ok" in roi_statuses  # второй посчитан нормально, а не тоже упал

    failed = hips[hips["hip_roi_status"] == "not_evaluated"]
    assert len(failed) == 1
    assert "RuntimeError" in failed.iloc[0]["hip_signals"]

    # Сабмит по-прежнему валиден и полон, несмотря на упавший чекер.
    submission = submission_from_manifest(frame)
    validate_submission(submission, expected_rows=cfg.EXPECTED_TEST_FILES)


@requires_data
def test_spine_checker_exception_does_not_fail_the_batch(monkeypatch):
    """Тот же смоук, но для чекеров позвоночника (`_attach_spine_checkers`)."""
    import columba.spine_checkers as spine_checkers_module
    from columba.inference import submission_from_manifest

    def broken(*args, **kwargs):
        raise ValueError("однородный кадр (тест изоляции ошибок)")

    monkeypatch.setattr(spine_checkers_module, "run_spine_checkers", broken)

    frame = describe_inputs(cfg.TEST_DIR)
    assert len(frame) == cfg.EXPECTED_TEST_FILES

    spine_rows = frame[frame["region_final"] == cfg.REGION_SPINE]
    assert len(spine_rows) == 1
    assert spine_rows.iloc[0]["spine_axis_status"] == "not_evaluated"
    assert spine_rows.iloc[0]["spine_positioning_status"] == "not_evaluated"
    assert spine_rows.iloc[0]["spine_objects_status"] == "not_evaluated"
    assert "ValueError" in spine_rows.iloc[0]["spine_signals"]

    submission = submission_from_manifest(frame)
    validate_submission(submission, expected_rows=cfg.EXPECTED_TEST_FILES)
