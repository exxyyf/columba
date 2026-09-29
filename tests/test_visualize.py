"""Этап 8, бонус 1: `visualize.py` — визуализация нарушений на реальных кадрах.

Никакой новой логики детекции здесь нет (только рисование уже посчитанных
ориентиров/чекеров) — тесты проверяют, что рендер не падает, возвращает
непустой PNG и что информация о регионе/чекерах в `info` соответствует
реальному содержимому кадра (по `data/Для теста`, как остальные тесты
интеграции).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from columba import config as cfg
from columba.dicom_io import normalize, read_dicom
from columba.spine_keypoints import STATUS_NOT_EVALUATED, STATUS_OK
from columba.visualize import figure_to_png_bytes, render_dicom_file, render_violation_overlay

from conftest import requires_data

SPINE_FILE = cfg.TEST_DIR / "CR000000_ПОП.dcm"
HIP_FILES = (cfg.TEST_DIR / "CR000000_ППОБ.dcm", cfg.TEST_DIR / "CR000001_ЛПОБ.dcm")


def _pixels(path: Path):
    result = read_dicom(path)
    assert result.ok
    return normalize(result.pixels, result.tags)


@requires_data
def test_render_spine_detects_region_and_checkers():
    fig, info = render_violation_overlay(_pixels(SPINE_FILE))
    assert info["region"] == cfg.REGION_SPINE
    assert set(info["checkers"]) == {"spine_axis", "spine_positioning", "spine_objects"}
    for result in info["checkers"].values():
        assert result.status in (STATUS_OK, STATUS_NOT_EVALUATED)
    png = figure_to_png_bytes(fig)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # сигнатура PNG
    assert len(png) > 1000


@requires_data
@pytest.mark.parametrize("hip_file", HIP_FILES)
def test_render_hip_detects_region_and_checkers(hip_file):
    fig, info = render_violation_overlay(_pixels(hip_file))
    assert info["region"] == cfg.REGION_HIP
    assert info["side"] in ("left", "right", None)
    assert set(info["checkers"]) == {"hip_positioning", "hip_roi"}
    png = figure_to_png_bytes(fig)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


@requires_data
def test_render_hip_with_explicit_cnn_predictor_flags_positioning():
    from columba.hip_cnn import load_hip_positioning_predictor

    predictor = load_hip_positioning_predictor()
    if predictor is None:
        pytest.skip("веса hip_positioning_cnn.pt не скачаны в этом окружении")
    fig, info = render_violation_overlay(_pixels(HIP_FILES[0]), hip_cnn_predictor=predictor)
    assert info["checkers"]["hip_positioning"].status == STATUS_OK


@requires_data
def test_render_dicom_file_matches_render_violation_overlay():
    fig, info = render_dicom_file(SPINE_FILE)
    assert fig is not None
    assert info["region"] == cfg.REGION_SPINE


def test_render_dicom_file_degrades_on_unreadable_file(tmp_path):
    bad = tmp_path / "not_a_dicom.dcm"
    bad.write_bytes(b"not a real dicom file")
    fig, info = render_dicom_file(bad)
    assert fig is None
    assert info is None
