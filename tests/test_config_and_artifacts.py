"""Константы из одного места и состав артефактов этапа."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from columba import config as cfg

from conftest import requires_data

SRC_DIR = Path(cfg.__file__).resolve().parent
SPACING_VALUES = (cfg.PIXEL_SPACING_MM_Y, cfg.PIXEL_SPACING_MM_X)
# Числовой литерал в исходнике, не срезанный по краям соседним словом/точкой
# (иначе "18.41.005" или "framealpha=0.65" ловились бы частями/случайно).
NUMBER_RE = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")


def test_pixel_spacing_values():
    assert cfg.PIXEL_SPACING_MM == (1.05, 0.60)
    assert cfg.PIXEL_SPACING_MM_Y > cfg.PIXEL_SPACING_MM_X
    assert cfg.ANISOTROPY_Y_OVER_X == pytest.approx(1.75)


def _spacing_literals_in(text: str) -> list[str]:
    """Числовые литералы `text`, равные (по значению) pixel spacing."""
    return [
        match.group()
        for match in NUMBER_RE.finditer(text)
        if any(float(match.group()) == pytest.approx(v, abs=1e-9) for v in SPACING_VALUES)
    ]


def test_pixel_spacing_defined_in_one_place():
    """Ни один модуль, кроме config.py, не зашивает pixel spacing числом.

    Сравнение по ЗНАЧЕНИЮ (не по строке): литерал вида "0.60" или "1.050" —
    та же величина, что и "0.6"/"1.05", и тоже обязана ловиться.
    """
    offenders = []
    for path in SRC_DIR.rglob("*.py"):
        if path.name == "config.py":
            continue
        text = path.read_text(encoding="utf-8")
        offenders.extend(f"{path.name}: {literal}" for literal in _spacing_literals_in(text))
    assert offenders == [], f"pixel spacing продублирован: {offenders}"


def test_spacing_literal_detector_catches_trailing_zero_forms():
    """Регресс: раньше "0.60"/"1.050" не ловились (сравнение строк, не чисел)."""
    assert _spacing_literals_in("PIXEL_Y = 1.050") == ["1.050"]
    assert _spacing_literals_in("PIXEL_X = 0.60") == ["0.60"]
    assert _spacing_literals_in("PIXEL_X = 0.600") == ["0.600"]


def test_spacing_literal_detector_has_no_false_positives():
    """Не должен ловить похожие, но другие числа: версии софта, framealpha и т. п."""
    assert _spacing_literals_in('SOFTWARE_VERSION = "18.41.005"') == []
    assert _spacing_literals_in("framealpha=0.65") == []
    assert _spacing_literals_in("VAL_FRACTION = 0.2") == []


def test_criteria_schema_is_complete():
    assert len(cfg.CRITERIA) == 7
    assert {c.zone for c in cfg.CRITERIA} == set(cfg.ZONES)
    assert len(cfg.CRITERIA_BY_ZONE[cfg.ZONE_SPINE]) == 3
    assert len(cfg.CRITERIA_BY_ZONE[cfg.ZONE_HIP_RIGHT]) == 2
    assert len(cfg.CRITERIA_BY_ZONE[cfg.ZONE_HIP_LEFT]) == 2
    assert {c.output_label for c in cfg.CRITERIA} == set(cfg.OUTPUT_LABEL_KEYS)
    assert all(
        cfg.OUTPUT_LABELS_BY_KEY[c.output_label].region == cfg.ZONE_TO_REGION[c.zone]
        for c in cfg.CRITERIA
    )
    # Колонки «Итог» перечислены явно и не пересекаются с критериями.
    assert not {c.column for c in cfg.CRITERIA} & set(cfg.MARKUP_IGNORED_COLUMNS)


@requires_data
def test_stage_artifacts_are_written(stage0):
    artifacts = stage0["artifacts_dir"]
    for name in (
        cfg.MANIFEST_PARQUET,
        cfg.MANIFEST_CSV,
        cfg.TARGETS_CSV,
        cfg.SPLIT_JSON,
        cfg.ANOMALY_LOG,
        cfg.SUBMISSION_REFERENCE_CSV,
    ):
        assert (artifacts / name.name).exists(), name.name

    manifest = pd.read_parquet(artifacts / cfg.MANIFEST_PARQUET.name)
    assert len(manifest) == cfg.EXPECTED_TRAIN_FILES
    for column in ("relative_path", "study_folder", "pixel_hash", "dedup_group_id", "region", "read_status"):
        assert column in manifest.columns

    payload = json.loads((artifacts / cfg.SPLIT_JSON.name).read_text(encoding="utf-8"))
    assert payload["seed"] == cfg.SEED
    assert payload["group_column"] == cfg.SPLIT_GROUP_COLUMN


@requires_data
def test_anomaly_log_records_file_count_reconciliation(stage0):
    text = (stage0["artifacts_dir"] / cfg.ANOMALY_LOG.name).read_text(encoding="utf-8")
    assert "502" in text and "499" in text
    assert "## Нечитаемые файлы (статус Failure) — 0" in text
