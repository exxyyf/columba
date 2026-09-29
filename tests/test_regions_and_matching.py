"""Шаги 4-5: регион, латеральность, матчинг с «Калибровкой»."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from columba import config as cfg
from columba.dicom_io import read_dicom
from columba.regions import SIDE_LEFT, SIDE_RIGHT, classify_region, hip_side, resolve_sides_within_study

from conftest import requires_data

# Файлы тестового набора названы организаторами: ПОП — позвоночник,
# ППОБ / ЛПОБ — правый / левый проксимальный отдел бедра. Это единственная
# внешняя проверка эвристик региона и стороны, которая у нас есть.
REFERENCE_FILES = {
    "CR000000_ПОП.dcm": (cfg.REGION_SPINE, None),
    "CR000000_ППОБ.dcm": (cfg.REGION_HIP, SIDE_RIGHT),
    "CR000001_ЛПОБ.dcm": (cfg.REGION_HIP, SIDE_LEFT),
}


def test_classify_region_mapping():
    assert classify_region(300) == cfg.REGION_SPINE
    assert classify_region(280) == cfg.REGION_HIP
    assert classify_region(248) == cfg.REGION_UNKNOWN
    assert classify_region(None) == cfg.REGION_UNKNOWN


@requires_data
@pytest.mark.parametrize("file_name", sorted(REFERENCE_FILES))
def test_heuristics_on_named_reference_files(file_name):
    expected_region, expected_side = REFERENCE_FILES[file_name]
    result = read_dicom(cfg.TEST_DIR / file_name)
    assert result.ok
    region = classify_region(result.tags["Columns"])
    assert region == expected_region
    if expected_side is not None:
        assert hip_side(result.pixels).side == expected_side


def test_resolve_sides_within_study_uses_relative_order():
    assert resolve_sides_within_study([0.30, -0.20]) == [SIDE_RIGHT, SIDE_LEFT]
    # Оба значения одного знака — относительный порядок всё равно разводит их.
    assert resolve_sides_within_study([-0.05, -0.30]) == [SIDE_RIGHT, SIDE_LEFT]
    assert resolve_sides_within_study([float("nan")]) == [None]


@requires_data
def test_every_readable_image_has_region(manifest):
    readable = manifest[manifest["read_status"] == "Success"]
    assert readable["region"].notna().all()
    assert set(readable["region"]) <= {cfg.REGION_SPINE, cfg.REGION_HIP, cfg.REGION_UNKNOWN}
    # Ровно два кадра шириной 248 px (двустороннее эндопротезирование ТБС).
    assert int((readable["region"] == cfg.REGION_UNKNOWN).sum()) == 2


@requires_data
def test_hip_side_resolved_for_every_hip_image(manifest):
    hips = manifest[(manifest["region"] == cfg.REGION_HIP) & manifest["is_group_representative"]]
    assert hips["hip_side"].notna().all()
    assert hips["hip_side_confident"].all(), "нашёлся снимок со слабым сигналом латеральности"


@requires_data
def test_two_hip_images_of_a_study_get_different_sides(manifest):
    hips = manifest[(manifest["region"] == cfg.REGION_HIP) & manifest["is_group_representative"]]
    per_study = hips.groupby("study_folder")["hip_side"]
    for study, sides in per_study:
        values = list(sides)
        assert len(values) == len(set(values)), f"{study}: две проекции получили одну сторону"


@requires_data
def test_markup_matches_folders_one_to_one(manifest, markup):
    assert markup["study_folder"].is_unique
    assert set(markup["study_folder"]) == set(manifest["study_folder"])
    assert bool(manifest["matched_markup"].all())


@requires_data
def test_annotated_zones_agree_with_imaged_zones(manifest, markup):
    """Расхождения зон допускаются только как известные, залогированные случаи."""
    representatives = manifest[manifest["is_group_representative"].fillna(False)]
    lookup = markup.set_index("study_folder")
    mismatched = []
    for study, chunk in representatives.groupby("study_folder"):
        row = lookup.loc[study]
        annotated = {
            zone for zone in cfg.ZONES
            if any(pd.notna(row[c.key]) for c in cfg.CRITERIA_BY_ZONE[zone])
        }
        imaged = set(chunk["zone_key"].dropna())
        if annotated != imaged or chunk["zone_key"].isna().any():
            mismatched.append(study)
    assert sorted(mismatched) == [
        "2.25.11175860580562939441493697221497597640",  # правое бедро эндопротезировано
        "2.25.12798473087614376830819854616447908612",  # двустороннее эндопротезирование ТБС
    ]
