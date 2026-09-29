"""Шаг 7: сплит без утечек."""

from __future__ import annotations

import pytest

from columba import config as cfg
from columba.split import build_split
from columba.targets import LABEL_PREFIX

from conftest import requires_data


@requires_data
def test_train_and_val_do_not_intersect(split):
    train = set(split["dedup_groups"]["train"])
    val = set(split["dedup_groups"]["val"])
    assert train and val
    assert train & val == set()
    assert set(split["groups"]["train"]) & set(split["groups"]["val"]) == set()


@requires_data
def test_split_is_grouped_by_study(split, targets):
    """Снимки одного исследования не расползаются между фолдами."""
    assignment = {}
    for name in ("train", "val"):
        for group in split["groups"][name]:
            assignment[group] = name
    usable = targets[targets["has_target"]]
    by_study = usable.groupby("study_folder")["dedup_group_id"].apply(list)
    for study, group_ids in by_study.items():
        folds = {
            "train" if gid in set(split["dedup_groups"]["train"]) else "val"
            for gid in group_ids
        }
        assert len(folds) == 1, f"{study} разъехалось между фолдами"
        assert folds == {assignment[study]}


@requires_data
def test_every_annotated_image_is_assigned(split, targets):
    usable = set(targets.loc[targets["has_target"], "dedup_group_id"])
    assigned = set(split["dedup_groups"]["train"]) | set(split["dedup_groups"]["val"])
    assert assigned == usable
    assert set(split["excluded_dedup_groups"]["ids"]) == set(
        targets.loc[~targets["has_target"], "dedup_group_id"]
    )


@requires_data
def test_rare_labels_present_in_both_folds(split):
    """Редкие классы (в первую очередь «Некорректная область интереса»)."""
    for key in cfg.OUTPUT_LABEL_KEYS:
        assert split["stats"]["train"][key] > 0, key
        assert split["stats"]["val"][key] > 0, key


@requires_data
def test_regions_present_in_both_folds(split):
    for name in ("train", "val"):
        by_region = split["stats"][name]["by_region"]
        assert by_region.get(cfg.REGION_SPINE, 0) > 0
        assert by_region.get(cfg.REGION_HIP, 0) > 0


@requires_data
def test_val_fraction_is_close_to_target(split):
    train = len(split["groups"]["train"])
    val = len(split["groups"]["val"])
    share = val / (train + val)
    assert abs(share - cfg.VAL_FRACTION) < 0.05


@requires_data
def test_split_is_reproducible(targets, split):
    again = build_split(targets)
    assert again["groups"] == split["groups"]
    assert again["dedup_groups"] == split["dedup_groups"]


@requires_data
def test_split_changes_with_seed(targets, split):
    other = build_split(targets, seed=cfg.SEED + 1)
    assert other["groups"] != split["groups"]
