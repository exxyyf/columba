"""Шаг 7: сплит без утечек.

Группировка — по исследованию (пациенту): позвоночник и оба бедра одного
человека всегда едут в один фолд. Стратификация — итеративная (Sechidis et al.)
по меткам вида «регион:тип нарушения», чтобы редкие классы (в первую очередь
«Некорректная область интереса», 3-4 позитива на сторону) не уехали целиком
в один фолд. Seed зафиксирован, результат сохраняется файлом.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import OUTPUT_LABEL_KEYS, SEED, SPLIT_GROUP_COLUMN, VAL_FRACTION
from .targets import LABEL_PREFIX

TRAIN = "train"
VAL = "val"
SPLITS = (TRAIN, VAL)


def build_split(
    targets: pd.DataFrame,
    *,
    seed: int = SEED,
    val_fraction: float = VAL_FRACTION,
    group_column: str = SPLIT_GROUP_COLUMN,
) -> dict:
    """Сформировать сплит. Возвращает словарь, готовый к записи в split.json."""
    usable = targets[targets["has_target"].fillna(False)].copy()
    excluded = targets[~targets["has_target"].fillna(False)]

    group_labels = _group_label_sets(usable, group_column)
    groups = sorted(group_labels)
    assignment = _iterative_stratification(
        groups, group_labels, {TRAIN: 1.0 - val_fraction, VAL: val_fraction}, seed=seed
    )

    usable["split"] = usable[group_column].map(assignment)
    payload = {
        "seed": seed,
        "val_fraction": val_fraction,
        "group_column": group_column,
        # Фактические токены стратификации из _group_label_sets: присутствие
        # региона, ключи строк словаря и «чистый регион» как прокси отсутствия
        # нарушений (quality_class напрямую токеном не является).
        "stratified_by": [
            "<region>:present",
            *OUTPUT_LABEL_KEYS,
            "<region>:clean",
        ],
        "groups": {
            name: sorted(g for g, s in assignment.items() if s == name) for name in SPLITS
        },
        "dedup_groups": {
            name: sorted(usable.loc[usable["split"] == name, "dedup_group_id"]) for name in SPLITS
        },
        "excluded_dedup_groups": {
            "reason": "зона не размечена в таблице «Калибровка» — таргета нет",
            "ids": sorted(excluded["dedup_group_id"].dropna()),
        },
        "stats": _stats(usable),
    }
    return payload


def _group_label_sets(usable: pd.DataFrame, group_column: str) -> dict[str, list[str]]:
    """Токены стратификации для каждой группы (исследования)."""
    tokens: dict[str, list[str]] = defaultdict(list)
    for row in usable.itertuples():
        group = getattr(row, group_column)
        region = row.region
        tokens[group].append(f"{region}:present")
        any_violation = False
        for key in OUTPUT_LABEL_KEYS:
            value = getattr(row, f"{LABEL_PREFIX}{key}")
            if pd.notna(value) and int(value) == 1:
                tokens[group].append(key)
                any_violation = True
        if not any_violation:
            tokens[group].append(f"{region}:clean")
    return {group: sorted(set(values)) for group, values in tokens.items()}


def _iterative_stratification(
    groups: list[str],
    group_labels: dict[str, list[str]],
    proportions: dict[str, float],
    *,
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    remaining = set(groups)

    desired_total = {name: proportions[name] * len(groups) for name in proportions}
    label_counts: dict[str, int] = defaultdict(int)
    for group in groups:
        for label in group_labels[group]:
            label_counts[label] += 1
    desired_label = {
        label: {name: proportions[name] * count for name in proportions}
        for label, count in label_counts.items()
    }

    assignment: dict[str, str] = {}
    while remaining:
        pending = {
            label: sum(1 for g in remaining if label in group_labels[g]) for label in label_counts
        }
        pending = {label: n for label, n in pending.items() if n > 0}
        if pending:
            # Самая редкая среди ещё не распределённых меток — первой.
            rarest = min(sorted(pending), key=lambda label: (pending[label], label))
            candidates = sorted(g for g in remaining if rarest in group_labels[g])
        else:
            rarest = None
            candidates = sorted(remaining)

        rng.shuffle(candidates)
        for group in candidates:
            if rarest is not None:
                target = _argmax_split(desired_label[rarest], desired_total, rng)
            else:
                target = _argmax_split(desired_total, desired_total, rng)
            assignment[group] = target
            remaining.discard(group)
            desired_total[target] -= 1
            for label in group_labels[group]:
                desired_label[label][target] -= 1
    return assignment


def _argmax_split(primary: dict[str, float], secondary: dict[str, float], rng: random.Random) -> str:
    best = max(primary.values())
    tied = sorted(name for name, value in primary.items() if value == best)
    if len(tied) == 1:
        return tied[0]
    best2 = max(secondary[name] for name in tied)
    tied2 = [name for name in tied if secondary[name] == best2]
    return tied2[0] if len(tied2) == 1 else rng.choice(tied2)


def _stats(usable: pd.DataFrame) -> dict:
    stats: dict = {}
    for name in SPLITS:
        part = usable[usable["split"] == name]
        entry = {
            "studies": int(part["study_folder"].nunique()),
            "dedup_groups": int(len(part)),
            "files": int(part["n_files"].sum()),
            "by_region": {str(k): int(v) for k, v in part["region"].value_counts().items()},
            "quality_class_positive": int((part["quality_class"] == 1).sum()),
        }
        for key in OUTPUT_LABEL_KEYS:
            entry[key] = int((part[f"{LABEL_PREFIX}{key}"] == 1).sum())
        stats[name] = entry
    return stats


def save_split(payload: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_split(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
