"""Шаг 6: перегенерация таргетов из 7 бинарных критериев.

Правила, зафиксированные в плане этапа:

* столбец «Итог» полностью игнорируется — он воспроизводится заново;
* quality_class = OR по критериям СВОЕЙ зоны (хотя бы одно нарушение -> 1);
* пропуск в критерии = зоны нет у пациента: такая ячейка не участвует в OR
  и НЕ превращается в ноль. Если вся зона не размечена — таргета нет;
* параллельно строится мультилейбл-вектор по 5 строкам выходного словаря
  (пары «регион + тип нарушения»; уникальных строк — 4).
"""

from __future__ import annotations

import pandas as pd

from .config import (
    CRITERIA,
    CRITERIA_BY_ZONE,
    OUTPUT_LABEL_KEYS,
    OUTPUT_LABELS_BY_KEY,
    OUTPUT_LABELS_BY_REGION,
    VIOLATION_TYPE_SEPARATOR,
    ZONE_SPINE,
    ZONE_TO_REGION,
)

LABEL_PREFIX = "label_"
CRITERION_KEYS = tuple(c.key for c in CRITERIA)
LABEL_COLUMNS = tuple(f"{LABEL_PREFIX}{key}" for key in OUTPUT_LABEL_KEYS)


def violation_string(active_label_keys) -> str:
    """Собрать поле `violation_type` из ключей активных строк словаря.

    Порядок внутри поля не важен, но детерминирован; повторяющиеся строки
    («Некорректная укладка» есть и у позвоночника, и у бедра) не дублируются.
    """
    seen: list[str] = []
    for key in OUTPUT_LABEL_KEYS:
        if key in set(active_label_keys):
            text = OUTPUT_LABELS_BY_KEY[key].violation_type
            if text not in seen:
                seen.append(text)
    return VIOLATION_TYPE_SEPARATOR.join(seen)


def build_targets(manifest: pd.DataFrame, markup: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Собрать таблицу таргетов уровня уникального изображения (dedup-группы).

    Возвращает (targets, anomalies).
    """
    anomalies: list[dict] = []
    markup_by_study = markup.set_index("study_folder", drop=False)

    readable = manifest[manifest["dedup_group_id"].notna()]
    rows = []
    for group_id, chunk in readable.groupby("dedup_group_id", sort=True):
        studies = sorted(set(chunk["study_folder"]))
        spans_studies = len(studies) > 1
        if spans_studies:
            anomalies.append(
                {
                    "kind": "dedup_group_spans_studies",
                    "dedup_group_id": group_id,
                    "detail": ", ".join(studies),
                }
            )
        study = studies[0]
        zone = chunk["zone_key"].dropna().unique()
        zone_key = zone[0] if len(zone) == 1 else None
        if len(zone) > 1:
            anomalies.append(
                {
                    "kind": "dedup_group_ambiguous_zone",
                    "dedup_group_id": group_id,
                    "detail": ", ".join(map(str, zone)),
                }
            )
        if spans_studies:
            # Метки принадлежат конкретному исследованию (markup match — по
            # study_folder); группа, растянутая на несколько исследований, не
            # может честно унаследовать таргет studies[0] — это был бы таргет
            # "случайного" исследования. Уходит в "без таргета" тем же путём,
            # что и dedup_group_ambiguous_zone (zone_key = None ниже).
            zone_key = None

        row = {
            "dedup_group_id": group_id,
            "study_folder": study,
            "region": chunk["region"].iloc[0],
            "hip_side": chunk["hip_side"].dropna().iloc[0] if chunk["hip_side"].notna().any() else pd.NA,
            "zone_key": zone_key if zone_key is not None else pd.NA,
            "n_files": int(len(chunk)),
            "rows_px": int(chunk["rows"].iloc[0]) if pd.notna(chunk["rows"].iloc[0]) else pd.NA,
            "cols_px": int(chunk["cols"].iloc[0]) if pd.notna(chunk["cols"].iloc[0]) else pd.NA,
        }

        markup_row = markup_by_study.loc[study] if study in markup_by_study.index else None
        row.update(_zone_targets(zone_key, markup_row, group_id, study, anomalies))
        rows.append(row)

    targets = pd.DataFrame(rows)
    ordered = [
        "dedup_group_id",
        "study_folder",
        "region",
        "hip_side",
        "zone_key",
        "n_files",
        "rows_px",
        "cols_px",
        "zone_annotated",
        "has_target",
        "quality_class",
        *CRITERION_KEYS,
        *LABEL_COLUMNS,
        "violation_type",
    ]
    targets = targets.reindex(columns=[c for c in ordered if c in targets.columns])
    for column in (*CRITERION_KEYS, *LABEL_COLUMNS, "quality_class"):
        if column in targets:
            targets[column] = targets[column].astype("Int64")
    return targets.sort_values("dedup_group_id", ignore_index=True), anomalies


def _zone_targets(zone_key, markup_row, group_id: str, study: str, anomalies: list[dict]) -> dict:
    """Критерии, quality_class и мультилейбл для одной зоны."""
    result: dict = {key: pd.NA for key in CRITERION_KEYS}
    result.update({column: pd.NA for column in LABEL_COLUMNS})
    result["zone_annotated"] = False
    result["has_target"] = False
    result["quality_class"] = pd.NA
    result["violation_type"] = pd.NA

    if zone_key is None or markup_row is None:
        return result

    zone_criteria = CRITERIA_BY_ZONE[zone_key]
    values = {c.key: markup_row[c.key] for c in zone_criteria}
    present = {k: int(v) for k, v in values.items() if pd.notna(v)}
    result.update(present)

    if not present:
        # Зона не размечена: таргета нет. Нули сюда не подставляем.
        return result

    if len(present) != len(zone_criteria):
        anomalies.append(
            {
                "kind": "partially_annotated_zone",
                "dedup_group_id": group_id,
                "detail": f"{study} / {zone_key}: заполнены {sorted(present)} из {[c.key for c in zone_criteria]}",
            }
        )

    result["zone_annotated"] = True
    result["has_target"] = True
    result["quality_class"] = 1 if any(v == 1 for v in present.values()) else 0

    # Мультилейбл по 5 строкам выходного словаря. Строки чужого региона —
    # честные нули для этого снимка, а не подстановка вместо пропуска.
    labels = {column: 0 for column in LABEL_COLUMNS}
    active: list[str] = []
    for criterion in zone_criteria:
        if present.get(criterion.key) == 1:
            labels[f"{LABEL_PREFIX}{criterion.output_label}"] = 1
            active.append(criterion.output_label)
    result.update(labels)
    result["violation_type"] = violation_string(active)

    region = ZONE_TO_REGION[zone_key]
    allowed = {label.key for label in OUTPUT_LABELS_BY_REGION[region]}
    assert set(active) <= allowed, f"{zone_key}: тип нарушения вне словаря региона {region}"
    return result


def audit_against_legacy_totals(targets: pd.DataFrame, markup: pd.DataFrame) -> list[dict]:
    """Сравнить перегенерированный quality_class со столбцом «Итог».

    Результат идёт только в лог: «Итог» на таргеты не влияет.
    """
    legacy_column = {
        ZONE_SPINE: "legacy_total_spine",
        "hip_right": "legacy_total_hip_right",
        "hip_left": "legacy_total_hip_left",
    }
    markup_by_study = markup.set_index("study_folder")
    findings: list[dict] = []
    for row in targets.itertuples():
        if not row.has_target or pd.isna(row.zone_key):
            continue
        study = row.study_folder
        if study not in markup_by_study.index:
            continue
        legacy = markup_by_study.loc[study, legacy_column[row.zone_key]]
        if pd.isna(legacy):
            continue
        if int(legacy) != int(row.quality_class):
            findings.append(
                {
                    "kind": "legacy_total_mismatch",
                    "dedup_group_id": row.dedup_group_id,
                    "detail": (
                        f"{study} / {row.zone_key}: «Итог»={int(legacy)}, "
                        f"перегенерированный quality_class={int(row.quality_class)}, "
                        f"комментарий: {markup_by_study.loc[study, 'comment'] or '—'}"
                    ),
                }
            )
    return findings


def check_no_violation_without_criteria(targets: pd.DataFrame) -> pd.DataFrame:
    """Строки, где quality_class = 1 при нулевых критериях (их быть не должно)."""
    annotated = targets[targets["has_target"].fillna(False)]
    positive = annotated[annotated["quality_class"] == 1]
    criteria_sum = positive[list(CRITERION_KEYS)].fillna(0).sum(axis=1)
    return positive[criteria_sum == 0]
