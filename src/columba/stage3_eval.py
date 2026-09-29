"""Этап 3, шаг 8: оценка метрик агрегированного сабмита на реальной разметке.

Запуск: `uv run python -m columba.stage3_eval`.

Что делает:

* один раз считает манифест `describe_inputs(cfg.STUDIES_DIR)` (чекеры
  этапа 2 + арбитраж региона) и строит из него сабмит через
  `inference.submission_from_manifest` (агрегатор этапа 3 подключается по
  умолчанию, `predictor=AUTO`) — тот же путь, что и `run_inference`, только
  без повторного прогона чекеров ради собственного манифеста с `pixel_hash`;
* связывает строки сабмита с `artifacts/targets.csv` по `pixel_hash`, а не по
  `dedup_group_id`: манифест `describe_inputs` строит СВОИ dedup-группы
  (`assign_dedup_groups`, см. `dedup.py`) — идентификаторы могут не совпасть с
  `artifacts/manifest.parquet` (сборка этапа 0, откуда взят `targets.csv`).
  `pixel_hash` — SHA-256 от массива пикселей (`dedup.py:pixel_hash`),
  детерминирован и совпадает при одинаковом содержимом кадра независимо от
  способа сборки манифеста, поэтому связывание через него надёжно. В
  DEBUG_COLUMNS сабмита `pixel_hash` нет, но `submission_from_manifest` не
  меняет ни число, ни порядок строк манифеста (`build_submission` только
  выбирает/добавляет колонки) — значит, i-я строка сабмита соответствует i-й
  строке манифеста; `attach_pixel_hash` сверяет это по `relative_path`
  построчно (assert) и приклеивает `pixel_hash` по позиции, без повторного
  прогона и без join по `relative_path` как единственной гарантии (это
  дополнительная подстраховка, а не замена сверки). Дальше `pixel_hash ->
  dedup_group_id` берётся из `artifacts/manifest.parquet`, и уже по этому
  `dedup_group_id` строки сабмита соединяются с `targets.csv`;
* метрики считаются на уникальных размеченных изображениях (`has_target`),
  отдельно train/val/all: бинарная F1 и ROC-AUC по `quality_prob`, F1 по
  каждому из 4 типов словаря (`config.VIOLATION_TYPES`) и их Macro-F1,
  95% ДИ кластерным бутстрэпом по `study_folder` (переиспользуется
  `spine_eval.cluster_bootstrap_ci`, seed `config.SEED`, число итераций —
  `config.STAGE3_EVAL_BOOTSTRAP_N` в `main()`, явный параметр в тестах);
* F1 типа, для которого в срезе нет ни одного истинного позитива, не
  определена — возвращается NaN (не 0 и не 1), тем же соглашением, что и
  `spine_eval._f1_stat`, ДАЖЕ если есть ложные срабатывания (predicted
  positives > 0): F1 = 2PR/(P+R) при recall не определён (0/0), поэтому
  функция явно отдаёт NaN, а не 0/1 — в отличие, например, от sklearn с
  `zero_division=0`, который в этом случае вернул бы 0 и оштрафовал бы
  Macro-F1. Отчёт по каждому типу дополнительно печатает число предсказанных
  позитивов и явную пометку, когда F1=NaN при pred>0 (модель ложно
  срабатывает на типе без разметки в срезе — не 0 в Macro-F1, но и не скрыто).
  Macro-F1 — среднее по типам с определённым F1 (NaN-типы не входят) — так
  Macro-F1 не проваливается искусственно из-за структурно редких типов
  (например, «Некорректная область интереса»);
* sanity: та же функция метрик на `artifacts/submission_reference.csv`
  (сабмит из истинных меток) против `targets.csv` обязана дать 1.0 по всем
  метрикам — тест на корректность функции метрик, не пайплайна;
* пишет `artifacts/stage3_report.md` + `.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .inference import describe_inputs, submission_from_manifest
from .inventory import load_manifest
from .spine_eval import cluster_bootstrap_ci, flag_metrics, ranking_auc

# Пути артефактов этапа: в config нет готовых констант под них (в отличие от
# `SPINE_CHECKER_REPORT`), поэтому объявлены здесь как модульные константы.
STAGE3_REPORT_MD = cfg.ARTIFACTS_DIR / "stage3_report.md"
STAGE3_REPORT_JSON = STAGE3_REPORT_MD.with_suffix(".json")

SPLITS: tuple[str, ...] = ("train", "val")


# --------------------------------------------------------------------------- #
# Загрузка артефактов
# --------------------------------------------------------------------------- #


def load_targets(path: Path | str = cfg.TARGETS_CSV) -> pd.DataFrame:
    """`targets.csv` с пустым (не NaN) `violation_type` при отсутствии нарушений.

    `pandas.read_csv` по умолчанию читает пустое поле как NaN — в `targets.py`
    пустая строка `violation_type` пишется явно при `quality_class == 0`,
    значит после чтения её нужно вернуть на место.
    """
    frame = pd.read_csv(path)
    frame["violation_type"] = frame["violation_type"].fillna("")
    return frame


def _read_submission_csv(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["violation_type"] = frame["violation_type"].fillna("")
    return frame


# --------------------------------------------------------------------------- #
# Связывание сабмита с targets.csv
# --------------------------------------------------------------------------- #

LINKED_COLUMNS: tuple[str, ...] = (
    "dedup_group_id",
    "study_folder",
    "split",
    "region",
    "quality_class_true",
    "quality_class_pred",
    "quality_prob_pred",
    "violation_type_true",
    "violation_type_pred",
)


def _pred_frame(submission: pd.DataFrame) -> pd.DataFrame:
    return submission.rename(
        columns={
            "quality_class": "quality_class_pred",
            "quality_prob": "quality_prob_pred",
            "violation_type": "violation_type_pred",
        }
    )


def _true_frame(targets: pd.DataFrame) -> pd.DataFrame:
    return targets.rename(columns={"quality_class": "quality_class_true", "violation_type": "violation_type_true"})


def _finalize_linked(merged: pd.DataFrame) -> pd.DataFrame:
    """Оставить только размеченные уникальные изображения, привести колонки."""
    merged = merged[merged["has_target"].fillna(False)].copy()
    merged["violation_type_true"] = merged["violation_type_true"].fillna("")
    merged["violation_type_pred"] = merged["violation_type_pred"].fillna("")
    merged = merged.drop_duplicates("dedup_group_id")
    return merged.reindex(columns=list(LINKED_COLUMNS)).reset_index(drop=True)


def link_via_dedup_group_id(submission: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Связать сабмит с targets.csv напрямую по `dedup_group_id`.

    Годится только когда `dedup_group_id` сабмита УЖЕ те же, что в targets.csv
    — так у `artifacts/submission_reference.csv` (построен из того же манифеста
    этапа 0, что и targets.csv). Для сабмита `run_inference` на произвольном
    каталоге это не гарантировано — там `link_via_pixel_hash`.
    """
    pred = _pred_frame(submission)[["dedup_group_id", "quality_class_pred", "quality_prob_pred", "violation_type_pred"]]
    pred = pred.drop_duplicates("dedup_group_id")
    merged = _true_frame(targets).merge(pred, on="dedup_group_id", how="inner")
    return _finalize_linked(merged)


def attach_pixel_hash(submission: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Приклеить `pixel_hash` к сабмиту, построенному из ЭТОГО манифеста.

    `submission_from_manifest` строит сабмит из `manifest` без изменения числа
    и порядка строк (`build_submission` только выбирает/добавляет колонки),
    значит i-я строка сабмита соответствует i-й строке манифеста. В
    `DEBUG_COLUMNS` сабмита `pixel_hash` нет, но есть `relative_path` — им
    построчно сверяется соответствие (assert) перед тем, как приклеить
    `pixel_hash` по позиции, без повторного прогона `describe_inputs` и без
    merge по `relative_path` как единственной гарантии совпадения.
    """
    submission = submission.reset_index(drop=True)
    manifest = manifest.reset_index(drop=True)
    if len(submission) != len(manifest):
        raise ValueError(
            f"сабмит ({len(submission)} строк) и манифест ({len(manifest)} строк) не совпадают по длине — "
            "не тот манифест или submission_from_manifest изменил число строк"
        )
    mismatched = submission["relative_path"].values != manifest["relative_path"].values
    if mismatched.any():
        raise ValueError(
            f"сабмит и манифест разошлись по relative_path на {int(mismatched.sum())} строках — не тот манифест"
        )
    submission = submission.copy()
    submission["pixel_hash"] = manifest["pixel_hash"].values
    return submission


def link_via_pixel_hash(
    submission_with_pixel_hash: pd.DataFrame,
    manifest_reference: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Связать сабмит (с колонкой `pixel_hash`, см. `attach_pixel_hash`) с targets.csv.

    `manifest_reference` — `artifacts/manifest.parquet` (манифест этапа 0, тот
    же источник, что и `dedup_group_id` в targets.csv).
    """
    pred = _pred_frame(submission_with_pixel_hash)
    hash_to_group = (
        manifest_reference.dropna(subset=["pixel_hash"])
        .drop_duplicates("pixel_hash")
        .set_index("pixel_hash")["dedup_group_id"]
    )
    pred["dedup_group_id"] = pred["pixel_hash"].map(hash_to_group)
    pred = pred.dropna(subset=["dedup_group_id"])
    pred = pred.drop_duplicates("dedup_group_id")[
        ["dedup_group_id", "quality_class_pred", "quality_prob_pred", "violation_type_pred"]
    ]
    merged = _true_frame(targets).merge(pred, on="dedup_group_id", how="inner")
    return _finalize_linked(merged)


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #


def _has_type(text, vtype: str) -> bool:
    if text is None:
        return False
    if isinstance(text, float) and text != text:  # NaN
        return False
    return vtype in str(text).split(cfg.VIOLATION_TYPE_SEPARATOR)


def _with_type_columns(linked: pd.DataFrame) -> pd.DataFrame:
    frame = linked.copy()
    for vtype in cfg.VIOLATION_TYPES:
        frame[f"__true__{vtype}"] = frame["violation_type_true"].apply(lambda t, v=vtype: _has_type(t, v))
        frame[f"__pred__{vtype}"] = frame["violation_type_pred"].apply(lambda t, v=vtype: _has_type(t, v))
    return frame


def _binary_f1_stat(df: pd.DataFrame) -> float:
    labels = df["quality_class_true"].values.astype(int)
    if not labels.sum():
        return float("nan")
    return flag_metrics(df["quality_class_pred"].values.astype(bool).astype(int), labels)["f1"]


def _binary_auc_stat(df: pd.DataFrame) -> float:
    return ranking_auc(df["quality_prob_pred"].values.astype(float), df["quality_class_true"].values.astype(int))


def _type_f1_stat(vtype: str):
    true_col, pred_col = f"__true__{vtype}", f"__pred__{vtype}"

    def statistic(df: pd.DataFrame) -> float:
        true = df[true_col].values.astype(bool)
        if not true.sum():
            return float("nan")
        pred = df[pred_col].values.astype(bool)
        return flag_metrics(pred, true)["f1"]

    return statistic


def macro_f1(values) -> float:
    """Среднее по типам с определённым F1 (NaN-типы — без позитивов — не входят)."""
    valid = [v for v in values if v == v]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def _macro_f1_stat(df: pd.DataFrame) -> float:
    return macro_f1([_type_f1_stat(vtype)(df) for vtype in cfg.VIOLATION_TYPES])


def _estimate(frame: pd.DataFrame, statistic, n: int) -> dict:
    if len(frame) == 0:
        return {"value": float("nan"), "ci95": (float("nan"), float("nan"))}
    point = statistic(frame)
    ci = cluster_bootstrap_ci(frame, statistic, n)
    return {"value": round(point, 3) if point == point else float("nan"), "ci95": ci}


def compute_metrics(linked: pd.DataFrame, bootstrap_n: int) -> dict:
    """Метрики на срезе уникальных размеченных изображений `linked`.

    `linked` — результат `link_via_*` (колонки `LINKED_COLUMNS`).
    """
    frame = _with_type_columns(linked)
    result: dict = {
        "n": len(frame),
        "positives": int(frame["quality_class_true"].sum()) if len(frame) else 0,
        "binary_f1": _estimate(frame, _binary_f1_stat, bootstrap_n),
        "binary_auc": _estimate(frame, _binary_auc_stat, bootstrap_n),
        "by_type": {},
    }
    for vtype in cfg.VIOLATION_TYPES:
        estimate = _estimate(frame, _type_f1_stat(vtype), bootstrap_n)
        predicted_positives = int(frame[f"__pred__{vtype}"].sum()) if len(frame) else 0
        estimate["predicted_positives"] = predicted_positives
        estimate["note"] = (
            "F1 не определена (нет истинных позитивов в срезе), но есть "
            f"{predicted_positives} предсказанных — ложные срабатывания не в Macro-F1"
            if (estimate["value"] != estimate["value"] and predicted_positives > 0)
            else ""
        )
        result["by_type"][vtype] = estimate
    result["macro_f1"] = _estimate(frame, _macro_f1_stat, bootstrap_n)
    return result


def region_binary_breakdown(linked: pd.DataFrame, bootstrap_n: int) -> dict:
    """Бинарная F1/AUC отдельно по регионам — справочно (доп. п.8 плана)."""
    result = {}
    for region in (cfg.REGION_SPINE, cfg.REGION_HIP):
        subset = linked[linked["region"] == region]
        result[region] = {
            "n": len(subset),
            "positives": int(subset["quality_class_true"].sum()) if len(subset) else 0,
            "binary_f1": _estimate(subset, _binary_f1_stat, bootstrap_n),
            "binary_auc": _estimate(subset, _binary_auc_stat, bootstrap_n),
        }
    return result


def evaluate(linked: pd.DataFrame, bootstrap_n: int) -> dict:
    """Метрики train/val/all + разбивка по регионам."""
    report = {}
    slices = {"train": linked[linked["split"] == "train"], "val": linked[linked["split"] == "val"], "all": linked}
    for name, subset in slices.items():
        entry = compute_metrics(subset, bootstrap_n)
        entry["by_region"] = region_binary_breakdown(subset, bootstrap_n)
        report[name] = entry
    return report


# --------------------------------------------------------------------------- #
# Sanity
# --------------------------------------------------------------------------- #


def sanity_check(bootstrap_n: int, *, targets: pd.DataFrame | None = None) -> dict:
    """Метрики на `submission_reference.csv` против targets.csv — ожидание 1.0."""
    targets = targets if targets is not None else load_targets()
    submission_reference = _read_submission_csv(cfg.SUBMISSION_REFERENCE_CSV)
    linked = link_via_dedup_group_id(submission_reference, targets)
    return compute_metrics(linked, bootstrap_n)


# --------------------------------------------------------------------------- #
# CLI / отчёт
# --------------------------------------------------------------------------- #


def _fmt(estimate: dict) -> str:
    value, (low, high) = estimate["value"], estimate["ci95"]
    if value != value:
        return "NaN"
    if low != low:
        return f"{value} [ДИ н/о]"
    return f"{value} [{low}; {high}]"


def _write_report(report: dict, path: Path = STAGE3_REPORT_MD) -> None:
    lines = [
        "# Отчёт метрик этапа 3 (шаг 8)",
        "",
        "Генерируется `uv run python -m columba.stage3_eval` (машиночитаемая копия —",
        "`.json` рядом). Метрики — на уникальных размеченных изображениях",
        f"(`has_target`), 95% ДИ — кластерный бутстрэп по исследованиям",
        f"({cfg.STAGE3_EVAL_BOOTSTRAP_N} реплик, seed {cfg.SEED}). F1 типа без истинных",
        "позитивов в срезе — NaN (не 0 и не 1); Macro-F1 — среднее по типам с",
        "определённым F1.",
        "",
    ]
    for split_name in (*SPLITS, "all"):
        entry = report[split_name]
        lines += [
            f"## {split_name} (n={entry['n']}, позитивов {entry['positives']})",
            "",
            f"- Бинарная F1: {_fmt(entry['binary_f1'])}",
            f"- ROC-AUC: {_fmt(entry['binary_auc'])}",
            f"- Macro-F1 (4 типа): {_fmt(entry['macro_f1'])}",
            "",
            "| Тип нарушения | F1 [95% ДИ] | Pred+ |",
            "|---|---|---|",
        ]
        for vtype in cfg.VIOLATION_TYPES:
            type_entry = entry["by_type"][vtype]
            note = f" — {type_entry['note']}" if type_entry.get("note") else ""
            lines.append(f"| {vtype} | {_fmt(type_entry)} | {type_entry['predicted_positives']}{note} |")
        lines += ["", "Разбивка по регионам (бинарная часть, справочно):", ""]
        for region, region_entry in entry["by_region"].items():
            lines.append(
                f"- {region}: n={region_entry['n']}/{region_entry['positives']}, "
                f"F1 {_fmt(region_entry['binary_f1'])}, AUC {_fmt(region_entry['binary_auc'])}"
            )
        lines.append("")

    sanity = report.get("sanity")
    if sanity is not None:
        lines += [
            "## Sanity: submission_reference.csv против targets.csv",
            "",
            "Ожидание — 1.0 по всем метрикам (проверка функции метрик, не пайплайна).",
            "",
            f"- Бинарная F1: {_fmt(sanity['binary_f1'])}",
            f"- ROC-AUC: {_fmt(sanity['binary_auc'])}",
            f"- Macro-F1: {_fmt(sanity['macro_f1'])}",
        ]
        for vtype in cfg.VIOLATION_TYPES:
            lines.append(f"- {vtype}: {_fmt(sanity['by_type'][vtype])}")
        lines.append("")

    if "runtime_seconds" in report:
        lines += [
            "## Время прогона",
            "",
            f"- `describe_inputs` + `submission_from_manifest` на `cfg.STUDIES_DIR`: "
            f"{report['runtime_seconds']} c ({report['runtime_seconds_per_file']} c/файл).",
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def main() -> dict:
    started = time.perf_counter()
    manifest = describe_inputs(cfg.STUDIES_DIR)
    submission = submission_from_manifest(manifest)
    elapsed = time.perf_counter() - started

    submission = attach_pixel_hash(submission, manifest)
    manifest_reference = load_manifest(cfg.MANIFEST_PARQUET)
    targets = load_targets()
    linked = link_via_pixel_hash(submission, manifest_reference, targets)

    report = evaluate(linked, cfg.STAGE3_EVAL_BOOTSTRAP_N)
    report["sanity"] = sanity_check(cfg.STAGE3_EVAL_BOOTSTRAP_N, targets=targets)
    report["runtime_seconds"] = round(elapsed, 1)
    report["runtime_seconds_per_file"] = round(elapsed / max(len(submission), 1), 3)

    _write_report(report)
    print(f"отчёт: {STAGE3_REPORT_MD}", flush=True)
    for split_name in (*SPLITS, "all"):
        entry = report[split_name]
        print(
            f"{split_name}: n={entry['n']}/{entry['positives']}, "
            f"F1={_fmt(entry['binary_f1'])}, AUC={_fmt(entry['binary_auc'])}, "
            f"Macro-F1={_fmt(entry['macro_f1'])}",
            flush=True,
        )
    return report


if __name__ == "__main__":
    main()
