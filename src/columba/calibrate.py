"""Этап 6: калибровка `quality_prob`, пороги, метрики, чек-лист комментариев.

Запуск: `uv run python -m columba.calibrate`.

Идея (по образцу `hip_eval.py`/`spine_eval.py`, шаг 7/8 предыдущих этапов):
`aggregate.py` уже считает по каждому файлу region-агрегированный z-скор —
`max((score - threshold) / scale)` среди чекеров региона со статусом `ok`
(один и тот же приём что у `_spine_row_prediction`/`_hip_row_prediction`,
`quality_prob = sigmoid(z)` временно, до этой калибровки). Задача этапа —
заменить этот НАИВНЫЙ `sigmoid` на честно откалиброванное отображение
z -> вероятность (Platt/изотоника), подобранное на РЕАЛЬНЫХ исходах, с
метриками и порогами класса.

**Ключевая находка (см. `stages/stage_6.md`).** Прямое использование
уже посчитанного `quality_prob`/`score` из `describe_inputs` на train было бы
методологической ошибкой: `hip_positioning` использует CNN, чьи веса
обучены НА ВСЁМ train — инференс на train через развёрнутые веса даёт
почти идеальное разделение (AUC региона бедра 0.991 в `stage3_report.md`),
хотя честная OOF-оценка той же CNN — 0.765 (`stages/stage_4.md`).
Калибровка, подобранная на таких скорах, была бы переуверенной именно там,
где сигнала на самом деле меньше. Поэтому здесь z пересчитывается заново
на ЧЕСТНЫХ скорах: train — по вложенной групповой CV каждого чекера
(`spine_eval.nested_cv`/`hip_eval.nested_group_cv`, либо готовые OOF-предсказания
CNN из Colab), val — рантайм-скор (val ни разу не участвовал в подборе
порогов/весов ни одного чекера, поэтому уже честен без пересчёта).

Изотоника — без sklearn, свой PAVA (pool-adjacent-violators), как и весь
проект избегает sklearn/торчевого обучения вне Colab-шага.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .aggregate import _CHECKER_THRESHOLD_SCALE, _hip_checker_threshold_scale
from .hip_eval import _fit_logreg
from .hip_eval import build_signal_table as hip_signal_table
from .hip_eval import nested_group_cv as hip_nested_group_cv
from .inventory import load_manifest
from .spine_eval import RUNTIME_SETUPS as SPINE_RUNTIME_SETUPS
from .spine_eval import (
    best_f1_threshold,
    cluster_bootstrap_ci,
    compute_rule_scores,
    crest_col,
    flag_metrics,
    metal_col,
    nested_cv as spine_nested_cv,
    ranking_auc,
    spine_frame,
)

STAGE6_REPORT_MD = cfg.ARTIFACTS_DIR / "stage6_report.md"
STAGE6_REPORT_JSON = STAGE6_REPORT_MD.with_suffix(".json")
STAGE6_SCORES_CSV = STAGE6_REPORT_MD.with_suffix(".csv")

_NEG_INF = float("-inf")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def _ranked(scores: np.ndarray) -> np.ndarray:
    """NaN-скор ранжируется как наименее подозрительный (см. spine_eval/hip_eval)."""
    scores = np.asarray(scores, dtype=float)
    return np.where(np.isnan(scores), -1e9, scores)


# --------------------------------------------------------------------------- #
# Честный (OOF) region-агрегированный z-скор — позвоночник
# --------------------------------------------------------------------------- #

SPINE_CV_CANDIDATES: dict[str, dict[str, str]] = {
    "spine_axis": {"score": "score_axis"},
    "spine_positioning": {f"band_{b:g}": crest_col(b) for b in cfg.SPINE_CALIB_CREST_BAND_GRID_MM},
    "spine_objects": {
        f"t{t:g}_a{a:g}": metal_col(t, a)
        for t in cfg.SPINE_CALIB_METAL_THIN_GRID
        for a in cfg.SPINE_CALIB_METAL_AMBIENT_GRID
    },
}
SPINE_LABEL_COL = {"spine_axis": "y_axis", "spine_positioning": "y_positioning", "spine_objects": "y_objects"}
# `RUNTIME_SETUPS` называет предметы "spine_objects (rule)" (в скобках —
# отличать от CNN-ветки в отчёте spine_eval) — нормализуем к ключу
# `OUTPUT_LABELS`/`_CHECKER_THRESHOLD_SCALE`.
SPINE_DEPLOYED_COL = {
    name.split(" (")[0]: score_col for name, score_col, _flag_col, _label_col, _threshold in SPINE_RUNTIME_SETUPS
}


def build_oof_spine_table(verbose: bool = True) -> pd.DataFrame:
    """Позвоночник: train — OOF вложенной CV на чекер, val — рантайм-скор.

    z = max по трём чекерам региона `(score - threshold) / scale`
    (`aggregate._CHECKER_THRESHOLD_SCALE` — те же константы, что в проде, не
    копия). CNN-ветка предметов (`spine_objects_cnn`) НЕ переобучается здесь
    — она проиграла правилу на этапе 2 (AUC 0.772 против 0.820, decisions
    этапа 2) и не участвует в рантайме, пересчитывать её OOF заново для
    калибровки было бы дорого и бессмысленно (`spine_eval.run_eval` это
    делает для собственного сравнения, не для этого модуля).
    """
    frame = spine_frame()
    frame = compute_rule_scores(frame, verbose=verbose).reset_index(drop=True)
    train_mask = frame["fold"] == "train"

    z = pd.Series(_NEG_INF, index=frame.index)
    any_ok = pd.Series(False, index=frame.index)
    for key, candidates in SPINE_CV_CANDIDATES.items():
        label_col = SPINE_LABEL_COL[key]
        result = spine_nested_cv(frame, candidates, label_col)
        score = result["score"].astype(float)
        deployed = frame[SPINE_DEPLOYED_COL[key]].astype(float)
        score = score.where(train_mask, deployed)  # val: честный рантайм-скор
        threshold, scale = _CHECKER_THRESHOLD_SCALE[key]
        zi = (score - threshold) / scale
        ok = zi.notna()
        z = np.where(ok, np.maximum(z, zi.fillna(_NEG_INF)), z)
        any_ok = any_ok | ok
    z = pd.Series(z, index=frame.index)
    z[~any_ok] = np.nan

    out = frame[["dedup_group_id", "study_folder", "fold", "software_version"]].rename(columns={"fold": "split"})
    out["region"] = cfg.REGION_SPINE
    out["z_oof"] = z.values
    if verbose:
        print(f"позвоночник: z-скор собран на {len(out)} снимках", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Честный (OOF) region-агрегированный z-скор — бедро
# --------------------------------------------------------------------------- #


def build_oof_hip_table(verbose: bool = True) -> pd.DataFrame:
    """Бедро: `hip_positioning` — готовый OOF CNN из Colab (train — вложенная
    CV внутри ноутбука, val — веса, обученные только на train, честный
    холд-аут без пересчёта); `hip_roi` — train пересчитывается через
    `hip_eval.nested_group_cv` на `neg_rows_px` и переводится в шкалу
    `field_height_deficit` (та же величина с точностью до знака и сдвига
    на REF, `1 - rows_px/REF == 1 + neg_rows_px/REF`), val — рантайм-сигнал
    (не зависит от подбора REF, но для единообразия с чекером
    `hip_positioning` тоже берётся из вложенной CV, если фолд размечен).
    """
    table = hip_signal_table(verbose=verbose)
    train_mask = table["split"] == "train"

    oof_path, val_pred_path = cfg.HIP_POSITIONING_CNN_OOF_TRAIN_CSV, cfg.HIP_POSITIONING_CNN_VAL_PRED_CSV
    if oof_path.exists() and val_pred_path.exists():
        cnn = pd.concat(
            [pd.read_csv(oof_path), pd.read_csv(val_pred_path)], ignore_index=True
        ).rename(columns={"prob": "cnn_prob"})
        table = table.merge(cnn[["dedup_group_id", "cnn_prob"]], on="dedup_group_id", how="left")
    else:
        # CNN бедра — опциональный артефакт (веса + OOF-предсказания скачиваются
        # вручную, см. WEIGHTS.md); без него `hip_positioning` честно не даёт
        # z-скора, а не падает — калибровка всё равно работает по `hip_roi`.
        if verbose:
            print(f"нет OOF CNN бедра ({oof_path.name}) — hip_positioning исключён из z_oof", flush=True)
        table["cnn_prob"] = np.nan
    threshold_pos, scale_pos = _hip_checker_threshold_scale("hip_positioning")
    z_pos = (table["cnn_prob"] - threshold_pos) / scale_pos

    roi_cv = hip_nested_group_cv(table, ["neg_rows_px"], label_col="y_roi", verbose=verbose)
    ref = cfg.HIP_ROI_FIELD_HEIGHT_REF_PX
    deficit_oof_train = 1.0 + roi_cv["oof_score"].astype(float) / ref  # neg_rows_px -> field_height_deficit
    deficit = deficit_oof_train.where(train_mask, table["field_height_deficit"])
    threshold_roi, scale_roi = _hip_checker_threshold_scale("hip_roi")
    z_roi = (deficit - threshold_roi) / scale_roi

    z = np.maximum(z_pos.fillna(_NEG_INF).values, z_roi.fillna(_NEG_INF).values)
    any_ok = z_pos.notna().values | z_roi.notna().values
    z = np.where(any_ok, z, np.nan)

    out = table[["dedup_group_id", "study_folder", "split", "software_version"]].copy()
    out["region"] = cfg.REGION_HIP
    out["z_oof"] = z
    if verbose:
        print(f"бедро: z-скор собран на {len(out)} снимках", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Общая таблица + истинные метки
# --------------------------------------------------------------------------- #


def build_calibration_table(verbose: bool = True) -> pd.DataFrame:
    """Все размеченные изображения (`has_target`), region-агрегированный
    честный z-скор + истинный `quality_class`. Файлы, для которых ни один
    чекер региона не дал `ok` (обе части NaN) — исключаются: калибровке
    учиться не на чем, они и в рантайме уходят на приор, не на сигмоиду."""
    spine = build_oof_spine_table(verbose=verbose)
    hip = build_oof_hip_table(verbose=verbose)
    combined = pd.concat([spine, hip], ignore_index=True)

    targets = pd.read_csv(cfg.TARGETS_CSV)[["dedup_group_id", "quality_class", "region", "has_target", "split"]]
    targets = targets[targets["has_target"].fillna(False)]
    merged = targets.drop(columns=["region"]).merge(combined, on=["dedup_group_id", "split"], how="inner")
    merged = merged.rename(columns={"quality_class": "y_true"})
    merged = merged.dropna(subset=["z_oof"]).reset_index(drop=True)
    if verbose:
        dropped = len(targets) - len(merged)
        print(f"калибровочная таблица: {len(merged)} снимков ({dropped} без ok-чекера — исключены)", flush=True)
    return merged


# --------------------------------------------------------------------------- #
# Platt (логрегрессия) и изотоника (PAVA) — без sklearn
# --------------------------------------------------------------------------- #


def fit_platt(z: np.ndarray, y: np.ndarray) -> dict:
    """Platt-калибровка: `p = sigmoid(A*z + B)`, логрегрессия одного признака
    (`hip_eval._fit_logreg`, тот же градиентный спуск, что у правил бедра —
    явно допущено ТЗ как «правило поверх сигнала», не нейросеть)."""
    mean, std = float(np.mean(z)), float(np.std(z)) or 1.0
    z_std = (z - mean) / std
    weights, bias = _fit_logreg(
        z_std.reshape(-1, 1),
        y.astype(float),
        l2=cfg.STAGE6_PLATT_L2,
        lr=cfg.STAGE6_PLATT_LR,
        epochs=cfg.STAGE6_PLATT_EPOCHS,
    )

    def predict(z_new: np.ndarray) -> np.ndarray:
        z_new_std = (np.asarray(z_new, dtype=float) - mean) / std
        return _sigmoid(z_new_std * weights[0] + bias)

    return {"kind": "platt", "predict": predict, "mean": mean, "std": std, "weight": float(weights[0]), "bias": bias}


def _pava(values: np.ndarray) -> np.ndarray:
    """Pool-Adjacent-Violators: неубывающая изотоническая аппроксимация
    `values` (уже упорядоченных по возрастанию z), взвешенный стек, O(n)."""
    level_values: list[float] = []
    level_weights: list[float] = []
    level_counts: list[int] = []
    for value in values:
        level_values.append(float(value))
        level_weights.append(1.0)
        level_counts.append(1)
        while len(level_values) > 1 and level_values[-2] > level_values[-1]:
            v2, w2, c2 = level_values.pop(), level_weights.pop(), level_counts.pop()
            v1, w1, c1 = level_values.pop(), level_weights.pop(), level_counts.pop()
            level_values.append((v1 * w1 + v2 * w2) / (w1 + w2))
            level_weights.append(w1 + w2)
            level_counts.append(c1 + c2)
    result = np.empty(len(values))
    idx = 0
    for value, count in zip(level_values, level_counts):
        result[idx : idx + count] = value
        idx += count
    return result


def fit_isotonic(z: np.ndarray, y: np.ndarray) -> dict:
    """Изотоническая калибровка: неубывающая ступенчатая функция z -> p
    (PAVA), интерполяция для новых z — ближайшая известная ступень."""
    order = np.argsort(z, kind="stable")
    z_sorted = z[order]
    p_sorted = _pava(y.astype(float)[order])

    def predict(z_new: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(z_sorted, np.asarray(z_new, dtype=float), side="right") - 1
        idx = np.clip(idx, 0, len(p_sorted) - 1)
        # z ниже минимума train — экстраполяция первой ступенью (не 0):
        below = np.asarray(z_new, dtype=float) < z_sorted[0]
        out = p_sorted[idx]
        out = np.where(below, p_sorted[0], out)
        return out

    return {"kind": "isotonic", "predict": predict, "z_sorted": z_sorted, "p_sorted": p_sorted}


def brier_score(prob: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((prob - y) ** 2))


def log_loss(prob: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(prob, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --------------------------------------------------------------------------- #
# Метрики с 95% ДИ (кластерный бутстрэп по study_folder)
# --------------------------------------------------------------------------- #


def _estimate(frame: pd.DataFrame, statistic, n: int) -> dict:
    if len(frame) == 0:
        return {"value": float("nan"), "ci95": (float("nan"), float("nan"))}
    point = statistic(frame)
    ci = cluster_bootstrap_ci(frame, statistic, n)
    return {"value": round(point, 4) if point == point else float("nan"), "ci95": ci}


def calibration_metrics(frame: pd.DataFrame, prob_col: str, n_bootstrap: int) -> dict:
    def auc_stat(df: pd.DataFrame) -> float:
        return ranking_auc(_ranked(df[prob_col].values), df["y_true"].values.astype(int))

    def brier_stat(df: pd.DataFrame) -> float:
        return brier_score(df[prob_col].values, df["y_true"].values.astype(int))

    def logloss_stat(df: pd.DataFrame) -> float:
        return log_loss(df[prob_col].values, df["y_true"].values.astype(int))

    return {
        "n": len(frame),
        "positives": int(frame["y_true"].sum()),
        "auc": _estimate(frame, auc_stat, n_bootstrap),
        "brier": _estimate(frame, brier_stat, n_bootstrap),
        "log_loss": _estimate(frame, logloss_stat, n_bootstrap),
    }


def threshold_metrics(frame: pd.DataFrame, prob_col: str, threshold: float, n_bootstrap: int) -> dict:
    flags = frame[prob_col].values >= threshold

    def f1_stat(df: pd.DataFrame, thr=threshold) -> float:
        labels = df["y_true"].values.astype(int)
        if not labels.sum():
            return float("nan")
        return flag_metrics(df[prob_col].values >= thr, labels)["f1"]

    labels = frame["y_true"].values.astype(int)
    base = flag_metrics(flags, labels)
    return {
        "threshold": round(float(threshold), 4),
        **base,
        "f1_ci95": cluster_bootstrap_ci(frame, f1_stat, n_bootstrap) if labels.sum() else (float("nan"), float("nan")),
    }


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall) if (b2 * precision + recall) else 0.0


def recall_weighted_threshold(frame: pd.DataFrame, prob_col: str, beta: float) -> float:
    """Порог, максимизирующий F-beta (beta>1 — вес в пользу recall): ложный
    негатив («пропущенное нарушение укладки») стоит дороже ложной тревоги в
    контексте контроля качества — цена FN обсуждаема, поэтому вместе с этим
    порогом всегда печатается и обычный best-F1 (beta=1), выбор рабочей
    точки — отдельное решение (см. stages/stage_6.md, «Пороги класса»)."""
    labels = frame["y_true"].values.astype(int)
    candidates = np.unique(frame[prob_col].values)
    best_threshold, best_score = float("inf"), -1.0
    for threshold in candidates:
        predicted = frame[prob_col].values >= threshold
        m = flag_metrics(predicted, labels)
        score = _fbeta(m["precision"], m["recall"], beta)
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score, best_threshold = score, threshold
    return float(best_threshold)


# --------------------------------------------------------------------------- #
# Чек-лист комментариев врачей
# --------------------------------------------------------------------------- #


def doctor_comment_checklist(frame: pd.DataFrame, prob_col: str, threshold: float) -> pd.DataFrame:
    """Снимки с непустым `markup_comment` (уровень исследования, см.
    `stages/stage_4.md`) — чек-лист сложных случаев (шаг ТЗ этапа 6): как
    пайплайн (откалиброванный порог) справляется именно на них."""
    manifest = load_manifest()
    reps = manifest[manifest["is_group_representative"].fillna(False)][["dedup_group_id", "markup_comment"]]
    merged = frame.merge(reps, on="dedup_group_id", how="left")
    commented = merged[merged["markup_comment"].notna() & (merged["markup_comment"].str.strip() != "")].copy()
    commented["predicted"] = commented[prob_col] >= threshold
    commented["true"] = commented["y_true"].astype(bool)
    commented["correct"] = commented["predicted"] == commented["true"]
    return commented[
        ["dedup_group_id", "region", "split", "markup_comment", "y_true", prob_col, "predicted", "correct"]
    ].sort_values(["correct", "region", "dedup_group_id"])


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #


def _fmt(estimate: dict) -> str:
    value, (low, high) = estimate["value"], estimate["ci95"]
    if value != value:
        return "NaN"
    if low != low:
        return f"{value} [ДИ н/о]"
    return f"{value} [{low}; {high}]"


def _write_report(report: dict, path: Path = STAGE6_REPORT_MD) -> None:
    lines = [
        "# Отчёт калибровки этапа 6",
        "",
        "Генерируется `uv run python -m columba.calibrate`. z-скор — честный",
        "(train: вложенная CV/Colab-OOF на чекер, val: рантайм-скор, не",
        "участвовавший в подборе порогов того чекера). 95% ДИ — кластерный",
        f"бутстрэп по `study_folder` ({cfg.STAGE6_EVAL_BOOTSTRAP_N} реплик).",
        "",
        "## Калибровка: Platt против изотоники",
        "",
    ]
    for name in ("naive_sigmoid", "platt", "isotonic"):
        entry = report["calibration"].get(name)
        if not entry:
            continue
        lines.append(f"### {name}")
        for split_name in ("train", "val"):
            m = entry[split_name]
            lines.append(
                f"- {split_name}: n={m['n']}/{m['positives']}, AUC {_fmt(m['auc'])}, "
                f"Brier {_fmt(m['brier'])}, log-loss {_fmt(m['log_loss'])}"
            )
        lines.append("")

    lines += [f"**Выбор: {report['chosen_calibration']}** — {report['calibration_choice_reason']}", ""]

    lines += ["## Пороги класса (на откалиброванной вероятности)", ""]
    for name, entry in report["thresholds"].items():
        lines += [
            f"### {name} (порог {entry['train']['threshold']})",
            "",
        ]
        for split_name in ("train", "val"):
            m = entry[split_name]
            lines.append(
                f"- {split_name}: precision {m['precision']}, recall {m['recall']}, "
                f"F1 {m['f1']} {_fmt({'value': m['f1'], 'ci95': m['f1_ci95']})}, "
                f"TP/FP/FN/TN {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}"
            )
        lines.append("")

    checklist = report.get("doctor_checklist_summary")
    if checklist:
        lines += [
            "## Чек-лист комментариев врачей",
            "",
            f"n={checklist['n']}, верно={checklist['correct']} ({checklist['accuracy']}), "
            f"неверно={checklist['incorrect']}",
            "",
            "Ошибки (см. `.csv` для полного списка с комментариями):",
            "",
        ]
        for row in checklist["incorrect_rows"]:
            lines.append(f"- {row}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def main() -> dict:
    table = build_calibration_table()
    train_mask = table["split"] == "train"
    z_train = table.loc[train_mask, "z_oof"].values
    y_train = table.loc[train_mask, "y_true"].values.astype(int)

    naive = {"predict": _sigmoid}
    platt = fit_platt(z_train, y_train)
    isotonic = fit_isotonic(z_train, y_train)

    report: dict = {"calibration": {}}
    prob_cols = {}
    for name, model in (("naive_sigmoid", naive), ("platt", platt), ("isotonic", isotonic)):
        col = f"prob_{name}"
        table[col] = model["predict"](table["z_oof"].values)
        prob_cols[name] = col
        report["calibration"][name] = {
            split_name: calibration_metrics(table[table["split"] == split_name], col, cfg.STAGE6_EVAL_BOOTSTRAP_N)
            for split_name in ("train", "val")
        }
    train = table[train_mask]

    # Выбор: меньший Brier на val (честный холд-аут) — калибровка мерится
    # качеством вероятности, не ранжированием (AUC монотонными
    # преобразованиями z не меняется вообще, поэтому одинаков у всех трёх).
    val_brier = {name: report["calibration"][name]["val"]["brier"]["value"] for name in ("platt", "isotonic")}
    chosen = min(val_brier, key=lambda n: val_brier[n] if val_brier[n] == val_brier[n] else float("inf"))
    report["chosen_calibration"] = chosen
    report["calibration_choice_reason"] = (
        f"меньший Brier на val: platt={val_brier['platt']}, isotonic={val_brier['isotonic']}"
    )
    chosen_col = prob_cols[chosen]

    report["thresholds"] = {}
    for name, threshold in (
        ("default_0.5", 0.5),
        ("best_f1_train", best_f1_threshold(train[chosen_col].values, y_train)["threshold"]),
        ("recall_weighted_f2_train", recall_weighted_threshold(train, chosen_col, beta=2.0)),
    ):
        report["thresholds"][name] = {
            split_name: threshold_metrics(
                table[table["split"] == split_name], chosen_col, threshold, cfg.STAGE6_EVAL_BOOTSTRAP_N
            )
            for split_name in ("train", "val")
        }

    checklist = doctor_comment_checklist(table, chosen_col, report["thresholds"]["best_f1_train"]["train"]["threshold"])
    checklist.to_csv(STAGE6_REPORT_MD.with_name("stage6_doctor_checklist.csv"), index=False)
    incorrect = checklist[~checklist["correct"]]
    report["doctor_checklist_summary"] = {
        "n": len(checklist),
        "correct": int(checklist["correct"].sum()),
        "incorrect": len(incorrect),
        "accuracy": round(float(checklist["correct"].mean()), 3) if len(checklist) else float("nan"),
        "incorrect_rows": [
            f"{r.dedup_group_id} ({r.region}, {r.split}): «{r.markup_comment}», true={int(r.y_true)}, "
            f"prob={round(float(getattr(r, chosen_col)), 3)}"
            for r in incorrect.itertuples()
        ],
    }

    table.to_csv(STAGE6_SCORES_CSV, index=False)
    _write_report(report)
    print(f"отчёт: {STAGE6_REPORT_MD}; чек-лист: stage6_doctor_checklist.csv", flush=True)
    print(f"выбрана калибровка: {chosen}", flush=True)
    for split_name in ("train", "val"):
        m = report["calibration"][chosen][split_name]
        print(f"{split_name}: AUC={_fmt(m['auc'])}, Brier={_fmt(m['brier'])}", flush=True)
    return report


if __name__ == "__main__":
    main()
