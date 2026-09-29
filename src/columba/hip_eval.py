"""Этап 4, шаг 7: калибровка порогов и метрики чекера(ов) бедра.

Запуск: `uv run python -m columba.hip_eval`.

Что делает (по образцу `spine_eval.py`, этап 2 шаг 8):

* `build_signal_table` собирает таблицу сигналов на реальных снимках бедра
  (`artifacts/manifest.csv`/`targets.csv`, `region == hip`, `has_target`) —
  ориентиры и сигналы считает `hip_landmarks` (модуль разрабатывался
  параллельно; импортируется ВНУТРИ функции, чтобы этот модуль без него
  оставался импортируемым и тестируемым на синтетике);
* `signal_auc_table` — разделяющая способность (AUC + 95% ДИ) каждого
  сигнала-кандидата и его |модуля| на train/val, а также AUC как разделителя
  версии софта (риск прокси версии, `stages/stage_4.md`);
* `nested_group_cv` — вложенная групповая CV по `study_folder` внутри train
  (аналог `spine_eval.nested_cv`, конвенция `SPINE_EVAL_CV_FOLDS`-фолдов):
  на обучающей части фолда выбирается конструкция-кандидат (один сигнал,
  его |модуль|, либо логрегрессия на numpy по подмножеству сигналов) по AUC
  и порог флага по max F1, метрики — OOF, с 95% ДИ кластерного бутстрэпа;
* `runtime_operating_point` — метрики фиксированного правила (готовые
  score_fn + порог) на train (in-sample) и val (смоук);
* `version_breakdown` — метрики по версиям софта отдельно; явный красный
  флаг, если AUC скора как разделителя версии заметно выше AUC против метки;
* `main()` пишет `artifacts/hip_checker_report.md` + `.json` + `.csv`.

Переиспользует `ranking_auc`, `flag_metrics`, `best_f1_threshold`,
`cluster_bootstrap_ci`, `_ranked`, `_auc_stat`, `_f1_stat` из `spine_eval` —
не дублирует их. sklearn не используется; единственное «обучение» —
логистическая регрессия на numpy (градиентный спуск, 2-5 признаков),
что явно допущено ТЗ как правило поверх сигналов, а не нейросеть.

Новые константы объявлены здесь (модульно), `config.py` не тронут:
`HIP_EVAL_BOOTSTRAP_N`, `HIP_EVAL_REPORT`, `HIP_LOGREG_L2`, `HIP_LOGREG_LR`,
`HIP_LOGREG_EPOCHS`, `HIP_VERSION_REDFLAG_MARGIN`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .inventory import load_manifest
from .spine_eval import (
    _auc_stat,
    _f1_stat,
    _ranked,
    best_f1_threshold,
    cluster_bootstrap_ci,
    flag_metrics,
    ranking_auc,
)

# --------------------------------------------------------------------------- #
# Константы (модульные — config.py не трогаем)
# --------------------------------------------------------------------------- #

N_CV_FOLDS = cfg.SPINE_EVAL_CV_FOLDS  # тот же принцип, что у spine_eval
HIP_EVAL_BOOTSTRAP_N = 2000  # кластерный бутстрэп по study_folder, 95% ДИ
HIP_EVAL_REPORT = cfg.ARTIFACTS_DIR / "hip_checker_report.md"

# Гиперпараметры логрегрессии-кандидата (numpy, не нейросеть; подбор
# коэффициентов на CPU за секунды — как SPINE_METAL_MIN_SCORE и т. п.).
HIP_LOGREG_L2 = 1.0
HIP_LOGREG_LR = 0.5
HIP_LOGREG_EPOCHS = 500

# Порог «заметно выше»: AUC скора как разделителя версии софта считается
# красным флагом, только если превышает AUC против метки более чем на
# столько (иначе шумовое превышение при маленьких n даёт ложную тревогу).
HIP_VERSION_REDFLAG_MARGIN = 0.05

# Колонки-метаданные таблицы сигналов (см. build_signal_table); всё
# остальное в таблице считается колонкой сигнала.
METADATA_COLS = {
    "dedup_group_id",
    "study_folder",
    "split",
    "software_version",
    "y_positioning",
    "y_roi",
}


def _signal_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in METADATA_COLS]


# --------------------------------------------------------------------------- #
# Таблица сигналов на реальных данных
# --------------------------------------------------------------------------- #


def build_signal_table(verbose: bool = True) -> pd.DataFrame:
    """Собирает таблицу сигналов бедра на реальных данных.

    `hip_landmarks` (детектор ориентиров + сигналов) разрабатывался
    параллельно — импорт нарочно внутри функции, чтобы модуль оставался
    импортируемым и тестируемым и без него.
    """
    from . import hip_landmarks  # noqa: PLC0415 — намеренно внутри функции
    from .dicom_io import normalize, read_dicom_strict

    targets = pd.read_csv(cfg.TARGETS_CSV)
    manifest = load_manifest()
    reps = manifest[manifest["is_group_representative"].fillna(False)][
        ["dedup_group_id", "abs_path", "software_version"]
    ]
    frame = targets[(targets["region"] == cfg.REGION_HIP) & (targets["has_target"] == True)].merge(
        reps, on="dedup_group_id", how="left"
    )

    rows = []
    for row in frame.itertuples():
        result = read_dicom_strict(row.abs_path)
        pixels = normalize(result.pixels, result.tags)
        landmarks = hip_landmarks.detect_hip_landmarks(pixels, side=row.hip_side)
        signals = hip_landmarks.compute_hip_signals(pixels, landmarks)
        roi_signals = hip_landmarks.compute_hip_roi_signals(pixels, landmarks)
        entry = {
            "dedup_group_id": row.dedup_group_id,
            "study_folder": row.study_folder,
            "split": row.split,
            "software_version": row.software_version,
            "y_positioning": int(row.label_hip_positioning),
            "y_roi": int(row.label_hip_roi),
        }
        for key in hip_landmarks.HIP_SIGNAL_KEYS:
            entry[key] = signals.get(key, np.nan)
        for key in hip_landmarks.HIP_ROI_SIGNAL_KEYS:
            entry[key] = roi_signals.get(key, np.nan)
        # Кандидат для калибровки шага 4 этапа 5: монотонный «выше = хуже»
        # без привязки к REF (в отличие от field_height_deficit, зависящего
        # от уже откалиброванного config.HIP_ROI_FIELD_HEIGHT_REF_PX) — так
        # `nested_group_cv`/`best_f1_threshold` калибруют REF по факту, не
        # предполагая его заранее.
        entry["neg_rows_px"] = -entry["rows_px"]
        entry["neg_top_margin_px"] = -entry["top_margin_px"]
        entry["neg_bottom_margin_px"] = -entry["bottom_margin_px"]
        rows.append(entry)
    if verbose:
        print(f"таблица сигналов бедра: {len(rows)} снимков", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# AUC сигналов-кандидатов
# --------------------------------------------------------------------------- #


def _version_binary(table: pd.DataFrame) -> tuple[pd.Series, pd.Series, tuple] | None:
    """Бинарная разметка «версия A / версия B» по двум самым частым версиям.

    None, если версий меньше двух (разделитель не определён).
    """
    counts = table["software_version"].value_counts()
    if len(counts) < 2:
        return None
    top2 = tuple(counts.index[:2])
    mask = table["software_version"].isin(top2)
    labels = (table.loc[mask, "software_version"] == top2[1]).astype(int)
    return mask, labels, top2


def _estimate(frame: pd.DataFrame, statistic, n_bootstrap: int) -> dict:
    point = statistic(frame)
    return {
        "value": round(point, 3) if point == point else float("nan"),
        "ci95": cluster_bootstrap_ci(frame, statistic, n=n_bootstrap),
    }


def signal_auc_table(
    table: pd.DataFrame,
    label_col: str = "y_positioning",
    n_bootstrap: int = HIP_EVAL_BOOTSTRAP_N,
) -> pd.DataFrame:
    """AUC каждого сигнала (raw и |raw|) на train/val + AUC как разделителя
    версии софта, все — с 95% ДИ кластерного бутстрэпа по `study_folder`.
    """
    rows: list[dict] = []
    version_bin = _version_binary(table)

    for col in _signal_columns(table):
        for transform_name, use_abs in (("raw", False), ("abs", True)):

            def stat(df: pd.DataFrame, col=col, use_abs=use_abs, label_col=label_col) -> float:
                values = df[col].astype(float)
                if use_abs:
                    values = values.abs()
                return ranking_auc(_ranked(values.values), df[label_col].values.astype(int))

            for fold_name in ("train", "val"):
                subset = table[table["split"] == fold_name]
                if not len(subset):
                    continue
                estimate = _estimate(subset, stat, n_bootstrap)
                rows.append(
                    {
                        "signal": col,
                        "transform": transform_name,
                        "fold": fold_name,
                        "n": len(subset),
                        "positives": int(subset[label_col].sum()),
                        "nan_fraction": round(float(subset[col].isna().mean()), 3),
                        "auc": estimate["value"],
                        "ci_low": estimate["ci95"][0],
                        "ci_high": estimate["ci95"][1],
                    }
                )

            if version_bin is not None:
                mask, version_labels, versions = version_bin
                vsubset = table.loc[mask].assign(_version_label=version_labels.values)

                def vstat(df: pd.DataFrame, col=col, use_abs=use_abs) -> float:
                    values = df[col].astype(float)
                    if use_abs:
                        values = values.abs()
                    return ranking_auc(_ranked(values.values), df["_version_label"].values.astype(int))

                estimate = _estimate(vsubset, vstat, n_bootstrap)
                rows.append(
                    {
                        "signal": col,
                        "transform": transform_name,
                        "fold": f"vs_software_version {versions}",
                        "n": len(vsubset),
                        "positives": int(version_labels.sum()),
                        "nan_fraction": round(float(vsubset[col].isna().mean()), 3),
                        "auc": estimate["value"],
                        "ci_low": estimate["ci95"][0],
                        "ci_high": estimate["ci95"][1],
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Логрегрессия-кандидат (numpy, без sklearn) — правило поверх сигналов
# --------------------------------------------------------------------------- #


def _fit_logreg(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = HIP_LOGREG_L2,
    lr: float = HIP_LOGREG_LR,
    epochs: int = HIP_LOGREG_EPOCHS,
    seed: int = cfg.SEED,
) -> tuple[np.ndarray, float]:
    """Градиентный спуск L2-регуляризованной логрегрессии. Детерминирован
    при фиксированном seed (используется только для инициализации весов)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if n == 0:
        return np.zeros(d), 0.0
    weights = rng.normal(0.0, 0.01, d)
    bias = 0.0
    for _ in range(epochs):
        z = X @ weights + bias
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = X.T @ (p - y) / n + l2 * weights / n
        grad_b = float((p - y).mean())
        weights = weights - lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def _fit_logreg_scorer(fit_frame: pd.DataFrame, cols: list[str], label_col: str):
    """Обучает логрегрессию на `fit_frame` (стандартизация/импутация — по
    fit-части, без утечки) и возвращает score_fn(frame) -> np.ndarray."""
    fit_vals = fit_frame[cols].astype(float)
    impute = fit_vals.median().fillna(0.0)
    fit_imputed = fit_vals.fillna(impute)
    mean = fit_imputed.mean()
    std = fit_imputed.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    X_fit = ((fit_imputed - mean) / std).values
    y_fit = fit_frame[label_col].values.astype(float)
    weights, bias = _fit_logreg(X_fit, y_fit)

    def score(frame: pd.DataFrame) -> np.ndarray:
        values = frame[cols].astype(float).fillna(impute)
        X = ((values - mean) / std).values
        z = X @ weights + bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    return score


def _candidate_scores(fit: pd.DataFrame, hold: pd.DataFrame, spec, label_col: str):
    """Скоры кандидата на fit/hold. `spec` — имя колонки сигнала (raw),
    `"abs:<колонка>"` (|сигнал|), либо кортеж/список колонок (логрегрессия)."""
    if isinstance(spec, str):
        if spec.startswith("abs:"):
            col = spec[4:]
            return fit[col].astype(float).abs().values, hold[col].astype(float).abs().values
        return fit[spec].astype(float).values, hold[spec].astype(float).values
    cols = list(spec)
    scorer = _fit_logreg_scorer(fit, cols, label_col)
    return scorer(fit), scorer(hold)


def _make_score_fn(train: pd.DataFrame, spec, label_col: str):
    """Как `_candidate_scores`, но возвращает переиспользуемую функцию
    score_fn(frame) — для рабочей точки рантайма (обучение один раз на train)."""
    if isinstance(spec, str):
        if spec.startswith("abs:"):
            col = spec[4:]
            return lambda frame: frame[col].astype(float).abs().values
        return lambda frame: frame[spec].astype(float).values
    return _fit_logreg_scorer(train, list(spec), label_col)


def _describe_candidate(spec) -> str:
    if isinstance(spec, str):
        return spec
    return "logreg(" + ",".join(spec) + ")"


# --------------------------------------------------------------------------- #
# Вложенная групповая CV
# --------------------------------------------------------------------------- #


def _study_cv_folds(table: pd.DataFrame, n_folds: int) -> pd.Series:
    """Номер CV-фолда для строк train (по `study_folder`), NaN для val.

    Один study_folder всегда попадает в ровно один фолд — гарантия
    отсутствия утечки между внутренним обучением и отложенной частью
    (проверяется явно тестами)."""
    train_mask = table["split"] == "train"
    studies = sorted(table.loc[train_mask, "study_folder"].unique())
    fold_of_study = {study: i % n_folds for i, study in enumerate(studies)}
    return table["study_folder"].map(fold_of_study).where(train_mask)


def _summary(frame: pd.DataFrame, score_col: str, flag_col: str, label_col: str, n_bootstrap: int) -> dict:
    labels = frame[label_col].values.astype(int)
    return {
        "n": len(frame),
        "positives": int(labels.sum()),
        **flag_metrics(frame[flag_col].values, labels),
        "auc": _estimate(frame, _auc_stat(score_col, label_col), n_bootstrap),
        "f1": _estimate(frame, _f1_stat(flag_col, label_col), n_bootstrap),
    }


def nested_group_cv(
    table: pd.DataFrame,
    candidates: list,
    label_col: str = "y_positioning",
    n_folds: int | None = None,
    n_bootstrap: int = HIP_EVAL_BOOTSTRAP_N,
    verbose: bool = True,
) -> dict:
    """Вложенная групповая `n_folds`-фолдовая CV по `study_folder` внутри
    train. В каждом фолде на обучающей части выбирается конструкция-кандидат
    (максимум AUC, при равенстве — первая в порядке списка `candidates`) и
    порог флага (максимум F1), затем применяются к отложенной части. Seed —
    `config.SEED` (через `_fit_logreg`), воспроизводимо между запусками.
    """
    n_folds = n_folds or N_CV_FOLDS
    cv_fold = _study_cv_folds(table, n_folds)
    oof_score = pd.Series(np.nan, index=table.index)
    oof_flag = pd.Series(False, index=table.index)
    folds_log = []
    for fold_id in range(n_folds):
        fit_mask = cv_fold.notna() & (cv_fold != fold_id)
        hold_mask = cv_fold == fold_id
        if not hold_mask.any():
            continue
        fit = table[fit_mask]
        hold = table[hold_mask]
        labels_fit = fit[label_col].values.astype(int)

        best_auc = float("-inf")
        best_spec, best_fit_scores, best_hold_scores = None, None, None
        for spec in candidates:
            fit_scores, hold_scores = _candidate_scores(fit, hold, spec, label_col)
            auc = ranking_auc(_ranked(fit_scores), labels_fit)
            auc_cmp = auc if auc == auc else float("-inf")
            if auc_cmp > best_auc:
                best_auc = auc_cmp
                best_spec, best_fit_scores, best_hold_scores = spec, fit_scores, hold_scores

        threshold = best_f1_threshold(_ranked(best_fit_scores), labels_fit)["threshold"]
        oof_score.loc[hold.index] = best_hold_scores
        oof_flag.loc[hold.index] = _ranked(best_hold_scores) >= threshold
        folds_log.append(
            {
                "fold": fold_id,
                "candidate": _describe_candidate(best_spec),
                "fit_auc": round(best_auc, 3) if best_auc == best_auc and best_auc != float("-inf") else float("nan"),
                "threshold": round(float(threshold), 4),
                "holdout_positives": int(hold[label_col].sum()),
                "holdout_studies": sorted(hold["study_folder"].astype(str).unique().tolist()),
            }
        )

    train_mask = table["split"] == "train"
    train_oof = table.loc[train_mask].assign(
        _oof_score=oof_score.loc[train_mask], _oof_flag=oof_flag.loc[train_mask]
    )
    summary = _summary(train_oof, "_oof_score", "_oof_flag", label_col, n_bootstrap)
    if verbose:
        print(f"nested_group_cv[{label_col}]: {len(folds_log)} фолдов", flush=True)
    return {"folds": folds_log, "oof_score": oof_score, "oof_flag": oof_flag, "summary": summary}


# --------------------------------------------------------------------------- #
# Рабочая точка рантайма
# --------------------------------------------------------------------------- #


def runtime_operating_point(
    table: pd.DataFrame,
    score_fn,
    threshold: float,
    label_col: str = "y_positioning",
    n_bootstrap: int = HIP_EVAL_BOOTSTRAP_N,
) -> dict:
    """Метрики фиксированного правила (`score_fn`/`threshold` — ровно как в
    рантайме) на train (in-sample) и val (смоук)."""
    raw_scores = np.asarray(score_fn(table), dtype=float)
    ranked = _ranked(raw_scores)
    flags = ranked >= threshold
    frame = table.assign(_score=raw_scores, _flag=flags)
    return {
        fold_name: _summary(frame[frame["split"] == fold_name], "_score", "_flag", label_col, n_bootstrap)
        for fold_name in ("train", "val")
    }


# --------------------------------------------------------------------------- #
# Разбивка по версиям софта
# --------------------------------------------------------------------------- #


def version_breakdown(
    table: pd.DataFrame,
    scores,
    label_col: str = "y_positioning",
    n_bootstrap: int = HIP_EVAL_BOOTSTRAP_N,
) -> dict:
    """Метрики по версиям софта отдельно + красный флаг, если AUC скора как
    разделителя версии заметно (> `HIP_VERSION_REDFLAG_MARGIN`) выше AUC
    против метки (риск прокси версии софта, `stages/stage_4.md`)."""
    score_values = table[scores].values if isinstance(scores, str) else np.asarray(scores, dtype=float)
    frame = table.assign(_score=score_values)

    def label_stat(df: pd.DataFrame) -> float:
        return ranking_auc(_ranked(df["_score"].values), df[label_col].values.astype(int))

    by_version = {}
    for version, chunk in frame.groupby("software_version"):
        labels = chunk[label_col].values.astype(int)
        by_version[str(version)] = {
            "n": len(chunk),
            "positives": int(labels.sum()),
            "auc_vs_label": _estimate(chunk, label_stat, n_bootstrap),
        }

    result = {
        "by_version": by_version,
        "auc_vs_label_overall": _estimate(frame, label_stat, n_bootstrap),
        "auc_vs_version": None,
        "compared_versions": None,
        "red_flag": False,
    }
    version_bin = _version_binary(table)
    if version_bin is not None:
        mask, version_labels, versions = version_bin
        vsubset = frame.loc[mask].assign(_version_label=version_labels.values)

        def version_stat(df: pd.DataFrame) -> float:
            return ranking_auc(_ranked(df["_score"].values), df["_version_label"].values.astype(int))

        result["auc_vs_version"] = _estimate(vsubset, version_stat, n_bootstrap)
        result["compared_versions"] = versions
        auc_v = result["auc_vs_version"]["value"]
        auc_l = result["auc_vs_label_overall"]["value"]
        if auc_v == auc_v and auc_l == auc_l:
            result["red_flag"] = bool(auc_v > auc_l + HIP_VERSION_REDFLAG_MARGIN)
    return result


# --------------------------------------------------------------------------- #
# CNN-ветка (шаг 6): сравнение с геометрическим чекером на тех же метках
# --------------------------------------------------------------------------- #


def cnn_operating_point(
    table: pd.DataFrame,
    label_col: str = "y_positioning",
    n_bootstrap: int = HIP_EVAL_BOOTSTRAP_N,
) -> dict | None:
    """OOF/val-метрики CNN-ветки (`notebooks/hip_positioning_cnn.ipynb`)
    на тех же реальных метках и той же статистике (AUC/F1, 95% ДИ), что
    `nested_group_cv` — для честного сравнения на одной шкале. `None`, если
    предсказания ещё не скачаны из Colab (честная деградация отчёта, шаг 6
    остаётся опциональным).

    Предсказания уже out-of-fold (4-фолдовая групповая CV внутри Colab,
    seed 42) — здесь их AUC независимо пересчитывается заново на РЕАЛЬНЫХ
    метках (не доверяем числу из `report.json` без проверки), порог —
    `best_f1_threshold` на train (тот же метод, что у геометрического
    правила), val — смоук.
    """
    oof_path = cfg.HIP_POSITIONING_CNN_OOF_TRAIN_CSV
    val_path = cfg.HIP_POSITIONING_CNN_VAL_PRED_CSV
    if not oof_path.exists() or not val_path.exists():
        return None

    preds = pd.concat([pd.read_csv(oof_path), pd.read_csv(val_path)], ignore_index=True)
    preds = preds.rename(columns={"prob": "_score"})
    frame = table.merge(preds[["dedup_group_id", "_score"]], on="dedup_group_id", how="inner")

    train_labels = frame.loc[frame["split"] == "train", label_col].values.astype(int)
    train_scores = frame.loc[frame["split"] == "train", "_score"].values
    threshold = float(best_f1_threshold(train_scores, train_labels)["threshold"])
    frame = frame.assign(_flag=frame["_score"].values >= threshold)

    return {
        "n_matched": len(frame),
        "threshold": round(threshold, 4),
        "prob_scale_std_train": round(float(np.std(train_scores)), 4),
        **{
            fold_name: _summary(frame[frame["split"] == fold_name], "_score", "_flag", label_col, n_bootstrap)
            for fold_name in ("train", "val")
        },
    }


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #


def _write_report(report: dict, path: Path) -> None:
    lines = [
        "# Отчёт чекера бедра (этап 4, шаг 7)",
        "",
        "Генерируется `uv run python -m columba.hip_eval` (машиночитаемая копия —",
        "`.json` рядом, скоры по изображениям — `.csv`). 95% ДИ — кластерный",
        f"бутстрэп по `study_folder` ({HIP_EVAL_BOOTSTRAP_N} реплик).",
        "",
        "## 1. AUC сигналов-кандидатов",
        "",
    ]
    for row in report.get("signal_auc", []):
        lines.append(
            f"- {row['signal']} ({row['transform']}, {row['fold']}): AUC {row['auc']} "
            f"[{row['ci_low']}; {row['ci_high']}], n={row['n']}/{row['positives']}, "
            f"NaN {row['nan_fraction']}"
        )
    for key, title in (("cv_positioning", "hip_positioning"), ("cv_roi", "hip_roi")):
        entry = report.get(key)
        if not entry:
            continue
        summary = entry["summary"]
        lines += [
            "",
            f"## Вложенная CV: {title}",
            "",
            f"n/позитивов {summary['n']}/{summary['positives']}, AUC {summary['auc']['value']} "
            f"[{summary['auc']['ci95'][0]}; {summary['auc']['ci95'][1]}], "
            f"F1 {summary['f1']['value']} [{summary['f1']['ci95'][0]}; {summary['f1']['ci95'][1]}], "
            f"TP/FP/FN {summary['tp']}/{summary['fp']}/{summary['fn']}",
            "",
            "Выбор по фолдам:",
        ]
        for fold in entry["folds"]:
            lines.append(
                f"- ф{fold['fold']}: {fold['candidate']}, AUC {fold['fit_auc']}, "
                f"порог {fold['threshold']}, позитивов в отложенной {fold['holdout_positives']}"
            )
    vb = report.get("version_breakdown_positioning")
    if vb:
        lines += ["", "## Разбивка по версиям софта (hip_positioning)", ""]
        for version, entry in vb["by_version"].items():
            lines.append(
                f"- {version}: n={entry['n']}, позитивов {entry['positives']}, "
                f"AUC против метки {entry['auc_vs_label']['value']}"
            )
        if vb["auc_vs_version"] is not None:
            lines.append(
                f"- AUC как разделителя версии {vb['compared_versions']}: "
                f"{vb['auc_vs_version']['value']} против AUC-против-метки "
                f"{vb['auc_vs_label_overall']['value']}"
            )
            lines.append(f"- **Красный флаг риска версии: {vb['red_flag']}**")
    cnn = report.get("cnn_positioning")
    if cnn:
        lines += [
            "",
            "## CNN-ветка (шаг 6) против геометрического чекера",
            "",
            f"n сматчено {cnn['n_matched']}, порог (best-F1 на train OOF) {cnn['threshold']}, "
            f"scale (std train OOF) {cnn['prob_scale_std_train']}",
        ]
        for fold_name in ("train", "val"):
            summary = cnn[fold_name]
            lines.append(
                f"- {fold_name}: AUC {summary['auc']['value']} "
                f"[{summary['auc']['ci95'][0]}; {summary['auc']['ci95'][1]}], "
                f"F1 {summary['f1']['value']} [{summary['f1']['ci95'][0]}; {summary['f1']['ci95'][1]}], "
                f"TP/FP/FN {summary['tp']}/{summary['fp']}/{summary['fn']}"
            )
    elif "cnn_positioning" in report:
        lines += ["", "## CNN-ветка (шаг 6)", "", "Предсказания не найдены — CNN не обучена/не скачана."]
    errors = report.get("errors_positioning")
    if errors:
        lines += [
            "",
            "## Ошибки (OOF hip_positioning)",
            "",
            f"- FP: {errors['false_positives'] or '—'}",
            f"- FN: {errors['false_negatives'] or '—'}",
        ]
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    from . import hip_landmarks  # noqa: PLC0415 — только для разделения кандидатов по чекеру

    table = build_signal_table()
    signals = _signal_columns(table)

    # Кандидаты — отдельно по чекеру, не общий список: `hip_roi` имеет всего
    # 6 train-позитивов, многомерная логрегрессия по всем 11 сигналам сразу
    # (общий список ниже — для `hip_positioning`, где train-позитивов 29)
    # на такой выборке — верный путь к переобучению одного фолда случайным
    # образом. Для `hip_roi` — только одиночные сигналы-кандидаты, без
    # логрегрессии и без `abs:` (все направленные: меньше = хуже).
    positioning_signals = [c for c in hip_landmarks.HIP_SIGNAL_KEYS if c in signals]
    positioning_candidates: list = list(positioning_signals) + [f"abs:{c}" for c in positioning_signals]
    if len(positioning_signals) >= 2:
        positioning_candidates.append(tuple(positioning_signals))

    roi_candidates: list = ["neg_rows_px"]
    if "neg_top_margin_px" in signals:
        roi_candidates.append("neg_top_margin_px")
    if "neg_bottom_margin_px" in signals:
        roi_candidates.append("neg_bottom_margin_px")

    report: dict = {"signal_auc": signal_auc_table(table).to_dict(orient="records")}

    cv_positioning = nested_group_cv(table, positioning_candidates, label_col="y_positioning")
    cv_roi = nested_group_cv(table, roi_candidates, label_col="y_roi")
    report["cv_positioning"] = {"folds": cv_positioning["folds"], "summary": cv_positioning["summary"]}
    report["cv_roi"] = {"folds": cv_roi["folds"], "summary": cv_roi["summary"]}

    report["version_breakdown_positioning"] = version_breakdown(
        table, cv_positioning["oof_score"].fillna(0.0), label_col="y_positioning"
    )

    oof_flag = cv_positioning["oof_flag"]
    oof_score = cv_positioning["oof_score"]
    fp_mask = oof_score.notna() & oof_flag & (table["y_positioning"] == 0)
    fn_mask = oof_score.notna() & (~oof_flag) & (table["y_positioning"] == 1)
    report["errors_positioning"] = {
        "false_positives": table.loc[fp_mask, "dedup_group_id"].tolist(),
        "false_negatives": table.loc[fn_mask, "dedup_group_id"].tolist(),
    }

    report["cnn_positioning"] = cnn_operating_point(table)

    _write_report(report, HIP_EVAL_REPORT)
    scores = table.copy()
    scores["oof_score_positioning"] = cv_positioning["oof_score"]
    scores["oof_flag_positioning"] = cv_positioning["oof_flag"]
    scores["oof_score_roi"] = cv_roi["oof_score"]
    scores["oof_flag_roi"] = cv_roi["oof_flag"]
    scores_csv = HIP_EVAL_REPORT.with_suffix(".csv")
    scores.to_csv(scores_csv, index=False)
    Path(HIP_EVAL_REPORT.with_suffix(".json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"отчёт: {HIP_EVAL_REPORT}; скоры: {scores_csv}", flush=True)


if __name__ == "__main__":
    main()
