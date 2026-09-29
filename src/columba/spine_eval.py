"""Этап 2, шаг 8: калибровка порогов и метрики чекеров позвоночника.

Запуск: uv run python -m columba.spine_eval

Что делает:

* считает скоры всех трёх чекеров на 99 уникальных снимках ПОП;
* основная оценка — вложенная групповая CV по исследованиям внутри
  train-фолда: в каждом фолде на обучающей части заново выбираются
  подбиравшиеся на train параметры (полоса гребней, пороги правила металла —
  по AUC) и порог флага (по максимуму F1), затем применяются к отложенной
  части. Для CNN-ветки предметов — OOF-скоры с переобучением на каждый фолд;
* отдельно — рабочая точка рантайма (скоры и флаги ровно такие, как в
  run_inference, пороги из config) на train и val; val трогается как смоук;
* все метрики — с 95% ДИ кластерного бутстрэпа по исследованиям;
* «есть нарушение позвоночника» и разбивка по версиям софта считаются по
  рантайм-скорам (rule-based предметы); CNN — отдельной строкой;
* проверяет синтетику: повороты 6-10 градусов должны ловиться чекером оси,
  1-2 градуса — нет;
* пишет отчёт `artifacts/spine_checker_report.md` с метриками по фолдам,
  разбивкой по версиям софта для spine_objects и списком FP/FN.

Пороги в config правятся руками по итогам отчёта — они конфигурация, а не
артефакт обучения.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .dicom_io import normalize, read_dicom_strict
from .spine_checkers import (
    check_spine_axis,
    check_spine_objects,
    check_spine_positioning,
    crest_contrast_signal,
    metal_maps,
    metal_rule_score,
)
from .spine_keypoints import STATUS_OK, detect_vertebra_centers
from .spine_synth import rotate_raw_frame

N_CV_FOLDS = cfg.SPINE_EVAL_CV_FOLDS


# --------------------------------------------------------------------------- #
# Скоры чекеров
# --------------------------------------------------------------------------- #


def spine_frame() -> pd.DataFrame:
    from .spine_objects_cnn import objects_dataset_frame

    targets = pd.read_csv(cfg.TARGETS_CSV)
    frame = objects_dataset_frame().rename(columns={"label": "y_objects"})
    extra = targets[["dedup_group_id", "spine_positioning", "spine_axis"]]
    frame = frame.merge(extra, on="dedup_group_id")
    frame["y_axis"] = frame.pop("spine_axis").astype(int)
    frame["y_positioning"] = frame.pop("spine_positioning").astype(int)
    return frame


def crest_col(band_mm: float) -> str:
    return f"crest_band{band_mm:g}"


def metal_col(thin: float, ambient: float) -> str:
    return f"metal_thin{thin:g}_amb{ambient:g}"


def compute_rule_scores(frame: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Рантайм-скоры и флаги чекеров + скоры по сеткам калибровки.

    Рантайм-колонки (`score_*`, `flag_*`) получены теми же функциями и
    порогами config, что и в run_inference. Колонки сеток нужны вложенной CV.
    """
    rows = []
    for row in frame.itertuples():
        result = read_dicom_strict(row.abs_path)
        pixels = normalize(result.pixels, result.tags)
        keypoints = detect_vertebra_centers(pixels)
        axis = check_spine_axis(pixels, keypoints)
        positioning = check_spine_positioning(pixels, keypoints)
        objects = check_spine_objects(pixels)
        entry = {
            "dedup_group_id": row.dedup_group_id,
            "score_axis": axis.score,
            "flag_axis": bool(axis.flag),
            "axis_status": axis.status,
            "score_positioning": positioning.score,
            "flag_positioning": bool(positioning.flag),
            "score_objects_rule": objects.score,
            "flag_objects_rule": bool(objects.flag),
            # Сигналы симметрии укладки (план шага 5): логируются, во флаг не
            # входят; их разделяющая способность фиксируется в отчёте.
            "sig_silhouette_asym_mm": positioning.signals.get("silhouette_asymmetry_mm"),
            "sig_crest_tilt_deg": positioning.signals.get("crest_tilt_deg"),
            "sig_top_margin_deficit": positioning.signals.get("top_margin_deficit"),
        }
        for band in cfg.SPINE_CALIB_CREST_BAND_GRID_MM:
            # дефицит монотонен по контрасту: для AUC и порога берём -контраст
            entry[crest_col(band)] = -crest_contrast_signal(pixels, band_mm=band)
        maps = metal_maps(pixels)
        for thin in cfg.SPINE_CALIB_METAL_THIN_GRID:
            for ambient in cfg.SPINE_CALIB_METAL_AMBIENT_GRID:
                entry[metal_col(thin, ambient)] = metal_rule_score(
                    pixels, thin_contrast=thin, ambient_max=ambient, maps=maps
                )[0]
        rows.append(entry)
    if verbose:
        print(f"скоры чекеров посчитаны: {len(rows)} снимков", flush=True)
    return frame.merge(pd.DataFrame(rows), on="dedup_group_id")


def study_cv_folds(frame: pd.DataFrame) -> pd.Series:
    """Номер CV-фолда для строк train (по исследованиям), NaN для val."""
    train_mask = frame["fold"] == "train"
    studies = sorted(frame.loc[train_mask, "study_folder"].unique())
    fold_of_study = {study: i % N_CV_FOLDS for i, study in enumerate(studies)}
    return frame["study_folder"].map(fold_of_study).where(train_mask)


def objects_cnn_oof_scores(frame: pd.DataFrame, verbose: bool = True) -> pd.Series:
    """OOF-вероятности CNN предметов: групповая CV по исследованиям train.

    Для val-фолда — предсказание финальной модели (веса с диска), это
    единственный «смоук»-прогон val.
    """
    from .dicom_io import normalize as _norm
    from .dicom_io import read_dicom_strict as _read_strict
    from .spine_objects_cnn import (
        SpineObjectsPredictor,
        load_objects_predictor,
        train_objects_cnn,
    )

    scores = pd.Series(np.nan, index=frame.index)
    train_mask = frame["fold"] == "train"
    cv_fold = study_cv_folds(frame)
    for fold_id in range(N_CV_FOLDS):
        holdout = cv_fold == fold_id
        fit_frame = frame[train_mask & ~holdout].rename(columns={"y_objects": "label"})
        if verbose:
            print(f"CV-фолд {fold_id}: обучение на {len(fit_frame)} снимках", flush=True)
        model = train_objects_cnn(fit_frame, weights_path=None, verbose=False)
        predictor = SpineObjectsPredictor(model=model)
        for index in frame.index[train_mask & holdout]:
            result = _read_strict(frame.loc[index, "abs_path"])
            scores[index] = predictor.predict_proba(_norm(result.pixels, result.tags))
    final = load_objects_predictor()
    if final is not None:
        for index in frame.index[~train_mask]:
            result = _read_strict(frame.loc[index, "abs_path"])
            scores[index] = final.predict_proba(_norm(result.pixels, result.tags))
    return scores


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #


def ranking_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Порог с максимальным F1; при равенстве — больший порог (меньше FP)."""
    candidates = np.unique(scores)
    best = {"threshold": float("inf"), "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(((predicted == 1) & (labels == 1)).sum())
        fp = int(((predicted == 1) & (labels == 0)).sum())
        fn = int(((predicted == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"] or (f1 == best["f1"] and threshold > best["threshold"]):
            best = {
                "threshold": float(threshold),
                "f1": round(f1, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
            }
    return best


def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    return flag_metrics(scores >= threshold, labels)


def flag_metrics(predicted: np.ndarray, labels: np.ndarray) -> dict:
    predicted = np.asarray(predicted, dtype=bool)
    labels = np.asarray(labels, dtype=int)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
    }


# --------------------------------------------------------------------------- #
# Синтетика оси
# --------------------------------------------------------------------------- #


def synthetic_axis_check(frame: pd.DataFrame, n_images: int = 12, verbose: bool = True) -> dict:
    """Повороты чистых train-снимков: 6-10° ловятся, 1-2° — нет."""
    rng = np.random.default_rng(cfg.SPINE_SYNTH_SEED)
    clean = frame[(frame["fold"] == "train") & (frame["y_axis"] == 0)]
    picks = clean.sample(min(n_images, len(clean)), random_state=cfg.SPINE_SYNTH_SEED)
    outcomes = {"positive_caught": 0, "positive_total": 0, "negative_flagged": 0, "negative_total": 0}
    for row in picks.itertuples():
        result = read_dicom_strict(row.abs_path)
        pixels = normalize(result.pixels, result.tags)
        base = check_spine_axis(pixels, detect_vertebra_centers(pixels))
        if base.status != STATUS_OK:
            continue
        for angle in cfg.SPINE_SYNTH_AXIS_ANGLES_DEG:
            signed = float(angle if rng.random() < 0.5 else -angle)
            rotated = rotate_raw_frame(pixels, signed)
            check = check_spine_axis(rotated, detect_vertebra_centers(rotated))
            outcomes["positive_total"] += 1
            # поворот добавляется к собственному наклону кадра
            if check.status == STATUS_OK and check.flag:
                outcomes["positive_caught"] += 1
        for angle in cfg.SPINE_SYNTH_AXIS_NEGATIVE_DEG:
            signed = float(angle if rng.random() < 0.5 else -angle)
            rotated = rotate_raw_frame(pixels, signed)
            check = check_spine_axis(rotated, detect_vertebra_centers(rotated))
            outcomes["negative_total"] += 1
            if check.status == STATUS_OK and check.flag:
                outcomes["negative_flagged"] += 1
    if verbose:
        print("синтетика оси:", outcomes, flush=True)
    return outcomes


def synthetic_positioning_check(frame: pd.DataFrame, n_images: int = 12, verbose: bool = True) -> dict:
    """Кропы поля на чистых train-снимках: кроп низа должен поднимать флаг
    (гребни исчезают), кроп верха — сигнал top_margin_deficit (реальных
    позитивов с обрезкой сверху нет, флаг на него не завязан)."""
    from .spine_synth import crop_field

    rng = np.random.default_rng(cfg.SPINE_SYNTH_SEED)
    clean = frame[(frame["fold"] == "train") & (frame["y_positioning"] == 0)]
    picks = clean.sample(min(n_images, len(clean)), random_state=cfg.SPINE_SYNTH_SEED)
    outcomes = {
        "bottom_total": 0,
        "bottom_flagged": 0,
        "top_total": 0,
        "top_deficit_raised": 0,
    }
    for row in picks.itertuples():
        result = read_dicom_strict(row.abs_path)
        pixels = normalize(result.pixels, result.tags)
        base = check_spine_positioning(pixels, detect_vertebra_centers(pixels))
        if base.status != STATUS_OK or base.flag:
            continue  # интересуют только чистые исходники

        cropped = crop_field(pixels, rng, side="bottom")
        check = check_spine_positioning(cropped, detect_vertebra_centers(cropped))
        outcomes["bottom_total"] += 1
        if check.status == STATUS_OK and check.flag:
            outcomes["bottom_flagged"] += 1

        cropped = crop_field(pixels, rng, side="top")
        check = check_spine_positioning(cropped, detect_vertebra_centers(cropped))
        outcomes["top_total"] += 1
        base_deficit = base.signals.get("top_margin_deficit") or 0.0
        crop_deficit = check.signals.get("top_margin_deficit")
        if check.status == STATUS_OK and crop_deficit is not None and crop_deficit > base_deficit:
            outcomes["top_deficit_raised"] += 1
    if verbose:
        print("синтетика укладки:", outcomes, flush=True)
    return outcomes


def positioning_signal_aucs(frame: pd.DataFrame) -> dict:
    """Разделяющая способность сигналов-кандидатов укладки на train.

    Сигналы логируются чекером, во флаг не входят; |crest_tilt| дополнительно
    сверяется с меткой оси — гипотеза этапа 6 о перекосе таза у FN оси
    (открытый вопрос 5 decisions). NaN-значения ранжируются как «не
    подозрительно» (_ranked), их доля фиксируется.
    """
    train = frame[frame["fold"] == "train"]
    setups = [
        ("silhouette_asymmetry (abs) vs укладка", train["sig_silhouette_asym_mm"].abs(), "y_positioning"),
        ("crest_tilt (abs) vs укладка", train["sig_crest_tilt_deg"].abs(), "y_positioning"),
        ("top_margin_deficit vs укладка", train["sig_top_margin_deficit"], "y_positioning"),
        ("crest_tilt (abs) vs ось", train["sig_crest_tilt_deg"].abs(), "y_axis"),
    ]
    result = {}
    for name, scores, label_col in setups:
        values = scores.values.astype(float)
        result[name] = {
            "auc_train": round(ranking_auc(_ranked(values), train[label_col].values.astype(int)), 3),
            "nan_fraction": round(float(np.isnan(values).mean()), 3),
        }
    return result


# --------------------------------------------------------------------------- #
# Вложенная CV и бутстрэп
# --------------------------------------------------------------------------- #


def _ranked(scores: np.ndarray) -> np.ndarray:
    """NaN-скор (not_evaluated) ранжируется как наименее подозрительный —
    так же, как его флаг (False) в рантайме."""
    scores = np.asarray(scores, dtype=float)
    return np.where(np.isnan(scores), -1e9, scores)


def nested_cv(frame: pd.DataFrame, candidates: dict[str, str], label_col: str) -> dict:
    """Вложенная групповая CV по исследованиям внутри train-фолда.

    `candidates` — {описание параметров: колонка скоров}. В каждом фолде на
    обучающей части выбираются параметры (максимум AUC, при равенстве — первый
    в порядке сетки) и порог флага (максимум F1), затем применяются к
    отложенной части. Возвращает OOF-скоры, OOF-флаги и выбор по фолдам.
    """
    cv_fold = study_cv_folds(frame)
    oof_score = pd.Series(np.nan, index=frame.index)
    oof_flag = pd.Series(False, index=frame.index)
    folds = []
    for fold_id in range(N_CV_FOLDS):
        fit = frame[cv_fold.notna() & (cv_fold != fold_id)]
        hold = frame[cv_fold == fold_id]
        labels = fit[label_col].values.astype(int)
        aucs = {name: ranking_auc(_ranked(fit[col].values), labels) for name, col in candidates.items()}
        chosen = max(aucs, key=lambda name: aucs[name])
        column = candidates[chosen]
        threshold = best_f1_threshold(_ranked(fit[column].values), labels)["threshold"]
        oof_score[hold.index] = hold[column].values
        oof_flag[hold.index] = _ranked(hold[column].values) >= threshold
        folds.append(
            {
                "fold": fold_id,
                "params": chosen,
                "fit_auc": round(aucs[chosen], 3),
                "threshold": round(float(threshold), 4),
                "holdout_positives": int(hold[label_col].sum()),
            }
        )
    return {"score": oof_score, "flag": oof_flag, "folds": folds}


def cluster_bootstrap_ci(frame: pd.DataFrame, statistic, n: int = cfg.SPINE_EVAL_BOOTSTRAP_N) -> tuple:
    """95% ДИ статистики кластерным бутстрэпом по исследованиям.

    Реплики, где статистика не определена (нет позитивов), отбрасываются;
    если таких больше половины — ДИ не определён (NaN).
    """
    frame = frame.reset_index(drop=True)
    groups = list(frame.groupby("study_folder").indices.values())
    rng = np.random.default_rng(cfg.SEED)
    values = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        value = statistic(frame.iloc[np.concatenate([groups[i] for i in pick])])
        if value == value:
            values.append(value)
    if len(values) < n / 2:
        return (float("nan"), float("nan"))
    low, high = np.percentile(values, [2.5, 97.5])
    return (round(float(low), 3), round(float(high), 3))


def _auc_stat(score_col: str, label_col: str):
    return lambda df: ranking_auc(_ranked(df[score_col].values), df[label_col].values.astype(int))


def _f1_stat(flag_col: str, label_col: str):
    def statistic(df: pd.DataFrame) -> float:
        labels = df[label_col].values.astype(int)
        if not labels.sum():
            return float("nan")
        return flag_metrics(df[flag_col].values, labels)["f1"]

    return statistic


def _estimate(frame: pd.DataFrame, statistic) -> dict:
    point = statistic(frame)
    return {"value": round(point, 3) if point == point else float("nan"), "ci95": cluster_bootstrap_ci(frame, statistic)}


def _fold_summary(fold: pd.DataFrame, score_col: str, flag_col: str, label_col: str) -> dict:
    labels = fold[label_col].values.astype(int)
    return {
        "n": len(fold),
        "positives": int(labels.sum()),
        **flag_metrics(fold[flag_col].values, labels),  # его "f1" перекрывается оценкой с ДИ ниже
        "auc": _estimate(fold, _auc_stat(score_col, label_col)),
        "f1": _estimate(fold, _f1_stat(flag_col, label_col)),
    }


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #

RUNTIME_SETUPS = [
    # (имя, скор, флаг, метка, порог из config для нормировки «есть нарушение»)
    ("spine_axis", "score_axis", "flag_axis", "y_axis", cfg.SPINE_AXIS_MAX_ANGLE_DEG),
    ("spine_positioning", "score_positioning", "flag_positioning", "y_positioning", cfg.SPINE_POSITIONING_FLAG_DEFICIT),
    ("spine_objects (rule)", "score_objects_rule", "flag_objects_rule", "y_objects", cfg.SPINE_METAL_MIN_SCORE),
]


def run_eval(report_path: Path | str = cfg.SPINE_CHECKER_REPORT, verbose: bool = True) -> dict:
    frame = spine_frame()
    frame = compute_rule_scores(frame, verbose=verbose)
    frame["score_objects_cnn"] = objects_cnn_oof_scores(frame, verbose=verbose)
    frame = frame.reset_index(drop=True)
    frame["y_any"] = ((frame["y_axis"] + frame["y_positioning"] + frame["y_objects"]) > 0).astype(int)
    train_mask = frame["fold"] == "train"
    folds = [("train", frame[train_mask]), ("val", frame[~train_mask])]
    report: dict = {"cv": {}, "runtime": {}}

    # 1. Вложенная CV на train: честная оценка конструкции + выбор параметров.
    cv_setups = {
        "spine_axis": ({"фиксированная конструкция": "score_axis"}, "y_axis"),
        "spine_positioning": (
            {f"полоса {b:g} мм": crest_col(b) for b in cfg.SPINE_CALIB_CREST_BAND_GRID_MM},
            "y_positioning",
        ),
        "spine_objects (rule)": (
            {
                f"thin {t:g} / ambient {a:g}": metal_col(t, a)
                for t in cfg.SPINE_CALIB_METAL_THIN_GRID
                for a in cfg.SPINE_CALIB_METAL_AMBIENT_GRID
            },
            "y_objects",
        ),
        "spine_objects (cnn)": ({"CNN, переобучение на фолд": "score_objects_cnn"}, "y_objects"),
    }
    for name, (candidates, label_col) in cv_setups.items():
        result = nested_cv(frame, candidates, label_col)
        train = frame[train_mask].assign(_oof_score=result["score"], _oof_flag=result["flag"])
        report["cv"][name] = {
            "folds": result["folds"],
            **_fold_summary(train, "_oof_score", "_oof_flag", label_col),
        }
        frame[f"oof_score_{name}"] = result["score"]
        frame[f"oof_flag_{name}"] = result["flag"]

    # 2. Рабочая точка рантайма: флаги ровно как в run_inference.
    for name, score_col, flag_col, label_col, _ in RUNTIME_SETUPS:
        entry = {fold_name: _fold_summary(fold, score_col, flag_col, label_col) for fold_name, fold in folds}
        entry["train_best_threshold"] = best_f1_threshold(
            _ranked(frame.loc[train_mask, score_col].values), frame.loc[train_mask, label_col].values.astype(int)
        )
        report["runtime"][name] = entry

    # CNN предметов: не в рантайме; порог — максимум F1 по OOF train, val — финальная модель.
    cnn_threshold = best_f1_threshold(
        _ranked(frame.loc[train_mask, "score_objects_cnn"].values),
        frame.loc[train_mask, "y_objects"].values.astype(int),
    )
    frame["flag_objects_cnn"] = _ranked(frame["score_objects_cnn"].values) >= cnn_threshold["threshold"]
    report["cnn_objects"] = {
        "threshold": cnn_threshold,
        **{
            fold_name: _fold_summary(fold, "score_objects_cnn", "flag_objects_cnn", "y_objects")
            for fold_name, fold in [("train (OOF)", frame[train_mask]), ("val", frame[~train_mask])]
        },
    }

    # 3. «Есть нарушение позвоночника» (предвестник quality_class): OR рантайм-флагов;
    # скор — максимум скоров, нормированных на пороги config.
    def normalized(score_col: str, threshold: float) -> np.ndarray:
        return np.nan_to_num(frame[score_col].values.astype(float) / max(threshold, 1e-9), nan=0.0)

    runtime_norm = [normalized(s, t) for _, s, _, _, t in RUNTIME_SETUPS]
    frame["score_any"] = np.max(runtime_norm, axis=0)
    frame["flag_any"] = frame[[f for _, _, f, _, _ in RUNTIME_SETUPS]].any(axis=1)
    frame["score_any_cnn"] = np.max(
        runtime_norm[:2] + [normalized("score_objects_cnn", cnn_threshold["threshold"])], axis=0
    )
    frame["flag_any_cnn"] = frame[["flag_axis", "flag_positioning", "flag_objects_cnn"]].any(axis=1)
    folds = [("train", frame[train_mask]), ("val", frame[~train_mask])]  # срезы с новыми колонками
    report["any_violation"] = {
        "runtime (rule)": {
            fold_name: _fold_summary(fold, "score_any", "flag_any", "y_any") for fold_name, fold in folds
        },
        "вариант с CNN предметов": {
            fold_name: _fold_summary(fold, "score_any_cnn", "flag_any_cnn", "y_any") for fold_name, fold in folds
        },
    }

    # 4. Предметы по версиям софта (прокси-риск из этапа 1), весь датасет.
    report["objects_by_software"] = {}
    for version, chunk in frame.groupby("software_version"):
        labels = chunk["y_objects"].values.astype(int)
        report["objects_by_software"][str(version)] = {
            "n": len(chunk),
            "positives": int(labels.sum()),
            "auc_rule": round(ranking_auc(_ranked(chunk["score_objects_rule"].values), labels), 3),
            "rule_runtime": flag_metrics(chunk["flag_objects_rule"].values, labels),
            "auc_cnn": round(ranking_auc(_ranked(chunk["score_objects_cnn"].values), labels), 3),
        }

    report["synthetic_axis"] = synthetic_axis_check(frame, verbose=verbose)
    report["synthetic_positioning"] = synthetic_positioning_check(frame, verbose=verbose)
    report["positioning_signals"] = positioning_signal_aucs(frame)

    # 5. FP/FN рабочей точки рантайма — заготовка для просмотра глазами и этапа 6.
    report["errors"] = {}
    error_setups = [(n, f, l) for n, _, f, l, _ in RUNTIME_SETUPS] + [
        ("spine_objects (cnn)", "flag_objects_cnn", "y_objects")
    ]
    for name, flag_col, label_col in error_setups:
        predicted = frame[flag_col].values.astype(bool)
        labels = frame[label_col].values.astype(int)
        report["errors"][name] = {
            "false_positives": frame.loc[predicted & (labels == 0), "dedup_group_id"].tolist(),
            "false_negatives": frame.loc[~predicted & (labels == 1), "dedup_group_id"].tolist(),
        }

    _write_report_md(report, Path(report_path))
    scores_csv = Path(report_path).with_suffix(".csv")
    grid_cols = [c for c in frame.columns if c.startswith(("crest_band", "metal_thin"))]
    frame.drop(columns=["abs_path", *grid_cols]).to_csv(scores_csv, index=False)
    Path(report_path).with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if verbose:
        print(f"отчёт: {report_path}; скоры: {scores_csv}", flush=True)
    return report


def _fmt(estimate: dict) -> str:
    value, (low, high) = estimate["value"], estimate["ci95"]
    if low != low:
        return f"{value} [ДИ н/о]"
    return f"{value} [{low}; {high}]"


def _summary_row(label: str, entry: dict) -> str:
    return (
        f"| {label} | {entry['n']}/{entry['positives']} | {_fmt(entry['auc'])} | {_fmt(entry['f1'])} | "
        f"{entry['precision']}/{entry['recall']} | {entry['tp']}/{entry['fp']}/{entry['fn']} |"
    )


SUMMARY_HEADER = [
    "| Срез | n/позитивов | AUC [95% ДИ] | F1 [95% ДИ] | P/R | TP/FP/FN |",
    "|---|---|---|---|---|---|",
]


def _write_report_md(report: dict, path: Path) -> None:
    lines = [
        "# Отчёт чекеров позвоночника (этап 2, шаг 8)",
        "",
        "Генерируется `uv run python -m columba.spine_eval` (машиночитаемая копия —",
        "`.json` рядом). 95% ДИ — кластерный бутстрэп по исследованиям",
        f"({cfg.SPINE_EVAL_BOOTSTRAP_N} реплик); «ДИ н/о» — в большинстве реплик нет позитивов.",
        "",
        "## 1. Вложенная CV на train (основная оценка)",
        "",
        f"Групповая {N_CV_FOLDS}-фолдовая CV по исследованиям внутри train. В каждом фолде",
        "параметры, подбиравшиеся на train (полоса гребней, пороги правила металла),",
        "выбираются по AUC на обучающей части, порог флага — по максимуму F1; метрики —",
        "на отложенных частях. AUC — по объединённым OOF-скорам (при разных параметрах",
        "по фолдам шкалы скоров слегка различаются). Константы конструкций, выбранные",
        "при просмотре train глазами (полоса «тонкости», сглаживания midline и т. п.),",
        "здесь не перебираются — эта часть оптимизма остаётся.",
        "",
        *SUMMARY_HEADER,
    ]
    for name, entry in report["cv"].items():
        lines.append(_summary_row(name, entry))
    lines += ["", "Выбор по фолдам:", ""]
    for name, entry in report["cv"].items():
        chosen = "; ".join(
            f"ф{f['fold']}: {f['params']}, порог {f['threshold']:.3g} (поз. в отложенной {f['holdout_positives']})"
            for f in entry["folds"]
        )
        lines.append(f"- {name}: {chosen}")

    lines += [
        "",
        "## 2. Рабочая точка рантайма (пороги config)",
        "",
        "Скоры и флаги получены теми же функциями, что в `run_inference`.",
        "Train здесь in-sample (пороги выбирались на нём), val — смоук.",
        "",
        *SUMMARY_HEADER,
    ]
    for name, entry in report["runtime"].items():
        for fold_name in ("train", "val"):
            lines.append(_summary_row(f"{name} — {fold_name}", entry[fold_name]))
    lines += ["", "Порог max-F1 на всём train (справочно, против config):", ""]
    for name, entry in report["runtime"].items():
        best = entry["train_best_threshold"]
        lines.append(f"- {name}: {best['threshold']:.3g} (F1 {best['f1']})")

    cnn = report["cnn_objects"]
    lines += [
        "",
        "## 3. CNN-ветка предметов (не в рантайме)",
        "",
        f"Порог {cnn['threshold']['threshold']:.3g} — максимум F1 по OOF train; val — финальная модель.",
        "",
        *SUMMARY_HEADER,
        _summary_row("train (OOF)", cnn["train (OOF)"]),
        _summary_row("val", cnn["val"]),
        "",
        "## 4. «Есть нарушение позвоночника» (OR флагов)",
        "",
        *SUMMARY_HEADER,
    ]
    for variant, entry in report["any_violation"].items():
        for fold_name in ("train", "val"):
            lines.append(_summary_row(f"{variant} — {fold_name}", entry[fold_name]))

    lines += ["", "## 5. spine_objects по версиям софта (весь датасет)", ""]
    for version, entry in report["objects_by_software"].items():
        rule = entry["rule_runtime"]
        lines.append(
            f"- {version}: n={entry['n']}, позитивов {entry['positives']}; правило: AUC {entry['auc_rule']}, "
            f"TP/FP/FN {rule['tp']}/{rule['fp']}/{rule['fn']}; CNN: AUC {entry['auc_cnn']}"
        )
    synth = report["synthetic_axis"]
    synth_pos = report["synthetic_positioning"]
    lines += [
        "",
        "## 6. Синтетика оси и укладки",
        "",
        f"Повороты 6-10°: поймано {synth['positive_caught']}/{synth['positive_total']}; "
        f"повороты 1-2°: ложных флагов {synth['negative_flagged']}/{synth['negative_total']}.",
        "",
        f"Кропы низа (гребни срезаны): флаг укладки на "
        f"{synth_pos['bottom_flagged']}/{synth_pos['bottom_total']}; кропы верха: сигнал "
        f"top_margin_deficit вырос на {synth_pos['top_deficit_raised']}/{synth_pos['top_total']} "
        "(во флаг не входит — реальных позитивов с обрезкой сверху в разметке нет).",
        "",
        "## 7. Сигналы-кандидаты укладки (логируются, во флаг не входят)",
        "",
        "AUC на train; NaN (сигнал не посчитался) ранжируется как «не подозрительно».",
        "",
    ]
    for name, entry in report["positioning_signals"].items():
        lines.append(f"- {name}: AUC {entry['auc_train']}, доля NaN {entry['nan_fraction']}")
    lines += [
        "",
        "## 8. Ошибки рабочей точки (весь датасет)",
        "",
    ]
    for name, entry in report["errors"].items():
        lines.append(f"- {name}: FP {entry['false_positives'] or '—'}; FN {entry['false_negatives'] or '—'}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    payload = run_eval()
    print(json.dumps(payload["cv"], ensure_ascii=False, indent=2, default=str))
