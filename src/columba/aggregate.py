"""Этап 3: агрегация чекеров позвоночника (и приора бедра) в сабмит.

Модуль не пересчитывает пороги и не трогает пиксели — он только читает уже
посчитанные колонки манифеста (`describe_inputs`/`_attach_spine_checkers`
этапа 2): `region_final`, `{key}_score`/`{key}_flag`/`{key}_status` для трёх
чекеров позвоночника (`spine_axis`, `spine_positioning`, `spine_objects`) и
`read_status`. Обучения здесь нет — вся калибровка (пороги, `scale` сигмоиды,
приор бедра) зафиксирована константами в `config`.

Полярность (конвенция проекта): `quality_class = 1` и `quality_prob` —
вероятность НАРУШЕНИЯ. Флаг каждого чекера уже посчитан с нужной полярностью
и нужным сравнением с порогом (`>` для оси, `>=` для укладки/предметов,
см. `spine_checkers.py`) — агрегатор его не переопределяет, только
транслирует в строку словаря через `config.OUTPUT_LABELS`.

`quality_prob` — временная калибровка до Platt/isotonic этапа 6: для каждого
чекера региона `p = sigmoid((score - threshold) / scale)`, файловый
`quality_prob` = максимум `p` среди чекеров региона со статусом `ok`
(не среднее/сумма — иначе слабые сигналы разбавляли бы явное нарушение).
Скор `spine_axis` уже беззнаковый (`abs(angle)`, см. `spine_checkers.py`),
повторно брать `abs()` не нужно.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .dicom_io import STATUS_SUCCESS
from .spine_keypoints import STATUS_NOT_EVALUATED, STATUS_OK

# Порог флага и scale сигмоиды quality_prob — по ключу OutputLabel (не строка
# словаря): пороги уже существуют в config (этап 2), scale — новые константы
# этапа 3 (обоснование в stages/stage_3.md).
_CHECKER_THRESHOLD_SCALE: dict[str, tuple[float, float]] = {
    "spine_axis": (cfg.SPINE_AXIS_MAX_ANGLE_DEG, cfg.SPINE_AXIS_PROB_SCALE_DEG),
    "spine_positioning": (cfg.SPINE_POSITIONING_FLAG_DEFICIT, cfg.SPINE_POSITIONING_PROB_SCALE),
    "spine_objects": (cfg.SPINE_METAL_MIN_SCORE, cfg.SPINE_OBJECTS_PROB_SCALE),
}

# Этап 4: чекер бедра. `hip_roi` (этап 5) сюда не входит — для него нет
# колонок в манифесте, `_hip_row_prediction` трактует его как not_evaluated.
# Функция, не словарь-константа (в отличие от `_CHECKER_THRESHOLD_SCALE`
# выше): `HIP_POSITIONING_FLAG_THRESHOLD`/`PROB_SCALE` откалиброваны позже
# импорта модуля (`hip_eval`, шаг 7) и меняются тестами через monkeypatch —
# словарь, собранный один раз при импорте, замораживал бы старое значение.
def _hip_checker_threshold_scale(key: str) -> tuple[float, float] | None:
    if key == "hip_positioning":
        return (cfg.HIP_POSITIONING_FLAG_THRESHOLD, cfg.HIP_POSITIONING_PROB_SCALE)
    if key == "hip_roi":
        return (cfg.HIP_ROI_FLAG_THRESHOLD, cfg.HIP_ROI_PROB_SCALE)
    return None


def _sigmoid(x: float) -> float:
    with np.errstate(over="ignore"):
        return float(1.0 / (1.0 + np.exp(-x)))


def _resolve_region(region_final: object) -> str:
    """`region_final` -> регион сабмита, согласованно с `submission.region_to_output`.

    `region_final == unknown` после арбитража этапа 1 не встречается (входное
    условие этапа 3), но если такое случится, трактуем как
    `config.REGION_SUBMISSION_FALLBACK` — той же логикой, что уже применяет
    `build_submission`/`region_to_output` к строке `anatomical_region`.
    """
    if region_final in (cfg.REGION_SPINE, cfg.REGION_HIP):
        return str(region_final)
    return cfg.REGION_SUBMISSION_FALLBACK


def _spine_row_prediction(row) -> dict:
    """Агрегация трёх чекеров позвоночника для одной строки манифеста.

    Порядок типов в `violation_type` — фиксированный порядок
    `OUTPUT_LABELS_BY_REGION[REGION_SPINE]` (== порядок
    `VIOLATION_TYPES_BY_REGION`, проверено тестом посимвольной сверки этапа
    0/1), а не порядок срабатывания чекеров. Каждый `label.key` в этом
    кортеже встречается ровно один раз, поэтому отдельная дедупликация типов
    не нужна — порядок обхода уже даёт список без повторов.
    """
    ordered_types: list[str] = []
    not_evaluated_keys: list[str] = []
    probs: list[float] = []

    for label in cfg.OUTPUT_LABELS_BY_REGION[cfg.REGION_SPINE]:
        status = getattr(row, f"{label.key}_status", None)
        is_ok = status == STATUS_OK

        if status == STATUS_NOT_EVALUATED:
            not_evaluated_keys.append(label.key)

        if not is_ok:
            # Шаг 3: чекер без оценки -> по умолчанию «нарушения нет»,
            # независимо от значения flag в CheckerResult (там оно и так
            # False по построению `_not_evaluated`, но политика должна быть
            # явной). Если политику пересмотрят на `True` (этап 6), чекер
            # без скора обязан вносить в quality_prob не меньше 0.5 — иначе
            # он мог бы дать quality_class=1 при quality_prob < 0.5, что
            # нарушает конвенцию полярности (инвариант 1).
            flag = cfg.NOT_EVALUATED_TREATED_AS_VIOLATION
            if flag:
                probs.append(0.5)
        else:
            flag_value = getattr(row, f"{label.key}_flag", False)
            flag = bool(flag_value) if flag_value is not None and flag_value is not pd.NA else False
            score = getattr(row, f"{label.key}_score", float("nan"))
            if score == score:  # не NaN
                threshold, scale = _CHECKER_THRESHOLD_SCALE[label.key]
                probs.append(_sigmoid((float(score) - threshold) / scale))

        if flag:
            ordered_types.append(label.violation_type)

    quality_class = cfg.QUALITY_CLASS_VIOLATION if ordered_types else cfg.QUALITY_CLASS_OK
    quality_prob = max(probs) if probs else 0.0

    return {
        "quality_class": quality_class,
        "quality_prob": quality_prob,
        "violation_type": cfg.VIOLATION_TYPE_SEPARATOR.join(ordered_types),
        "not_evaluated_checkers": cfg.VIOLATION_TYPE_SEPARATOR.join(not_evaluated_keys),
        "region_used": cfg.REGION_SPINE,
    }


def _hip_row_prediction(row) -> dict:
    """Агрегация чекеров бедра для одной строки манифеста (этапы 4-5).

    По образцу `_spine_row_prediction`. И `hip_positioning` (этап 4, драйвер
    — CNN-вероятность), и `hip_roi` (этап 5, драйвер — `field_height_deficit`)
    имеют колонки и конечные калиброванные пороги — оба проходят обычную
    ветку `sigmoid((score - threshold) / scale)`. Если у файла региона
    `hip` нет вообще ни одного `ok`-чекера (старый манифест без
    `hip_checkers=False`-колонок, ИЛИ оба чекера вернули `not_evaluated` —
    например, `ImportError` модуля `hip_landmarks`, или `hip_positioning`
    без CNN-предиктора) — обратная совместимость с этапом 3: прежний приор
    `HIP_PRIOR_QUALITY_PROB`, `quality_class=0`. Спец-ветка `threshold ==
    inf` в коде ниже — на случай отката порога к заглушке (ручной или при
    регрессии калибровки), а не текущее рабочее состояние.
    """
    ordered_types: list[str] = []
    not_evaluated_keys: list[str] = []
    probs: list[float] = []
    any_ok = False

    for label in cfg.OUTPUT_LABELS_BY_REGION[cfg.REGION_HIP]:
        status = getattr(row, f"{label.key}_status", None)
        is_ok = status == STATUS_OK

        if status == STATUS_NOT_EVALUATED:
            not_evaluated_keys.append(label.key)

        if not is_ok:
            flag = cfg.NOT_EVALUATED_TREATED_AS_VIOLATION
            if flag:
                probs.append(0.5)
        else:
            any_ok = True
            flag_value = getattr(row, f"{label.key}_flag", False)
            flag = bool(flag_value) if flag_value is not None and flag_value is not pd.NA else False
            score = getattr(row, f"{label.key}_score", float("nan"))
            threshold_scale = _hip_checker_threshold_scale(label.key)
            if score == score and threshold_scale is not None:  # не NaN
                threshold, scale = threshold_scale
                if threshold == float("inf"):
                    probs.append(1.0 if flag else cfg.HIP_PRIOR_QUALITY_PROB)
                else:
                    probs.append(_sigmoid((float(score) - threshold) / scale))

        if flag:
            ordered_types.append(label.violation_type)

    if not any_ok:
        # Ни одного откалиброванного ok-чекера бедра — прежний приор,
        # обратная совместимость с этапом 3 (stage_3_decisions.md).
        return {
            "quality_class": cfg.QUALITY_CLASS_OK,
            "quality_prob": cfg.HIP_PRIOR_QUALITY_PROB,
            "violation_type": "",
            "not_evaluated_checkers": cfg.VIOLATION_TYPE_SEPARATOR.join(not_evaluated_keys),
            "region_used": cfg.REGION_HIP,
        }

    quality_class = cfg.QUALITY_CLASS_VIOLATION if ordered_types else cfg.QUALITY_CLASS_OK
    quality_prob = max(probs) if probs else cfg.HIP_PRIOR_QUALITY_PROB

    return {
        "quality_class": quality_class,
        "quality_prob": quality_prob,
        "violation_type": cfg.VIOLATION_TYPE_SEPARATOR.join(ordered_types),
        "not_evaluated_checkers": cfg.VIOLATION_TYPE_SEPARATOR.join(not_evaluated_keys),
        "region_used": cfg.REGION_HIP,
    }


def aggregate_predictions(manifest: pd.DataFrame) -> pd.DataFrame:
    """`Predictor` этапа 3: манифест (`describe_inputs`) -> предсказания.

    Строка на каждый читаемый файл (в т.ч. на каждую копию дубликата —
    предсказание детерминировано по колонкам чекеров, которые у дублей уже
    идентичны благодаря кэшу `_attach_spine_checkers`, поэтому агрегатору не
    нужен собственный кэш по `pixel_hash`/`dedup_group_id`). Нечитаемые файлы
    (`read_status != STATUS_SUCCESS`) не предсказываются — `build_submission`
    сохраняет для них `FAILURE_ROW_QUALITY_CLASS`/`FAILURE_ROW_QUALITY_PROB`
    как есть (эта функция просто не отдаёт для них строку).
    """
    if manifest.empty:
        return pd.DataFrame(
            columns=[
                "file_id",
                "quality_class",
                "quality_prob",
                "violation_type",
                "not_evaluated_checkers",
                "region_used",
            ]
        )

    records = []
    for row in manifest.itertuples():
        read_status = getattr(row, "read_status", STATUS_SUCCESS)
        if read_status != STATUS_SUCCESS:
            continue

        region = _resolve_region(getattr(row, "region_final", cfg.REGION_UNKNOWN))
        if region == cfg.REGION_HIP:
            prediction = _hip_row_prediction(row)
        else:
            prediction = _spine_row_prediction(row)

        records.append({"file_id": row.file_id, **prediction})

    return pd.DataFrame.from_records(records)
