"""Этап 4, шаг 9: чекер укладки/ротации бедра `hip_positioning`.

Единственный чекер этого этапа (`hip_roi` — этап 5, не здесь). Форма выхода —
та же, что у чекеров позвоночника (`spine_checkers.CheckerResult`): монотонный
скор (больше = хуже), флаг по порогу из config, статус ok/not_evaluated с
причиной, именованные сигналы для объяснимости.

Модуль ориентиров бедра (`columba.hip_landmarks`, ключевые точки/сигналы)
разрабатывался параллельно — импортируется ЛЕНИВО (внутри функций),
чтобы этот модуль и его тесты (кроме реального прогона `run_hip_checkers`)
работали и без него. Контракт модуля ориентиров:

    class HipLandmarks: status, reason, side, head_center_xy, shaft_points_xy,
        lesser_trochanter_xy, greater_trochanter_xy, to_dict()
    def detect_hip_landmarks(pixels_normalized, side=None) -> HipLandmarks
    def compute_hip_signals(pixels_normalized, landmarks) -> dict[str, float]
    HIP_SIGNAL_KEYS = ("shaft_tilt_deg", "lesser_trochanter_prominence",
        "lesser_trochanter_corridor_dev", "neck_shaft_angle_deg", "head_offset_mm")

Калибровка на реальных данных (`hip_eval.py`, `stages/stage_4.md`) показала,
что геометрические сигналы сами по себе слабы: вложенная групповая CV
(логрегрессия по всем 5 сигналам, 4 фолда) даёт `hip_positioning` OOF AUC
0.564 [0.452; 0.684] — доверительный интервал пересекает 0.5, не отличимо
от случайного с приемлемой уверенностью. CNN-ветка (шаг 6, обучена в Colab)
даёт OOF AUC 0.765 на том же train — заметно и статистически значимо
лучше. Поэтому CNN — основной источник score/flag, когда веса подключены
(`cfg.HIP_POSITIONING_FLAG_THRESHOLD`/`PROB_SCALE` откалиброваны на
вероятности CNN, не на геометрии); геометрические сигналы остаются в
`signals` только для объяснимости. Без CNN-предиктора (веса не скачаны) —
честный `not_evaluated`, а не флагование по геометрии-которая-не-лучше-
случайного: агрегатор (`aggregate.py`) в этом случае откатывается на приор
`HIP_PRIOR_QUALITY_PROB`, тот же путь, что уже был до этапа 4.
"""

from __future__ import annotations

from . import config as cfg
from .spine_checkers import CheckerResult
from .spine_keypoints import STATUS_NOT_EVALUATED, STATUS_OK

HIP_CHECKER_KEYS: tuple[str, ...] = ("hip_positioning", "hip_roi")


def _not_evaluated(reason: str, side_signals: dict | None = None) -> CheckerResult:
    return CheckerResult(
        checker="hip_positioning",
        status=STATUS_NOT_EVALUATED,
        score=float("nan"),
        flag=False,
        reason=reason,
        signals=dict(side_signals) if side_signals else {},
    )


def _side_confidence_signals(side_confident: bool | None) -> dict:
    """Признак уверенности стороны бедра (этап 9, п. 9.5) — только для
    объяснимости, никогда не меняет флаг/score.

    `side_confident` — из `regions.hip_side(...).confident`
    (`|score| >= cfg.HIP_SIDE_MIN_ABS_SCORE`), приходит из вызывающего кода
    (`inference._attach_hip_checkers`, колонка манифеста `hip_side_confident`)
    опциональным параметром: `None`, если вызывающий код ещё не прокидывает
    его (обратная совместимость, не требует правки `inference.py` разом с
    этим модулем). При `False` — сторона, которой зеркалируется вход CNN
    (`landmarks.side`, передаётся в `cnn_predictor.predict_pixels`), могла
    быть определена по слабому/шумовому сигналу; на train (`inventory`,
    331 кадр бедра) таких кадров 0 — риск на этой выборке не наблюдаем, но
    не исключён на закрытом тесте. Решение (см. stages/stage_9.md, раздел
    9.5): НЕ переводить такие кадры в `not_evaluated` (просадка recall при
    полностью неверной оценке доли low-confidence на закрытом тесте) —
    только пишем причину в `signals`, флаг остаётся как есть.
    """
    if side_confident is None:
        return {}
    signals = {"hip_side_confident": bool(side_confident)}
    if not side_confident:
        signals["hip_side_low_confidence_reason"] = (
            "сторона бедра определена с низкой уверенностью "
            "(|hip_side_score| < cfg.HIP_SIDE_MIN_ABS_SCORE) — вход CNN мог быть "
            "зеркально отражён неверно; флаг НЕ подавлен, только объяснимость"
        )
    return signals


def check_hip_positioning(
    pixels_normalized, landmarks, signals: dict, cnn_predictor=None, side_confident: bool | None = None
) -> CheckerResult:
    """Чекер `hip_positioning`: score/flag — от CNN, геометрия — для объяснимости.

    `landmarks.status != ok` -> `not_evaluated` с причиной (никогда не
    молчаливый 0) — ориентиры нужны для сигналов-объяснений, даже когда
    решение принимает CNN. Затем: `cnn_predictor is None` (веса не
    подключены) или его инференс не удался (честная деградация внутри
    `HipPositioningPredictor.predict_pixels`, `None` без исключений) ->
    тоже `not_evaluated` — геометрический скор сам по себе статистически не
    отличим от случайного (см. докстринг модуля), флагать по нему было бы
    менее честно, чем откатиться на приор агрегатора. Когда CNN дала
    вероятность — она и есть `score` (уже в [0, 1], больше = хуже,
    конвенция 1.4), `flag = score >= cfg.HIP_POSITIONING_FLAG_THRESHOLD`.
    Все именованные геометрические сигналы модуля A, ориентиры (`to_dict()`)
    и `cnn_probability` попадают в `signals` результата.

    `side_confident` (этап 9, п. 9.5) — опциональный признак уверенности
    определения стороны (`regions.hip_side(...).confident`), влияющий ТОЛЬКО
    на `signals` (объяснимость), не на `score`/`flag` — см.
    `_side_confidence_signals`.
    """
    side_signals = _side_confidence_signals(side_confident)

    if pixels_normalized.ndim != 2 or min(pixels_normalized.shape) < 8:
        return _not_evaluated("кадр вырожден", side_signals)
    if landmarks.status != STATUS_OK:
        return _not_evaluated(landmarks.reason or "ориентиры бедра не найдены", side_signals)

    out_signals: dict = dict(signals)
    out_signals.update(landmarks.to_dict())
    out_signals.update(side_signals)

    if cnn_predictor is None:
        return _not_evaluated(
            "CNN-предиктор не подключён (веса не скачаны); геометрический сигнал "
            "статистически не отличим от случайного (nested CV AUC 0.564 [0.452; 0.684], "
            "stage_4_decisions.md) — не используется для флага самостоятельно",
            side_signals,
        )

    cnn_probability = cnn_predictor.predict_pixels(pixels_normalized, landmarks.side)
    if cnn_probability is None:
        return _not_evaluated("CNN-инференс не удался (препроцессинг кропа)", side_signals)

    out_signals["cnn_probability"] = round(float(cnn_probability), 4)

    return CheckerResult(
        checker="hip_positioning",
        status=STATUS_OK,
        score=float(cnn_probability),
        flag=bool(cnn_probability >= cfg.HIP_POSITIONING_FLAG_THRESHOLD),
        signals=out_signals,
    )


def check_hip_roi(pixels_normalized, landmarks, roi_signals: dict) -> CheckerResult:
    """Чекер `hip_roi` (этап 5): отступы поля сканирования от ориентиров.

    Решающий сигнал — `roi_signals["field_height_deficit"]` (нормированный
    дефицит высоты кадра, `hip_landmarks.compute_hip_roi_signals`, по
    образцу `spine_checkers.check_spine_positioning`'s `crest_deficit`):
    считается по одним ИСХОДНЫМ пикселям кадра, ориентиры не нужны — в
    отличие от `check_hip_positioning`, здесь `landmarks.status != ok` НЕ
    даёт `not_evaluated`, margin-сигналы от ориентиров (`top_margin_px` и
    т.п.) в этом случае просто остаются NaN, только для объяснимости, во
    флаг не входят (см. `stages/stage_5.md`: 6 train-позитивов —
    слишком мало для надёжного многомерного правила).
    """
    if pixels_normalized.ndim != 2 or min(pixels_normalized.shape) < 8:
        return CheckerResult(
            checker="hip_roi", status=STATUS_NOT_EVALUATED, score=float("nan"), flag=False, reason="кадр вырожден"
        )

    score = roi_signals.get("field_height_deficit", float("nan"))
    if score != score:  # NaN
        return CheckerResult(
            checker="hip_roi",
            status=STATUS_NOT_EVALUATED,
            score=float("nan"),
            flag=False,
            reason="field_height_deficit не вычислен",
        )
    score = float(score)

    out_signals: dict = dict(roi_signals)
    if landmarks.status == STATUS_OK:
        out_signals.update(landmarks.to_dict())

    return CheckerResult(
        checker="hip_roi",
        status=STATUS_OK,
        score=score,
        flag=bool(score >= cfg.HIP_ROI_FLAG_THRESHOLD),
        signals=out_signals,
    )


def run_hip_checkers(
    pixels_normalized,
    side: str | None = None,
    cnn_predictor=None,
    side_confident: bool | None = None,
) -> dict[str, CheckerResult]:
    """Все чекеры бедра на одном кадре (`hip_positioning`, `hip_roi`).

    Импортирует `columba.hip_landmarks` лениво: если модуль ещё не
    существует (`ImportError`) или ещё не готов, исключение всплывает к
    вызывающему коду (`inference._attach_hip_checkers`), который обязан
    честно свести это к `not_evaluated`, без падения всего манифеста.
    `cnn_predictor` — опциональный `hip_cnn.HipPositioningPredictor`
    (`hip_cnn.load_hip_positioning_predictor()` у вызывающего кода); `None`,
    если веса не подключены — `hip_positioning` честно деградирует (см.
    докстринг `check_hip_positioning`), `hip_roi` от CNN не зависит вовсе.

    `side_confident` (этап 9, п. 9.5) — опциональный признак уверенности
    `side` (`regions.hip_side(...).confident`), пробрасывается только в
    `check_hip_positioning` (влияет на `hip_positioning_signals`, не на
    флаг/скор ни одного чекера) — см. `_side_confidence_signals`. Вызывающий
    код (`inference._attach_hip_checkers`) берёт его из колонки манифеста
    `hip_side_confident`; дефолт `None` сохраняет прежнее поведение для
    прямых вызовов без этого признака.
    """
    from .hip_landmarks import compute_hip_roi_signals, compute_hip_signals, detect_hip_landmarks

    landmarks = detect_hip_landmarks(pixels_normalized, side=side)
    signals = compute_hip_signals(pixels_normalized, landmarks)
    roi_signals = compute_hip_roi_signals(pixels_normalized, landmarks)
    return {
        "hip_positioning": check_hip_positioning(
            pixels_normalized, landmarks, signals, cnn_predictor=cnn_predictor, side_confident=side_confident
        ),
        "hip_roi": check_hip_roi(pixels_normalized, landmarks, roi_signals),
    }
