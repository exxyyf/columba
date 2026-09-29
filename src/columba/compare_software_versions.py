"""Этап 1, шаг 2: сравнение версий софта 18.41.005 vs 18.50.082.

Вопрос шага: нужна ли отдельная нормализация или стратификация по версии
софта перед обучением CNN. Ответ по интенсивностям и размерам кадров — «нет»
(см. stages/stage_1.md); здесь воспроизводимо считается вторая
половина аргумента — сравнение долей нарушений с доверительными интервалами
и тестом значимости. Итог — отчёт artifacts/eda/software_versions_report.md.

Запуск: `uv run python -m columba.compare_software_versions`
Требует artifacts/manifest.parquet и artifacts/targets.csv (создаются main.py).

Статистика без внешних зависимостей (scipy в окружении нет):
* 95% ДИ — интервал Уилсона: у него честное покрытие на малых n и он не
  вырождается при k, близких к 0 или n, в отличие от нормального (Вальда);
* значимость — двухпропорционный z-тест с пулированной дисперсией; p-value
  через функцию ошибок (math.erf).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

import pandas as pd

from . import config as cfg
from .inventory import load_manifest

# Квантиль стандартного нормального распределения для 95% двустороннего ДИ.
Z_95 = 1.959964


# --------------------------------------------------------------------------- #
# Статистика
# --------------------------------------------------------------------------- #


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% ДИ Уилсона для биномиальной доли k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_proportion_z_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Двухпропорционный z-тест (пулированная дисперсия): (z, двустороннее p)."""
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return (z, p_value)


# --------------------------------------------------------------------------- #
# Сбор данных
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ShareComparison:
    """Сравнение одной доли между двумя версиями софта."""

    name: str
    k1: int  # позитивы в старшей выборке (18.41.005)
    n1: int
    k2: int  # позитивы в младшей выборке (18.50.082)
    n2: int

    @property
    def ci1(self) -> tuple[float, float]:
        return wilson_interval(self.k1, self.n1)

    @property
    def ci2(self) -> tuple[float, float]:
        return wilson_interval(self.k2, self.n2)

    @property
    def test(self) -> tuple[float, float]:
        return two_proportion_z_test(self.k1, self.n1, self.k2, self.n2)


def compare_versions(manifest: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Собрать все сравнения долей по версиям софта из артефактов этапа 0."""
    version_by_study = manifest.groupby("study_folder")["software_version"].first()
    mixed = int((manifest.groupby("study_folder")["software_version"].nunique() > 1).sum())

    frame = targets[targets["has_target"].astype(bool)].copy()
    frame["software_version"] = frame["study_folder"].map(version_by_study)
    versions = sorted(frame["software_version"].dropna().unique())
    if len(versions) != 2:
        raise ValueError(f"ожидались ровно две версии софта, найдено: {versions}")
    old, new = versions

    def split_counts(subset: pd.DataFrame, positive: pd.Series) -> tuple[int, int, int, int]:
        mask_old = subset["software_version"] == old
        mask_new = subset["software_version"] == new
        return (
            int(positive[mask_old].sum()),
            int(mask_old.sum()),
            int(positive[mask_new].sum()),
            int(mask_new.sum()),
        )

    total = ShareComparison("quality_class (хотя бы одно нарушение)", *split_counts(frame, frame["quality_class"] > 0))

    # Отдельные метки словаря: знаменатель — изображения региона метки.
    labels = []
    for label in cfg.OUTPUT_LABELS:
        subset = frame[frame["region"] == label.region]
        positive = subset[f"label_{label.key}"].fillna(0) > 0
        labels.append(ShareComparison(f"{label.key} («{label.violation_type}», {label.region})", *split_counts(subset, positive)))

    # hip_positioning на уровне исследования: правое и левое бедро одного
    # пациента скоррелированы, честная независимая единица — пациент.
    hips = frame[frame["region"] == cfg.REGION_HIP]
    per_study = hips.groupby(["study_folder", "software_version"])["label_hip_positioning"].max().reset_index()
    hip_study = ShareComparison(
        "hip_positioning, уровень исследования", *split_counts(per_study, per_study["label_hip_positioning"] > 0)
    )
    pair_sums = hips.groupby("study_folder")["label_hip_positioning"].agg(["sum", "count"])
    pairs = pair_sums[pair_sums["count"] == 2]
    hip_pairs = {
        "n_pairs": len(pairs),
        "both_violated": int((pairs["sum"] == 2).sum()),
        "one_violated": int((pairs["sum"] == 1).sum()),
    }

    folds = (
        frame.groupby(["split", "software_version"])["dedup_group_id"].size().unstack(fill_value=0)
        if "split" in frame.columns
        else pd.DataFrame()
    )

    return {
        "versions": (old, new),
        "n_studies": manifest.groupby("software_version")["study_folder"].nunique().to_dict(),
        "n_unique_images": {v: int((frame["software_version"] == v).sum()) for v in versions},
        "mixed_version_studies": mixed,
        "total": total,
        "labels": labels,
        "hip_study": hip_study,
        "hip_pairs": hip_pairs,
        "folds": folds,
    }


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #


def _pct(k: int, n: int) -> str:
    return f"{k / n:.1%}" if n else "—"


def _ci(bounds: tuple[float, float]) -> str:
    return f"[{bounds[0]:.1%}; {bounds[1]:.1%}]"


def _comparison_row(c: ShareComparison) -> str:
    z, p = c.test
    return (
        f"| {c.name} | {c.k1}/{c.n1} = {_pct(c.k1, c.n1)} {_ci(c.ci1)} "
        f"| {c.k2}/{c.n2} = {_pct(c.k2, c.n2)} {_ci(c.ci2)} | {z:.2f} | {p:.3f} |"
    )


def render_report(stats: dict) -> str:
    old, new = stats["versions"]
    total: ShareComparison = stats["total"]
    z_total, p_total = total.test
    labels: list[ShareComparison] = stats["labels"]
    bonferroni = 0.05 / len(labels)

    label_rows = "\n".join(_comparison_row(c) for c in labels)
    hip_pos = next(c for c in labels if c.name.startswith("hip_positioning"))
    z_hip, p_hip = hip_pos.test
    spine_obj = next(c for c in labels if c.name.startswith("spine_objects"))
    _, p_spine_obj = spine_obj.test
    hip_study: ShareComparison = stats["hip_study"]
    z_study, p_study = hip_study.test
    pairs = stats["hip_pairs"]

    folds = stats["folds"]
    fold_rows = "\n".join(
        f"| {split} | " + " | ".join(str(int(folds.loc[split, v])) for v in (old, new)) + " |"
        for split in folds.index
    )

    return f"""# Версии софта {old} vs {new}: доли нарушений, ДИ и значимость

Этап 1 ([stage_1.md](../../stages/stage_1.md)). Вопрос: различаются ли
данные двух версий софта денситометра настолько, что перед обучением CNN
нужна отдельная нормализация или стратификация по версии. По интенсивностям
и размерам кадров различий нет (см.
[stage_1.md](../../stages/stage_1.md), «Версии софта»); здесь —
воспроизводимый расчёт по долям нарушений, породивший вывод «при n={total.n2}
биномиальные ДИ перекрываются — не повод для стратификации».

Отчёт генерируется скриптом: `uv run python -m columba.compare_software_versions`.

## Данные

Источник — артефакты этапа 0: версия софта на исследование из
`manifest.parquet` (тег SoftwareVersions; внутри исследования версии не
смешиваются: {stats["mixed_version_studies"]} смешанных), `quality_class` и
метки словаря на уникальное изображение из `targets.csv` (только изображения
с таргетом).

| | {old} | {new} |
| --- | --- | --- |
| исследований | {stats["n_studies"][old]} | {stats["n_studies"][new]} |
| уникальных изображений с таргетом | {total.n1} | {total.n2} |

## Расчёт

**Доля нарушений** — биномиальная доля k/n, где k — изображения с
`quality_class = 1`.

**95% ДИ Уилсона.** Для доли p̂ = k/n интервал — это центр
(p̂ + z²/2n)/(1 + z²/n) ± z·√(p̂(1−p̂)/n + z²/4n²)/(1 + z²/n), z = {Z_95}.
Выбран вместо нормального приближения (Вальда), потому что честнее ведёт
себя на малых n и при долях, близких к 0 или 1, — а у нас именно малые n
и редкие классы.

**Двухпропорционный z-тест** (пулированная дисперсия):
z = (p̂₁ − p̂₂)/√(p̄(1−p̄)(1/n₁ + 1/n₂)), p̄ — доля в объединённой выборке;
p-value двустороннее. Перекрытие ДИ — наглядный, но грубый критерий (ДИ
могут перекрываться и при значимой разнице), поэтому вывод опирается на тест.

## Результат: общая доля нарушений

| | {old} | {new} | z | p |
| --- | --- | --- | --- | --- |
| {total.name} | {total.k1}/{total.n1} = {_pct(total.k1, total.n1)} {_ci(total.ci1)} | {total.k2}/{total.n2} = {_pct(total.k2, total.n2)} {_ci(total.ci2)} | {z_total:.2f} | {p_total:.3f} |

Интервалы перекрываются широко, p = {p_total:.3f} — разница долей
статистически неотличима от шума выборки.

## По отдельным меткам словаря

Знаменатель каждой метки — изображения её региона (метка позвоночника не
применима к бедру и наоборот).

| метка | {old} | {new} | z | p |
| --- | --- | --- | --- | --- |
{label_rows}

Разница общей доли почти целиком за счёт `hip_positioning`
({hip_pos.k1}/{hip_pos.n1} = {_pct(hip_pos.k1, hip_pos.n1)} против
{hip_pos.k2}/{hip_pos.n2} = {_pct(hip_pos.k2, hip_pos.n2)}) — единственной
метки, чей p = {p_hip:.3f} проходит и номинальный порог 0,05, и порог
Бонферрони для пяти сравнений (0,05/5 = {bonferroni:.3g}). Разбор этого
сигнала:

1. **Уровень изображений завышает значимость.** Правое и левое бедро одного
   пациента сильно скоррелированы: из {pairs["both_violated"] + pairs["one_violated"]}
   исследований с нарушением `hip_positioning` (среди {pairs["n_pairs"]} полных
   пар бёдер) в {pairs["both_violated"]} нарушены оба бедра сразу. Честная
   независимая единица — исследование (пациент), а не снимок.
2. **На уровне исследований сигнал не значим.** Нарушение хотя бы на одном
   бедре: {hip_study.k1}/{hip_study.n1} = {_pct(hip_study.k1, hip_study.n1)}
   {_ci(hip_study.ci1)} против {hip_study.k2}/{hip_study.n2} =
   {_pct(hip_study.k2, hip_study.n2)} {_ci(hip_study.ci2)};
   z = {z_study:.2f}, p = {p_study:.3f}. Пограничное значение, но выше 0,05 —
   ещё до поправки на множественность.
3. **Согласованного «эффекта версии» нет.** `spine_objects` отклоняется в
   противоположную сторону ({_pct(spine_obj.k1, spine_obj.n1)} против
   {_pct(spine_obj.k2, spine_obj.n2)}, p = {p_spine_obj:.3f}, Бонферрони не
   проходит). Разнонаправленные отклонения по меткам — картина различий
   потока пациентов и периода съёмки, а не софта.
4. **Природа различия.** Доля нарушений — свойство укладки и работы
   лаборанта, а не пиксельных данных; даже реальное различие не было бы
   поводом для *нормализации входа*. Для классификатора региона (этап 1)
   частота нарушений и вовсе иррелевантна.

Практический след: `hip_positioning` — пограничный сигнал (p = {p_study:.3f}
на уровне исследований). На этап 6 (пороги, калибровка) стоит проверить
устойчивость метрик этого класса по версиям на валидации; на решения
этапа 1 он не влияет.

## Версии в фолдах сплита

Обе версии представлены в обоих фолдах — сплит этапа 0 менять не нужно:

| фолд | {old} | {new} |
| --- | --- | --- |
{fold_rows}

## Вывод

Отдельная нормализация и стратификация по версии софта **не нужны**:
интенсивности и размеры кадров идентичны, разница долей нарушений не
значима (и не относится к пиксельным данным), обе версии есть в обоих
фолдах. CNN этапа 1 обучается на объединённой выборке без признака версии.

## Не путать с ДИ этапа 6

«Все метрики с 95% ДИ через бутстрэп» из раздела 7 сводного документа — про
отчётность качества *моделей* (этап 6): на ~250 изображениях и 3–4 позитивах
редкого класса точечная метрика без интервала малоинформативна. Это отдельная
задача; здесь ДИ использованы для однократного решения о дизайне датасета.
"""


def main() -> dict:
    manifest = load_manifest(cfg.MANIFEST_PARQUET)
    targets = pd.read_csv(cfg.TARGETS_CSV)
    stats = compare_versions(manifest, targets)
    report_path = Path(cfg.SOFTWARE_VERSIONS_REPORT)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(stats), encoding="utf-8")
    total: ShareComparison = stats["total"]
    z, p = total.test
    print(
        f"{total.k1}/{total.n1} = {total.k1 / total.n1:.1%} vs "
        f"{total.k2}/{total.n2} = {total.k2 / total.n2:.1%}; z = {z:.2f}, p = {p:.3f}; "
        f"отчёт: {report_path}",
        flush=True,
    )
    return stats


if __name__ == "__main__":
    main()
