"""Шаг 8: короткий EDA-отчёт этапа 0."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as cfg  # noqa: E402
from .dicom_io import read_dicom  # noqa: E402
from .targets import LABEL_PREFIX  # noqa: E402


def write_eda_report(
    manifest: pd.DataFrame,
    targets: pd.DataFrame,
    split_payload: dict,
    eda_dir: Path | str,
) -> Path:
    eda_dir = Path(eda_dir)
    eda_dir.mkdir(parents=True, exist_ok=True)

    annotated = targets[targets["has_target"].fillna(False)]
    lines: list[str] = [
        "# EDA-отчёт этапа 0",
        "",
        "## Объём выгрузки",
        "",
        f"- файлов в обучающей выгрузке: **{len(manifest)}**",
        f"- читаемых: **{int((manifest['read_status'] == 'Success').sum())}**, "
        f"Failure: **{int((manifest['read_status'] == 'Failure').sum())}**",
        f"- исследований (папок): **{manifest['study_folder'].nunique()}**",
        f"- уникальных изображений (dedup-групп): **{manifest['dedup_group_id'].nunique()}**",
        "",
        "### Размер dedup-групп",
        "",
        _counts_table(manifest.groupby("dedup_group_id")["file_id"].count().value_counts().sort_index(),
                      "файлов в группе", "групп"),
        "",
        "## Версии ПО сканера",
        "",
        _counts_table(manifest["software_version"].value_counts(), "версия", "файлов"),
        "",
        "## Регионы и зоны (уровень уникального изображения)",
        "",
        _counts_table(targets["region"].value_counts(), "регион", "изображений"),
        "",
        _counts_table(targets["zone_key"].value_counts(dropna=False), "зона", "изображений"),
        "",
        "### Ширины кадров",
        "",
        _counts_table(manifest["cols"].value_counts().sort_index(), "ширина, px", "файлов"),
        "",
        "## Классы по регионам",
        "",
        _crosstab(annotated, "region", "quality_class"),
        "",
        "## Частоты типов нарушений",
        "",
        _label_table(annotated),
        "",
        "## Сплит",
        "",
        _split_table(split_payload),
        "",
        "## Маскирование (чёрные прямоугольники)",
        "",
        f"- уникальных изображений с найденным маскированием: "
        f"**{int(manifest.loc[manifest['is_group_representative'].fillna(False), 'has_mask_rect'].sum())}** "
        f"из {manifest['dedup_group_id'].nunique()}",
        "- примеры: `masking_examples.png`",
        "",
        "## Валидация эвристик",
        "",
        _heuristics_section(manifest),
        "",
        "## Геометрия",
        "",
        f"- pixel spacing: {cfg.PIXEL_SPACING_MM_Y} мм по Y, {cfg.PIXEL_SPACING_MM_X} мм по X "
        f"(анизотропия Y/X = {cfg.ANISOTROPY_Y_OVER_X:.3f}); источник — `columba.config`",
        "",
    ]

    _plot_class_distribution(annotated, eda_dir / "class_distribution.png")
    _plot_masking_examples(manifest, eda_dir / "masking_examples.png")
    _plot_region_examples(manifest, eda_dir / "region_examples.png")
    lines.extend(["## Иллюстрации", "", "- `class_distribution.png`", "- `region_examples.png`",
                  "- `masking_examples.png`", ""])

    report = eda_dir / cfg.EDA_REPORT.name
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _heuristics_section(manifest: pd.DataFrame) -> str:
    """Насколько порядок файлов согласуется со стороной, определённой по картинке."""
    hips = manifest[(manifest["region"] == cfg.REGION_HIP) & manifest["is_group_representative"].fillna(False)]
    agree = disagree = 0
    for _, chunk in hips.groupby("study_folder"):
        if len(chunk) != 2:
            continue
        sides = list(chunk.sort_values("file_name")["hip_side"])
        if sides == ["right", "left"]:
            agree += 1
        else:
            disagree += 1
    scores = hips["hip_side_score"].abs()
    return "\n".join(
        [
            f"- сторона бедра определена по содержимому кадра; минимальный |score| = **{scores.min():.3f}** "
            f"при пороге уверенности {cfg.HIP_SIDE_MIN_ABS_SCORE}",
            f"- порядок файлов в папке стороной НЕ управляет: «первый — правое» верно в {agree} "
            f"исследованиях из {agree + disagree}. Определять латеральность по именам файлов нельзя.",
        ]
    )


def _counts_table(series: pd.Series, key_name: str, value_name: str) -> str:
    rows = [f"| {key_name} | {value_name} |", "| --- | --- |"]
    rows += [f"| {index} | {int(value)} |" for index, value in series.items()]
    return "\n".join(rows)


def _crosstab(frame: pd.DataFrame, index: str, column: str) -> str:
    table = pd.crosstab(frame[index], frame[column])
    header = "| " + index + " | " + " | ".join(f"{column}={c}" for c in table.columns) + " | всего |"
    separator = "| " + " --- |" * (len(table.columns) + 2)
    rows = [header, separator]
    for name, row in table.iterrows():
        rows.append(f"| {name} | " + " | ".join(str(int(v)) for v in row) + f" | {int(row.sum())} |")
    return "\n".join(rows)


def _label_table(annotated: pd.DataFrame) -> str:
    rows = [
        "| регион | строка выходного словаря | ключ | позитивов |",
        "| --- | --- | --- | --- |",
    ]
    for label in cfg.OUTPUT_LABELS:
        column = f"{LABEL_PREFIX}{label.key}"
        positives = int((annotated[column] == 1).sum())
        rows.append(
            f"| {cfg.REGION_OUTPUT_NAMES[label.region]} | «{label.violation_type}» | "
            f"`{label.key}` | {positives} |"
        )
    rows.append("")
    rows.append("Уникальных строк словаря — 4: «Некорректная укладка» общая для обоих регионов.")
    return "\n".join(rows)


def _split_table(payload: dict) -> str:
    stats = payload["stats"]
    keys = ["studies", "dedup_groups", "files", "quality_class_positive", *cfg.OUTPUT_LABEL_KEYS]
    rows = ["| метрика | train | val |", "| --- | --- | --- |"]
    for key in keys:
        rows.append(f"| {key} | {stats['train'].get(key, 0)} | {stats['val'].get(key, 0)} |")
    for region in sorted(set(stats["train"]["by_region"]) | set(stats["val"]["by_region"])):
        rows.append(
            f"| region:{region} | {stats['train']['by_region'].get(region, 0)} | "
            f"{stats['val']['by_region'].get(region, 0)} |"
        )
    return "\n".join(rows)


def _plot_class_distribution(annotated: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    table = pd.crosstab(annotated["region"], annotated["quality_class"])
    table.plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set_title("quality_class по регионам")
    axes[0].set_xlabel("регион")
    axes[0].set_ylabel("уникальных изображений")

    counts = {
        f"{label.violation_type}\n({label.region})": int(
            (annotated[f"{LABEL_PREFIX}{label.key}"] == 1).sum()
        )
        for label in cfg.OUTPUT_LABELS
    }
    axes[1].barh(list(counts), list(counts.values()), color="#4c72b0")
    axes[1].set_title("Частоты типов нарушений")
    axes[1].invert_yaxis()
    for spine in ("top", "right"):
        axes[1].spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _plot_masking_examples(manifest: pd.DataFrame, path: Path, limit: int = 6) -> None:
    candidates = manifest[
        manifest["is_group_representative"].fillna(False) & manifest["has_mask_rect"].fillna(False)
    ].sort_values("mask_step_area_px", ascending=False).head(limit)
    _plot_grid(candidates, path, "Примеры маскирования (чёрные прямоугольники)")


def _plot_region_examples(manifest: pd.DataFrame, path: Path) -> None:
    representatives = manifest[manifest["is_group_representative"].fillna(False)]
    picks = []
    for zone in (*cfg.ZONES, None):
        subset = (
            representatives[representatives["zone_key"].isna()]
            if zone is None
            else representatives[representatives["zone_key"] == zone]
        )
        picks.extend(subset.head(2).to_dict("records"))
    _plot_grid(pd.DataFrame(picks), path, "Примеры по зонам")


def _plot_grid(frame: pd.DataFrame, path: Path, title: str) -> None:
    if frame.empty:
        return
    count = len(frame)
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.6 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, record in zip(axes.ravel(), frame.to_dict("records")):
        result = read_dicom(record["abs_path"])
        if not result.ok:
            continue
        axis.imshow(result.pixels, cmap="gray")
        axis.set_title(
            f"{record.get('zone_key') or record.get('region')}\n{record['study_folder'][-8:]}", fontsize=7
        )
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
