"""Шаг 3: дедупликация по пиксельному содержимому.

Хэш считается от декодированного массива пикселей, а не от байтов файла:
SOP Instance UID у копий разный, байты отличаются, пиксели идентичны.

Дедупликация — это АТРИБУТ манифеста, а не удаление файлов: в инференсе
каждый файл обязан получить свою строку в выходе.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def pixel_hash(pixels: np.ndarray) -> str:
    """SHA-256 от пиксельного массива (форма + dtype + байты)."""
    array = np.ascontiguousarray(pixels)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def assign_dedup_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """Проставить dedup_group_id / dedup_group_size / is_group_representative.

    Идентификатор группы детерминирован: группы нумеруются в порядке
    отсортированного хэша, представитель — файл с лексикографически
    минимальным относительным путём.
    """
    manifest = manifest.copy()
    manifest["dedup_group_id"] = pd.NA
    manifest["dedup_group_size"] = pd.NA
    manifest["is_group_representative"] = False

    readable = manifest["pixel_hash"].notna()
    ordered_hashes = sorted(manifest.loc[readable, "pixel_hash"].unique())
    hash_to_id = {h: f"g{idx:04d}" for idx, h in enumerate(ordered_hashes)}

    manifest.loc[readable, "dedup_group_id"] = manifest.loc[readable, "pixel_hash"].map(hash_to_id)
    sizes = manifest.loc[readable, "dedup_group_id"].value_counts()
    manifest.loc[readable, "dedup_group_size"] = manifest.loc[readable, "dedup_group_id"].map(sizes)

    representatives = (
        manifest.loc[readable]
        .sort_values(["dedup_group_id", "relative_path"])
        .groupby("dedup_group_id", sort=False)
        .head(1)
        .index
    )
    manifest.loc[representatives, "is_group_representative"] = True
    return manifest
