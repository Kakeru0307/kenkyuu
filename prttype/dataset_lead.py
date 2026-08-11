"""リードギター technique_type one-hot マップ生成ユーティリティ。

dataset_drum.py の beat_type 版に相当する。
学習データ生成（generate_lead_pairs.py）と推論（inference.py）で共用する。
"""

from __future__ import annotations

import numpy as np

from makeData.constants import N_TECHNIQUE_TYPES


def make_technique_onehot_map(
    technique_id: int,
    *,
    height: int,
    width: int,
    n_types: int = N_TECHNIQUE_TYPES,
) -> np.ndarray:
    """technique_type を (N, H, W) の one-hot 定数マップにする。"""
    out = np.zeros((n_types, height, width), dtype=np.float32)
    idx = int(technique_id)
    if not (0 <= idx < n_types):
        raise ValueError(f"technique_id out of range: {technique_id}")
    out[idx] = 1.0
    return out
