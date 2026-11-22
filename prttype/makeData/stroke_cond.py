"""バッキング stroke / arp 条件（one-hot）。drum beat_type と同型。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import GROOVE_COND_TYPES


def groove_cond_to_id(name: str) -> int:
    if name not in GROOVE_COND_TYPES:
        raise ValueError(
            f"unknown groove cond: {name!r}. candidates: {GROOVE_COND_TYPES}"
        )
    return GROOVE_COND_TYPES.index(name)


def make_groove_onehot_map(
    groove_id: int,
    *,
    height: int,
    width: int,
    n_types: int | None = None,
) -> np.ndarray:
    """groove を (N,H,W) の one-hot 定数マップにする。"""
    n = len(GROOVE_COND_TYPES) if n_types is None else int(n_types)
    out = np.zeros((n, height, width), dtype=np.float32)
    idx = int(groove_id)
    if not (0 <= idx < n):
        raise ValueError(f"groove_id out of range: {groove_id}")
    out[idx] = 1.0
    return out


def resolve_groove_cond_name(meta: dict[str, Any] | None) -> str:
    """合成 manifest エントリから条件名を決める。"""
    if not meta:
        return "eight_even"
    stroke = meta.get("stroke_groove")
    if isinstance(stroke, str) and stroke in GROOVE_COND_TYPES:
        return stroke
    pattern = meta.get("pattern")
    if pattern in ("progression_arpeggio", "arpeggio"):
        return "arpeggio"
    if isinstance(stroke, str):
        raise ValueError(f"unknown stroke_groove in meta: {stroke!r}")
    return "eight_even"
