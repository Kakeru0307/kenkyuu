"""BPM 条件チャンネル（U-Net 入力 11+1）。骨格に N は載せない。"""

from __future__ import annotations

import numpy as np

from makeData.constants import BPM_RANGE
from makeData.rhythm import preferred_attacks_range, sample_attacks_for_bpm

TONAL_CHANNELS = 11
COND_CHANNELS = 1
MODEL_IN_CHANNELS = TONAL_CHANNELS + COND_CHANNELS
MODEL_OUT_CHANNELS = TONAL_CHANNELS

BPM_MIN, BPM_MAX = float(BPM_RANGE[0]), float(BPM_RANGE[1])

__all__ = [
    "TONAL_CHANNELS",
    "COND_CHANNELS",
    "MODEL_IN_CHANNELS",
    "MODEL_OUT_CHANNELS",
    "bpm_to_unit",
    "make_bpm_cond_map",
    "concat_tonal_and_cond",
    "midi_tempo_bpm",
    "preferred_attacks_range",
    "sample_attacks_for_bpm",
]


def bpm_to_unit(bpm: float) -> float:
    return float(np.clip((float(bpm) - BPM_MIN) / (BPM_MAX - BPM_MIN), 0.0, 1.0))


def make_bpm_cond_map(
    bpm: float,
    *,
    height: int = 128,
    width: int = 128,
) -> np.ndarray:
    v = np.float32(bpm_to_unit(bpm))
    return np.full((1, height, width), v, dtype=np.float32)


def concat_tonal_and_cond(tonal_chw: np.ndarray, cond_chw: np.ndarray) -> np.ndarray:
    if tonal_chw.ndim != 3 or tonal_chw.shape[0] != TONAL_CHANNELS:
        raise ValueError(f"tonal は (11,H,W) である必要があります: {tonal_chw.shape}")
    if cond_chw.shape[0] != COND_CHANNELS:
        raise ValueError(f"cond は (1,H,W) である必要があります: {cond_chw.shape}")
    return np.concatenate(
        [tonal_chw.astype(np.float32, copy=False), cond_chw.astype(np.float32, copy=False)],
        axis=0,
    )


def midi_tempo_bpm(music) -> float:
    if getattr(music, "tempos", None):
        return float(music.tempos[0].qpm)
    return (BPM_MIN + BPM_MAX) / 2.0
