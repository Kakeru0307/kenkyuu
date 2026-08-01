"""target パッチから粗い骨格（入力）パッチを生成する。"""

from __future__ import annotations

import numpy as np

from midi_to_patch import TICKS_PER_BAR
from program_utils import GUITAR_PROGRAM

GUITAR_CATEGORY = GUITAR_PROGRAM // 8

PAIR_MODES = (
    "identity",
    "onset_to_full",
    "downbeat_chord",
    "root_per_bar",
    "melody_line",
)

SKELETON_MODES = frozenset({"downbeat_chord", "root_per_bar", "melody_line"})


def is_skeleton_mode(mode: str) -> bool:
    return mode in SKELETON_MODES


def _pitches_with_onsets_in_bar(channel: np.ndarray, bar_start: int, bar_end: int) -> list[int]:
    bar_slice = channel[bar_start:bar_end, :]
    pitches: list[int] = []
    for pitch in range(bar_slice.shape[1]):
        if np.any(bar_slice[:, pitch] == 1):
            pitches.append(pitch)
    return sorted(set(pitches))


def _select_chord_tones(pitches: list[int], *, max_notes: int = 4) -> list[int]:
    """小節内の音からコードトーンらしい音を最大 max_notes 個選ぶ。"""
    if not pitches:
        return []
    if len(pitches) <= max_notes:
        return pitches

    root = pitches[0]
    selected = [root]
    for interval in (4, 3, 7, 10):
        if len(selected) >= max_notes:
            break
        target = root + interval
        candidates = [p for p in pitches if p not in selected]
        if not candidates:
            break
        nearest = min(candidates, key=lambda pitch: abs(pitch - target))
        selected.append(nearest)
    return sorted(selected)[:max_notes]


def make_downbeat_chord_skeleton(
    tonal_chw: np.ndarray,
    *,
    category: int = GUITAR_CATEGORY,
    ticks_per_bar: int = TICKS_PER_BAR,
    max_chord_tones: int = 4,
) -> np.ndarray:
    """各小節の頭に、その小節で鳴っている音からコードトーンを置く。"""
    skeletal = np.zeros_like(tonal_chw)
    channel = tonal_chw[category]
    time_steps = tonal_chw.shape[1]

    for bar_start in range(0, time_steps, ticks_per_bar):
        bar_end = min(bar_start + ticks_per_bar, time_steps)
        pitches = _pitches_with_onsets_in_bar(channel, bar_start, bar_end)
        for pitch in _select_chord_tones(pitches, max_notes=max_chord_tones):
            skeletal[category, bar_start, pitch] = 1

    return skeletal


def make_root_per_bar_skeleton(
    tonal_chw: np.ndarray,
    *,
    category: int = GUITAR_CATEGORY,
    ticks_per_bar: int = TICKS_PER_BAR,
) -> np.ndarray:
    """各小節の頭にルート（最低音）だけを置く。"""
    skeletal = np.zeros_like(tonal_chw)
    channel = tonal_chw[category]
    time_steps = tonal_chw.shape[1]

    for bar_start in range(0, time_steps, ticks_per_bar):
        bar_end = min(bar_start + ticks_per_bar, time_steps)
        pitches = _pitches_with_onsets_in_bar(channel, bar_start, bar_end)
        if pitches:
            skeletal[category, bar_start, pitches[0]] = 1

    return skeletal


def make_melody_line_skeleton(
    tonal_chw: np.ndarray,
    *,
    category: int = GUITAR_CATEGORY,
) -> np.ndarray:
    """各時刻の最高音だけを残すメロディ線骨格。"""
    skeletal = np.zeros_like(tonal_chw)
    channel = tonal_chw[category]

    for time in range(tonal_chw.shape[1]):
        active = [pitch for pitch in range(128) if channel[time, pitch] == 1]
        if active:
            skeletal[category, time, max(active)] = 1

    return skeletal


def make_input_tonal(target_tonal: np.ndarray, mode: str) -> np.ndarray:
    """target tonal (11,128,128) から入力パッチを作る。"""
    if mode == "identity":
        return target_tonal.copy()

    if mode == "onset_to_full":
        input_tonal = np.zeros_like(target_tonal)
        input_tonal[target_tonal == 1] = 1
        return input_tonal

    if mode == "downbeat_chord":
        return make_downbeat_chord_skeleton(target_tonal)

    if mode == "root_per_bar":
        return make_root_per_bar_skeleton(target_tonal)

    if mode == "melody_line":
        return make_melody_line_skeleton(target_tonal)

    if mode == "full":
        return target_tonal.copy()

    raise ValueError(f"未対応の mode: {mode}")
