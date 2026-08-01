import muspy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math


def get_hyperscore_per_track(notes, T):
    """
    notes: [(onset, duration, pitch)] 其中onset/duration单位为16分音符
    T: bar_num
    """

    ticks_per_bar = 16
    notes = sorted(notes, key=lambda x: x[0])

    bar_indices = [[] for _ in range(T)]
    for idx, (onset, duration, pitch) in enumerate(notes):
        bar_idx = onset // ticks_per_bar
        if 0 <= bar_idx < T:
            bar_indices[bar_idx].append(idx)

    # Function: Melodic/Harmonic

    is_melodic_function = np.zeros(shape=(T, 8), dtype=bool)
    is_harmonic_function = np.zeros(shape=(T, 8), dtype=bool)

    # Rest percentage
    is_long_note = np.zeros(shape=(T, 8), dtype=bool)

    for t in range(T):
        if not bar_indices[t]:
            continue

        bar_notes = [notes[i] for i in bar_indices[t]]

        # 1. 音高
        avg_pitch = np.mean([p for _, _, p in bar_notes])
        pitch_bin = int(avg_pitch // 16)
        pitch_bin = pitch_bin if pitch_bin < 8 else 7

        # 2. avg_range and sim_notes
        groups = defaultdict(list)
        for onset, dur, pitch in bar_notes:
            groups[onset].append((pitch, dur))
        deltas = [(max(p for p, _ in group) - min(p for p, _ in group)) if len(group) > 1 else 0
                  for group in groups.values()]
        simu_notes_num = [len(group) for group in groups.values()]

        avg_range = np.mean(deltas) if deltas else 0
        avg_sim_notes_num = np.mean(simu_notes_num) if simu_notes_num else 0

        if avg_range == 0 or avg_sim_notes_num == 1:
            is_melodic_function[t, pitch_bin] = True
        else:
            is_harmonic_function[t, pitch_bin] = True

        # 3. is long note
        occupied = np.zeros(shape=(16,), dtype=bool)
        min_onset = min([onset for onset, _, _ in bar_notes])
        for onset, dur, pitch in bar_notes:
            onset = onset - min_onset
            occupied[onset: onset + dur] = True
        rest_percentage = 1.0 - np.mean(occupied)
        is_long_note[t, pitch_bin] = rest_percentage < 0.3

    return is_melodic_function, is_harmonic_function, is_long_note

# General MIDI Percussion Map (Channel 10) pitch 分类
# 分类规则：
#   0 = 低频 (Kick, Floor Tom)
#   1 = 中频 (Snare, Mid Tom, Hand Clap)
#   2 = 高频 (Hi-Hat, Cymbal, Ride)


DRUM_FREQ_CLASS = {
    # --- 低频 (0) ---
    35: 0,  # Acoustic Bass Drum (原声低音鼓)
    36: 0,  # Bass Drum 1 (低音鼓)
    41: 0,  # Low Floor Tom (低落地通鼓)
    43: 0,  # High Floor Tom (高落地通鼓)
    45: 0,  # Low Tom (低通鼓)
    47: 0,  # Low-Mid Tom (低中通鼓)

    # --- 中频 (1) ---
    37: 1,  # Side Stick (边击鼓)
    38: 1,  # Acoustic Snare (原声军鼓)
    39: 1,  # Hand Clap (拍手)
    40: 1,  # Electric Snare (电子军鼓)
    48: 1,  # High-Mid Tom (高中通鼓)
    50: 1,  # High Tom (高通鼓)

    # --- 高频 (2) ---
    42: 2,  # Closed Hi-Hat (闭镲)
    44: 2,  # Pedal Hi-Hat (踩镲)
    46: 2,  # Open Hi-Hat (开镲)
    49: 2,  # Crash Cymbal 1 (碎音镲)
    51: 2,  # Ride Cymbal 1 (Ride 镲)
    52: 2,  # Chinese Cymbal (中国镲)
    53: 2,  # Ride Bell (Ride 镲钟)
    55: 2,  # Splash Cymbal (溅镲)
    57: 2,  # Crash Cymbal 2 (碎音镲2)
    59: 2,  # Ride Cymbal 2 (Ride 镲2)
}


def drum_pitch_to_freq_class(pitch: int) -> int:
    """
    将 MIDI 打击乐 pitch 映射为频率类别:
    0 = 低频 (Kick, Floor Tom)
    1 = 中频 (Snare, Mid Tom, Clap)
    2 = 高频 (Hi-Hat, Cymbals, Ride)
    """
    return DRUM_FREQ_CLASS.get(pitch, 1)  # 默认归为中频


def get_hyperscore_drum(notes, T):
    ticks_per_bar = 16
    notes = sorted(notes, key=lambda x: x[0])

    bar_indices = [[] for _ in range(T)]
    for idx, (onset, duration, pitch) in enumerate(notes):
        bar_idx = onset // ticks_per_bar
        if 0 <= bar_idx < T:
            bar_indices[bar_idx].append(idx)

    has_note = np.zeros(shape=(T, 3), dtype=np.bool)

    is_long_note = np.zeros(shape=(T, 3), dtype=np.bool)

    for t in range(T):
        if not bar_indices[t]:
            continue
        bar_notes = [notes[i] for i in bar_indices[t]]
        notes_in_diff_freq_class = [[], [], []]
        for onset, duration, pitch in bar_notes:
            notes_in_diff_freq_class[drum_pitch_to_freq_class(
                pitch)].append((onset, duration, pitch))

        for freq_class in range(3):
            freq_notes = notes_in_diff_freq_class[freq_class]
            if len(freq_notes) < 1:
                continue
            has_note[t, freq_class] = True
            occupied = np.zeros(shape=(16,), dtype=bool)
            min_onset = min([onset for onset, _, _ in freq_notes])
            for onset, dur, pitch in freq_notes:
                onset = onset - min_onset
                occupied[onset: onset + dur] = True
            rest_percentage = 1.0 - np.mean(occupied)
            is_long_note[t, freq_class] = rest_percentage < 0.3

    return has_note, is_long_note
