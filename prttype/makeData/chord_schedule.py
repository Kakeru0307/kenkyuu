"""Chord timing helpers for bars / half-bars and loop voicing lifts."""

from __future__ import annotations

from .constants import TICKS_PER_BAR


def chord_slot(
    bar: int,
    tick_in_bar: int,
    *,
    bars_per_chord: int = 1,
    chords_per_bar: int = 1,
) -> int:
    """Monotonic chord slot index along the song timeline."""
    cpb = max(1, int(chords_per_bar))
    bpc = max(1, int(bars_per_chord))
    if cpb >= 2:
        half = 0 if int(tick_in_bar) < (TICKS_PER_BAR // 2) else 1
        return int(bar) * cpb + half
    return int(bar) // bpc


def pitches_for_slot(
    pitch_sets: list[list[int]],
    slot: int,
    *,
    raise_odd_loop_last: bool = True,
) -> list[int]:
    """Pick chord pitches for slot; raise last chord one octave on odd loops (2nd, 4th…)."""
    if not pitch_sets:
        return []
    n = len(pitch_sets)
    loop = int(slot) // n
    idx = int(slot) % n
    pitches = list(pitch_sets[idx])
    if raise_odd_loop_last and (loop % 2 == 1) and idx == n - 1:
        pitches = [min(127, int(p) + 12) for p in pitches]
    return pitches


def skeleton_onset_ticks(*, chords_per_bar: int = 1) -> list[int]:
    """Where to plant skeleton chord onsets inside a bar."""
    if max(1, int(chords_per_bar)) >= 2:
        return [0, TICKS_PER_BAR // 2]
    return [0]
