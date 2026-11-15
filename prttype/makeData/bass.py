"""ベースライン合成（コードルートのみ・energy で密度上限）。

発音グリッドと音価はすべて 4tick（四分）。8分以下の教師は作らない。
"""

from __future__ import annotations

import random
from typing import Literal

import muspy

from .builder import add_note, build_music
from .constants import (
    BASS_PROGRAM,
    BEAT_TICKS,
    DEFAULT_BARS,
    KEY_TO_SEMITONE,
    BASS_PITCH_MAX,
    BASS_PITCH_MIN,
    TICKS_PER_BAR,
)
from .progressions import ProgressionSpec, resolve_progression_chords

EnergyLevel = Literal["low", "mid", "high"]

# 小節あたり onset 上限 / 小節内 onset tick（四分）
BASS_DENSITY: dict[EnergyLevel, tuple[int, tuple[int, ...]]] = {
    "low": (1, (0,)),
    "mid": (2, (0, 2 * BEAT_TICKS)),
    "high": (4, (0, BEAT_TICKS, 2 * BEAT_TICKS, 3 * BEAT_TICKS)),
}


def make_bass_track(name: str = "Bass") -> muspy.Track:
    return muspy.Track(program=BASS_PROGRAM, is_drum=False, name=name)


def root_in_bass_range(root_key: str) -> int:
    """コードルートをベース音域（28–55）に移調する。"""
    pc = KEY_TO_SEMITONE[root_key]
    pitch = pc + 36  # C2 付近を基準
    while pitch < BASS_PITCH_MIN:
        pitch += 12
    while pitch > BASS_PITCH_MAX:
        pitch -= 12
    return pitch


def max_onsets_for_energy(energy: EnergyLevel) -> int:
    return BASS_DENSITY[energy][0]


def _chord_roots_for_bars(
    spec: ProgressionSpec,
    key: str,
    bars: int,
    bars_per_chord: int,
) -> list[str]:
    chords = resolve_progression_chords(spec, key)
    roots: list[str] = []
    bpc = max(1, bars_per_chord)
    for bar in range(bars):
        slot = (bar // bpc) % len(chords)
        roots.append(chords[slot][0])
    return roots


def _velocity_for_energy(energy: EnergyLevel, rng: random.Random) -> int:
    if energy == "low":
        return rng.randint(70, 85)
    if energy == "mid":
        return rng.randint(80, 95)
    return rng.randint(90, 110)


def _bar_onset_count(track: muspy.Track, bar: int) -> int:
    start = bar * TICKS_PER_BAR
    end = start + TICKS_PER_BAR
    return sum(1 for n in track.notes if start <= int(n.time) < end)


def _add_kick_aligned_roots(
    track: muspy.Track,
    *,
    roots: list[str],
    kick_times: set[int] | frozenset[int] | None,
    bars: int,
    energy: EnergyLevel,
    rng: random.Random,
) -> None:
    """キック時刻を四分に量子化し、小節上限内ならルートを足す。"""
    if not kick_times:
        return
    max_onsets = max_onsets_for_energy(energy)
    vel = _velocity_for_energy(energy, rng)
    for t in sorted(kick_times):
        bar = int(t) // TICKS_PER_BAR
        if bar < 0 or bar >= bars:
            continue
        t4 = (int(t) // BEAT_TICKS) * BEAT_TICKS
        if any(int(n.time) == t4 for n in track.notes):
            continue
        if _bar_onset_count(track, bar) >= max_onsets:
            continue
        root = root_in_bass_range(roots[bar])
        add_note(track, time=t4, pitch=root, duration=BEAT_TICKS, velocity=vel)


def generate_bass_line(
    *,
    spec: ProgressionSpec,
    key: str,
    bpm: float = 120.0,
    bars: int = DEFAULT_BARS,
    bars_per_chord: int = 1,
    energy: EnergyLevel = "mid",
    kick_times: set[int] | frozenset[int] | None = None,
    seed: int | None = None,
    rng: random.Random | None = None,
    home_mode: str | None = None,
) -> muspy.Music:
    """コードルートのみのベースライン。density は energy 表に固定。"""
    del home_mode  # スケール経過は使わない
    rng = rng or random.Random(seed)
    track = make_bass_track()
    roots = _chord_roots_for_bars(spec, key, bars, bars_per_chord)
    vel = _velocity_for_energy(energy, rng)
    _max, onsets = BASS_DENSITY[energy]

    for bar in range(bars):
        bar_t = bar * TICKS_PER_BAR
        root = root_in_bass_range(roots[bar])
        for onset in onsets:
            add_note(
                track,
                time=bar_t + onset,
                pitch=root,
                duration=BEAT_TICKS,
                velocity=vel,
            )

    _add_kick_aligned_roots(
        track,
        roots=roots,
        kick_times=kick_times,
        bars=bars,
        energy=energy,
        rng=rng,
    )
    return build_music(track, bars=bars, tempo=float(bpm))
