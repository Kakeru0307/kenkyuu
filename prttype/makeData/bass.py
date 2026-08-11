"""ベースライン合成（コードルート中心・energy で密度変化）。"""

from __future__ import annotations

import random
from typing import Literal

import muspy

from .builder import add_note, build_music
from .constants import (
    BASS_PITCH_MAX,
    BASS_PITCH_MIN,
    BASS_PROGRAM,
    BEAT_TICKS,
    DEFAULT_BARS,
    KEY_TO_SEMITONE,
    SCALE_INTERVALS,
    TICKS_PER_BAR,
)
from .progressions import ProgressionSpec, resolve_progression_chords

EnergyLevel = Literal["low", "mid", "high"]


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


def fifth_in_bass_range(root: int) -> int:
    fifth = root + 7
    if fifth > BASS_PITCH_MAX:
        fifth = root - 5
    if fifth < BASS_PITCH_MIN:
        fifth = root + 7
        while fifth > BASS_PITCH_MAX:
            fifth -= 12
    return fifth


def _scale_pitch_classes(mode: str, tonic_key: str) -> list[int]:
    tonic = KEY_TO_SEMITONE[tonic_key]
    return [(tonic + iv) % 12 for iv in SCALE_INTERVALS[mode]]


def _approach_pitch(target: int, rng: random.Random) -> int:
    """次ルートへの半音アプローチ。"""
    cand = target + rng.choice((-1, 1))
    if cand < BASS_PITCH_MIN:
        cand = target + 1
    if cand > BASS_PITCH_MAX:
        cand = target - 1
    return cand


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


def _add_kick_aligned_roots(
    track: muspy.Track,
    *,
    roots: list[str],
    kick_times: set[int] | frozenset[int] | None,
    bars: int,
    energy: EnergyLevel,
    rng: random.Random,
) -> None:
    """キック時刻付近にルートを置く（協調用の後処理強化）。"""
    if not kick_times:
        return
    vel = _velocity_for_energy(energy, rng)
    for t in sorted(kick_times):
        bar = t // TICKS_PER_BAR
        if bar < 0 or bar >= bars:
            continue
        # 既に同じ tick にノートがあればスキップ
        if any(n.time == t for n in track.notes):
            continue
        root = root_in_bass_range(roots[bar])
        add_note(track, time=t, pitch=root, duration=2, velocity=vel)


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
) -> muspy.Music:
    """コードルート中心のベースラインを生成する。"""
    rng = rng or random.Random(seed)
    track = make_bass_track()
    roots = _chord_roots_for_bars(spec, key, bars, bars_per_chord)
    next_roots = roots[1:] + [roots[0]]
    mode = spec.mode
    vel = _velocity_for_energy(energy, rng)

    for bar in range(bars):
        bar_t = bar * TICKS_PER_BAR
        root = root_in_bass_range(roots[bar])
        fifth = fifth_in_bass_range(root)
        next_root = root_in_bass_range(next_roots[bar])

        if energy == "low":
            # 全音符相当（小節頭のみ）
            add_note(track, time=bar_t, pitch=root, duration=TICKS_PER_BAR - 1, velocity=vel)
            continue

        if energy == "mid":
            # BPM 依存: 遅→2分、速→4分。root / fifth 交互
            if bpm < 90:
                onsets = (0, 2 * BEAT_TICKS)
                durs = (2 * BEAT_TICKS - 1, 2 * BEAT_TICKS - 1)
                pitches = (root, fifth)
            else:
                onsets = (0, BEAT_TICKS, 2 * BEAT_TICKS, 3 * BEAT_TICKS)
                durs = (BEAT_TICKS - 1,) * 4
                pitches = (root, root, fifth, fifth)
            for onset, dur, pitch in zip(onsets, durs, pitches):
                add_note(
                    track,
                    time=bar_t + onset,
                    pitch=pitch,
                    duration=dur,
                    velocity=vel,
                )
            continue

        # high: 8分駆動 or ウォーキング
        if rng.random() < 0.45:
            # walking: 4分 × 4、末拍は半音アプローチ
            scale_pcs = _scale_pitch_classes(mode, key)
            mid_pc = scale_pcs[rng.randint(0, len(scale_pcs) - 1)]
            mid = mid_pc + (root // 12) * 12
            while mid < BASS_PITCH_MIN:
                mid += 12
            while mid > BASS_PITCH_MAX:
                mid -= 12
            approach = _approach_pitch(next_root, rng)
            pitches = (root, fifth, mid, approach)
            for i, pitch in enumerate(pitches):
                add_note(
                    track,
                    time=bar_t + i * BEAT_TICKS,
                    pitch=pitch,
                    duration=BEAT_TICKS - 1,
                    velocity=vel,
                )
        else:
            # driving 8ths: root-fifth-octave-fifth 繰り返し
            octave = root + 12
            if octave > BASS_PITCH_MAX:
                octave = root
            pattern = (root, fifth, octave, fifth)
            for i in range(8):
                add_note(
                    track,
                    time=bar_t + i * 2,
                    pitch=pattern[i % 4],
                    duration=1,
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
