"""コード・音階のピッチ計算。"""

from __future__ import annotations

from .constants import (
    CHORD_INTERVALS,
    GUITAR_PITCH_MAX,
    GUITAR_PITCH_MIN,
    KEY_TO_SEMITONE,
    SCALE_INTERVALS,
)


def chord_pitches(key: str, quality: str, *, base_octave: int = 3) -> list[int]:
    root = KEY_TO_SEMITONE[key] + base_octave * 12
    intervals = CHORD_INTERVALS[quality]
    pitches = [root + interval for interval in intervals]
    return _clamp_guitar_range(pitches)


def power_chord_pitches(key: str, *, base_octave: int = 3) -> list[int]:
    """ルート＋完全5度の2音パワーコードをギター音域内で返す。"""
    root = KEY_TO_SEMITONE[key] + base_octave * 12
    while root < GUITAR_PITCH_MIN:
        root += 12
    while root + 7 > GUITAR_PITCH_MAX:
        root -= 12
    return [root, root + 7]


def scale_pitches(key: str, mode: str, *, base_octave: int = 3) -> list[int]:
    root = KEY_TO_SEMITONE[key] + base_octave * 12
    intervals = SCALE_INTERVALS[mode]
    pitches: list[int] = []
    for octave in range(2):
        for interval in intervals:
            pitch = root + interval + octave * 12
            if GUITAR_PITCH_MIN <= pitch <= GUITAR_PITCH_MAX:
                pitches.append(pitch)
    return pitches


def _clamp_guitar_range(pitches: list[int]) -> list[int]:
    clamped = [p for p in pitches if GUITAR_PITCH_MIN <= p <= GUITAR_PITCH_MAX]
    if clamped:
        return clamped
    return [max(GUITAR_PITCH_MIN, min(GUITAR_PITCH_MAX, pitches[0]))]
