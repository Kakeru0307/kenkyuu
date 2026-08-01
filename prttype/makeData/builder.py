"""MIDI ノート列・トラック組み立て。"""

from __future__ import annotations

import muspy

from program_utils import GUITAR_PROGRAM

from .constants import (
    DEFAULT_BARS,
    DEFAULT_RESOLUTION,
    DEFAULT_VELOCITY,
    TICKS_PER_BAR,
)


def make_guitar_track(name: str = "guitar") -> muspy.Track:
    return muspy.Track(program=GUITAR_PROGRAM, is_drum=False, name=name)


def add_note(
    track: muspy.Track,
    *,
    time: int,
    pitch: int,
    duration: int,
    velocity: int = DEFAULT_VELOCITY,
) -> None:
    if duration <= 0:
        return
    track.append(
        muspy.Note(
            time=time,
            pitch=pitch,
            velocity=velocity,
            duration=duration,
        )
    )


def add_chord(
    track: muspy.Track,
    pitches: list[int],
    *,
    time: int,
    duration: int,
    velocity: int = DEFAULT_VELOCITY,
) -> None:
    for pitch in pitches:
        add_note(track, time=time, pitch=pitch, duration=duration, velocity=velocity)


def build_music(
    track: muspy.Track,
    *,
    bars: int = DEFAULT_BARS,
    tempo: float = 120.0,
) -> muspy.Music:
    end_time = bars * TICKS_PER_BAR
    trimmed = muspy.Track(program=track.program, is_drum=track.is_drum, name=track.name)
    for note in track:
        if note.time < end_time:
            trimmed.append(note)
    return muspy.Music(
        resolution=DEFAULT_RESOLUTION,
        tracks=[trimmed],
        tempos=[muspy.Tempo(time=0, qpm=tempo)],
    )
