"""muspy.Music 向けの小さな共有ヘルパ。"""

from __future__ import annotations

import muspy

from makeData.constants import TICKS_PER_BAR


def crop_music_to_bars(music: muspy.Music, bars: int) -> muspy.Music:
    """先頭 bars 小節だけ残す（編成の短 span 用。生成は 8 小節以上で行い後で切る）。"""
    end = max(1, int(bars)) * TICKS_PER_BAR
    for track in music.tracks:
        kept: list[muspy.Note] = []
        for note in track.notes:
            t = int(note.time)
            if t >= end:
                continue
            dur = max(1, min(int(note.duration), end - t))
            kept.append(
                muspy.Note(
                    time=t,
                    duration=dur,
                    pitch=int(note.pitch),
                    velocity=int(note.velocity),
                )
            )
        track.notes = kept
    return music


def power_chord_attack_times(music: muspy.Music) -> set[int]:
    """第三音を含まず完全5度を含む同時発音の attack 時刻を返す。"""
    grouped: dict[int, set[int]] = {}
    for track in music.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            grouped.setdefault(int(note.time), set()).add(int(note.pitch))
    out: set[int] = set()
    for time, pitches in grouped.items():
        for root in pitches:
            intervals = {(pitch - root) % 12 for pitch in pitches}
            if 7 in intervals and 3 not in intervals and 4 not in intervals:
                out.add(time)
                break
    return out
