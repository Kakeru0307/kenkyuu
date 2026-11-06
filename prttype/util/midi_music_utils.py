"""muspy.Music 向けの小さな共有ヘルパ。"""

from __future__ import annotations

import muspy


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
