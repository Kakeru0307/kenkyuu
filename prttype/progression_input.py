"""コード進行からバッキング用の骨格 MIDI を直接作る。"""

from __future__ import annotations

from pathlib import Path

import muspy

from makeData.builder import add_chord, build_music, make_guitar_track
from makeData.chord_schedule import chord_slot, pitches_for_slot, skeleton_onset_ticks
from makeData.constants import DEFAULT_BARS, TICKS_PER_BAR
from makeData.progressions import (
    PROGRESSION_BY_NAME,
    PROGRESSIONS,
    ProgressionSpec,
    list_progression_summary,
    progression_chord_pitch_sets,
)


def get_progression(name: str) -> ProgressionSpec:
    if name not in PROGRESSION_BY_NAME:
        known = ", ".join(sorted(PROGRESSION_BY_NAME))
        raise KeyError(f"未知の進行名: {name}. 候補: {known}")
    return PROGRESSION_BY_NAME[name]


def build_backing_skeleton_music(
    *,
    progression: str | ProgressionSpec,
    key: str,
    bars: int = DEFAULT_BARS,
    bpm: float = 120.0,
    bars_per_chord: int = 1,
    chords_per_bar: int = 1,
    raise_odd_loop_last: bool = False,
    onset_duration: int = 2,
) -> muspy.Music:
    """進行の各コード頭にコードトーンだけ置いた骨格 MIDI を作る。

    chords_per_bar=2 のとき半小節（tick 0 と 8）にも置く。
    """
    spec = get_progression(progression) if isinstance(progression, str) else progression
    if bars < 8:
        raise ValueError("パッチ化のため bars は 8 以上にしてください")
    if bars_per_chord < 1:
        raise ValueError("bars_per_chord は 1 以上です")

    track = make_guitar_track("BackingSkeleton")
    pitch_sets = progression_chord_pitch_sets(spec, key)
    cpb = max(1, int(chords_per_bar))
    bpc = 1 if cpb >= 2 else max(1, int(bars_per_chord))
    onsets = skeleton_onset_ticks(chords_per_bar=cpb)

    for bar in range(bars):
        for tick in onsets:
            slot = chord_slot(bar, tick, bars_per_chord=bpc, chords_per_bar=cpb)
            pitches = pitches_for_slot(
                pitch_sets, slot, raise_odd_loop_last=raise_odd_loop_last
            )
            if not pitches:
                continue
            add_chord(
                track,
                pitches,
                time=bar * TICKS_PER_BAR + tick,
                duration=onset_duration,
            )

    music = build_music(track, bars=bars, tempo=float(bpm))
    return music


def save_backing_skeleton(
    output_path: Path,
    *,
    progression: str,
    key: str,
    bars: int = DEFAULT_BARS,
    bpm: float = 120.0,
    bars_per_chord: int = 1,
    chords_per_bar: int = 1,
    raise_odd_loop_last: bool = False,
) -> muspy.Music:
    music = build_backing_skeleton_music(
        progression=progression,
        key=key,
        bars=bars,
        bpm=bpm,
        bars_per_chord=bars_per_chord,
        chords_per_bar=chords_per_bar,
        raise_odd_loop_last=raise_odd_loop_last,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    muspy.write_midi(output_path, music)
    return music


def format_progression_catalog() -> str:
    lines = [
        f"{'name':22} {'family':16} {'key':4} chords",
        "-" * 72,
    ]
    for row in list_progression_summary():
        lines.append(
            f"{row['name']:22} {row['family']:16} {row['example_key']:4} {row['chords']}"
        )
    lines.append(f"\n合計 {len(PROGRESSIONS)} 種")
    return "\n".join(lines)
