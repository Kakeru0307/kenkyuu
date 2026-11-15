"""Lead melody thinning: density cap + scale/chord preference."""

from __future__ import annotations

from collections import defaultdict

import muspy

from makeData.constants import (
    CHORD_INTERVALS,
    KEY_TO_SEMITONE,
    SCALE_INTERVALS,
    TICKS_PER_BAR,
)
from makeData.progressions import ProgressionSpec, resolve_progression_chords


def scale_pitch_classes(key: str, mode: str) -> frozenset[int]:
    tonic = KEY_TO_SEMITONE[key]
    return frozenset((tonic + iv) % 12 for iv in SCALE_INTERVALS[mode])


def chord_pitch_classes(root: str, quality: str) -> frozenset[int]:
    root_pc = KEY_TO_SEMITONE[root]
    return frozenset((root_pc + iv) % 12 for iv in CHORD_INTERVALS[quality])


def chord_pcs_by_bar(
    spec: ProgressionSpec,
    key: str,
    bars: int,
    bars_per_chord: int = 1,
) -> list[frozenset[int]]:
    chords = resolve_progression_chords(spec, key)
    bpc = max(1, int(bars_per_chord))
    out: list[frozenset[int]] = []
    for bar in range(bars):
        slot = (bar // bpc) % len(chords)
        root, quality = chords[slot]
        out.append(chord_pitch_classes(root, quality))
    return out


def _note_score(
    note: muspy.Note,
    *,
    bar: int,
    scale_pcs: frozenset[int],
    chord_pcs: frozenset[int],
) -> float:
    """Higher = keep preferentially."""
    pc = int(note.pitch) % 12
    tick_in_bar = int(note.time) % TICKS_PER_BAR
    strong = tick_in_bar in (0, TICKS_PER_BAR // 2)
    score = float(note.velocity) + min(float(note.duration), float(TICKS_PER_BAR)) * 0.5
    if pc in chord_pcs:
        score += 40.0 if strong else 25.0
    elif pc in scale_pcs:
        score += 10.0 if strong else 5.0
    else:
        score -= 20.0 if not strong else 5.0
    if not strong and pc not in scale_pcs:
        score -= 25.0
    return score


def cap_lead_note_durations(
    music: muspy.Music,
    *,
    max_duration_ticks: int = 8,
) -> muspy.Music:
    """次ノート onset まで（かつ max）で切る。リードは単音想定。"""
    max_dur = max(1, int(max_duration_ticks))
    for track in music.tracks:
        if track.is_drum:
            continue
        notes = sorted(track.notes, key=lambda n: (int(n.time), int(n.pitch)))
        for i, note in enumerate(notes):
            end = int(note.time) + min(max(1, int(note.duration)), max_dur)
            if i + 1 < len(notes):
                next_on = int(notes[i + 1].time)
                if next_on > int(note.time):
                    end = min(end, next_on)
            note.duration = max(1, end - int(note.time))
        track.notes = notes
    return music


def thin_lead_melody(
    music: muspy.Music,
    *,
    key: str,
    mode: str,
    progression: ProgressionSpec | str,
    bars: int,
    bars_per_chord: int = 1,
    max_onsets_per_bar: int = 6,
    ticks_per_bar: int = TICKS_PER_BAR,
) -> muspy.Music:
    """Drop excess / non-melodic notes; keep monophonic-friendly sparse line."""
    from util.progression_input import get_progression

    spec = progression if isinstance(progression, ProgressionSpec) else get_progression(progression)
    scale_pcs = scale_pitch_classes(key, mode)
    chord_bars = chord_pcs_by_bar(spec, key, bars, bars_per_chord)
    max_onsets = max(1, int(max_onsets_per_bar))

    for track in music.tracks:
        if track.is_drum:
            continue
        by_bar: dict[int, list[muspy.Note]] = defaultdict(list)
        for note in track.notes:
            bar = int(note.time) // ticks_per_bar
            if bar < 0:
                continue
            by_bar[bar].append(note)

        kept: list[muspy.Note] = []
        for bar in range(max(bars, max(by_bar.keys(), default=-1) + 1)):
            notes = by_bar.get(bar, [])
            if not notes:
                continue
            chord_pcs = chord_bars[bar] if bar < len(chord_bars) else scale_pcs
            # collapse same-time stacks to top pitch first
            by_time: dict[int, list[muspy.Note]] = defaultdict(list)
            for n in notes:
                by_time[int(n.time)].append(n)
            collapsed: list[muspy.Note] = []
            for _t, group in sorted(by_time.items()):
                group = sorted(group, key=lambda n: n.pitch)
                collapsed.append(group[-1])

            # drop weak out-of-scale on weak beats
            filtered: list[muspy.Note] = []
            for n in collapsed:
                pc = int(n.pitch) % 12
                tick_in_bar = int(n.time) % ticks_per_bar
                strong = tick_in_bar in (0, ticks_per_bar // 2)
                if (not strong) and pc not in scale_pcs and pc not in chord_pcs:
                    continue
                filtered.append(n)

            if len(filtered) > max_onsets:
                ranked = sorted(
                    filtered,
                    key=lambda n: _note_score(
                        n, bar=bar, scale_pcs=scale_pcs, chord_pcs=chord_pcs
                    ),
                    reverse=True,
                )
                filtered = sorted(ranked[:max_onsets], key=lambda n: (n.time, n.pitch))
            kept.extend(filtered)

        track.notes = sorted(kept, key=lambda n: (n.time, n.pitch))
    return music
