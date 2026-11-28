"""Lead melody thinning: density cap + optional scale/chord preference."""

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

# 0=現状 / 1=ハード落としのみ廃止 / 2=減点弱め / 3=和声ほぼ無視 / 4=和声スコアなし
HARMONY_THIN_LEVELS = frozenset({0, 1, 2, 3, 4})

# major / minor pentatonic（tonic からの半音）
_PENTA_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 7, 9),
    "natural_minor": (0, 3, 5, 7, 10),
}
# ギター・ペンタでよく出る進行音程（絶対半音）
_MELODIC_FIT_INTERVALS = frozenset({1, 2, 3, 4, 5, 7, 12})
# 効果が出る程度（chord 40/25 より弱く、非ペンタ・スケール音を押し下げられる強さ）
PENTA_BONUS_STRONG = 26.0
PENTA_BONUS_WEAK = 20.0
PENTA_MELODIC_EXTRA = 18.0


def scale_pitch_classes(key: str, mode: str) -> frozenset[int]:
    tonic = KEY_TO_SEMITONE[key]
    return frozenset((tonic + iv) % 12 for iv in SCALE_INTERVALS[mode])


def pentatonic_pitch_classes(key: str, mode: str) -> frozenset[int]:
    tonic = KEY_TO_SEMITONE[key]
    intervals = _PENTA_INTERVALS.get(mode) or _PENTA_INTERVALS["natural_minor"]
    return frozenset((tonic + iv) % 12 for iv in intervals)


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
    penta_pcs: frozenset[int] | None = None,
    prev_pitch: int | None = None,
    harmony_thin_level: int = 0,
) -> float:
    """Higher = keep preferentially."""
    pc = int(note.pitch) % 12
    tick_in_bar = int(note.time) % TICKS_PER_BAR
    strong = tick_in_bar in (0, TICKS_PER_BAR // 2)
    score = float(note.velocity) + min(float(note.duration), float(TICKS_PER_BAR)) * 0.5
    level = int(harmony_thin_level)
    penta = penta_pcs or frozenset()

    if level < 4:
        in_penta = pc in penta
        if in_penta:
            score += PENTA_BONUS_STRONG if strong else PENTA_BONUS_WEAK
        if prev_pitch is not None and in_penta:
            interval = abs(int(note.pitch) - int(prev_pitch))
            deg = interval % 12
            if interval in _MELODIC_FIT_INTERVALS or deg in (1, 2, 3, 4, 5, 7):
                score += PENTA_MELODIC_EXTRA

    if level >= 3:
        return score

    if level == 2:
        chord_bonus_strong, chord_bonus_weak = 20.0, 12.0
        scale_bonus_strong, scale_bonus_weak = 6.0, 3.0
        oos_pen_strong, oos_pen_weak = 5.0, 8.0
        weak_oos_extra = 5.0
    else:
        # level 0 / 1: 現行スコア
        chord_bonus_strong, chord_bonus_weak = 40.0, 25.0
        scale_bonus_strong, scale_bonus_weak = 10.0, 5.0
        oos_pen_strong, oos_pen_weak = 5.0, 20.0
        weak_oos_extra = 25.0

    if pc in chord_pcs:
        score += chord_bonus_strong if strong else chord_bonus_weak
    elif pc in scale_pcs:
        score += scale_bonus_strong if strong else scale_bonus_weak
    else:
        score -= oos_pen_weak if not strong else oos_pen_strong
    if not strong and pc not in scale_pcs:
        score -= weak_oos_extra
    return score


def cap_lead_note_durations(
    music: muspy.Music,
    *,
    max_duration_ticks: int = 8,
    cut_to_next: bool = True,
    protect_keys: set[tuple[int, int]] | frozenset[tuple[int, int]] | None = None,
) -> muspy.Music:
    """音価を抑える。リードは単音想定。

    - cut_to_next: 次ノート onset 直前までで切る（onset/sustain 由来どちらも対象）
    - max_duration_ticks: >0 なら絶対上限も掛ける。<=0 なら上限なし（next-cut のみ可）
    - protect_keys: 絶対上限だけ免除（next-cut は免除しない）
    """
    use_max = int(max_duration_ticks) > 0
    max_dur = max(1, int(max_duration_ticks)) if use_max else None
    protected = protect_keys or frozenset()
    if not cut_to_next and not use_max:
        return music
    for track in music.tracks:
        if track.is_drum:
            continue
        notes = sorted(track.notes, key=lambda n: (int(n.time), int(n.pitch)))
        for i, note in enumerate(notes):
            start = int(note.time)
            end = start + max(1, int(note.duration))
            key = (start, int(note.pitch))
            if use_max and max_dur is not None and key not in protected:
                end = min(end, start + max_dur)
            if cut_to_next and i + 1 < len(notes):
                next_on = int(notes[i + 1].time)
                if next_on > start:
                    end = min(end, next_on)
            note.duration = max(1, end - start)
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
    harmony_thin_level: int = 0,
) -> muspy.Music:
    """Drop excess / non-melodic notes; keep monophonic-friendly sparse line.

    harmony_thin_level:
      0 = 現状（弱拍 OOS ハード落とし + 強い和声スコア）
      1 = ハード落としなし、スコアは現状
      2 = ハード落としなし、外音減点を弱める
      3 = ハード落としなし、和声ほぼ無視（vel/dur のみ）
      4 = ハード落としなし、和声スコアなし（単音化＋密度のみ）
    """
    from util.progression_input import get_progression

    level = int(harmony_thin_level)
    if level not in HARMONY_THIN_LEVELS:
        raise ValueError(f"harmony_thin_level must be in {sorted(HARMONY_THIN_LEVELS)}, got {level}")

    spec = progression if isinstance(progression, ProgressionSpec) else get_progression(progression)
    scale_pcs = scale_pitch_classes(key, mode)
    penta_pcs = pentatonic_pitch_classes(key, mode)
    chord_bars = chord_pcs_by_bar(spec, key, bars, bars_per_chord)
    max_onsets = max(1, int(max_onsets_per_bar))
    hard_drop_oos = level == 0

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
        prev_pitch: int | None = None
        for bar in range(max(bars, max(by_bar.keys(), default=-1) + 1)):
            notes = by_bar.get(bar, [])
            if not notes:
                continue
            chord_pcs = chord_bars[bar] if bar < len(chord_bars) else scale_pcs
            # 同時発音はペンタ優先、同点なら高音
            by_time: dict[int, list[muspy.Note]] = defaultdict(list)
            for n in notes:
                by_time[int(n.time)].append(n)
            collapsed: list[muspy.Note] = []
            for _t, group in sorted(by_time.items()):
                group = sorted(
                    group,
                    key=lambda n: (
                        1 if (int(n.pitch) % 12) in penta_pcs else 0,
                        int(n.pitch),
                    ),
                )
                collapsed.append(group[-1])

            filtered: list[muspy.Note] = []
            for n in collapsed:
                if hard_drop_oos:
                    pc = int(n.pitch) % 12
                    tick_in_bar = int(n.time) % ticks_per_bar
                    strong = tick_in_bar in (0, ticks_per_bar // 2)
                    if (not strong) and pc not in scale_pcs and pc not in chord_pcs:
                        continue
                filtered.append(n)

            if len(filtered) > max_onsets:
                timed = sorted(filtered, key=lambda n: (int(n.time), int(n.pitch)))
                prev_for: dict[int, int | None] = {}
                for i, n in enumerate(timed):
                    if i > 0:
                        prev_for[id(n)] = int(timed[i - 1].pitch)
                    else:
                        prev_for[id(n)] = prev_pitch
                ranked = sorted(
                    filtered,
                    key=lambda n: _note_score(
                        n,
                        bar=bar,
                        scale_pcs=scale_pcs,
                        chord_pcs=chord_pcs,
                        penta_pcs=penta_pcs,
                        prev_pitch=prev_for.get(id(n)),
                        harmony_thin_level=level,
                    ),
                    reverse=True,
                )
                filtered = sorted(ranked[:max_onsets], key=lambda n: (n.time, n.pitch))
            kept.extend(filtered)
            if filtered:
                prev_pitch = int(sorted(filtered, key=lambda n: (n.time, n.pitch))[-1].pitch)

        track.notes = sorted(kept, key=lambda n: (n.time, n.pitch))
    return music
