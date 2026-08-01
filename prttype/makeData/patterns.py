"""ストローク・アルペジオ・音階・進行パターン。"""

from __future__ import annotations

import random

import muspy

from .builder import add_chord, add_note, build_music, make_guitar_track
from .chord_schedule import chord_slot, pitches_for_slot
from .constants import (
    ATTACKS_PER_BAR_WEIGHTS,
    BACKING_POWER_CHORD_PROBABILITY,
    BAR_LENGTH_CHOICES,
    BEAT_TICKS,
    DEFAULT_BARS,
    DEFAULT_VELOCITY,
    LEAD_POWER_CHORD_PHRASE_PROBABILITY,
    PATTERN_WEIGHTS,
    PLACEMENT_WEIGHTS,
    STRUM_ARTICULATION_WEIGHTS,
    TICKS_PER_BAR,
)
from .progressions import (
    PROGRESSIONS,
    ProgressionSpec,
    progression_chord_pitch_sets,
    resolve_progression_chords,
)
from .rhythm import sample_attacks_for_bpm
from .voicings import chord_pitches, power_chord_pitches, scale_pitches


def choose_strum_articulation(rng: random.Random) -> str:
    names = list(STRUM_ARTICULATION_WEIGHTS.keys())
    weights = [STRUM_ARTICULATION_WEIGHTS[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def choose_attacks_per_bar(rng: random.Random) -> int:
    ns = list(ATTACKS_PER_BAR_WEIGHTS.keys())
    weights = [ATTACKS_PER_BAR_WEIGHTS[n] for n in ns]
    return int(rng.choices(ns, weights=weights, k=1)[0])


def choose_placement(rng: random.Random) -> str:
    names = list(PLACEMENT_WEIGHTS.keys())
    weights = [PLACEMENT_WEIGHTS[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def resolve_onset_ticks(
    n: int,
    placement: str,
    *,
    ticks_per_bar: int = TICKS_PER_BAR,
    rng: random.Random,
) -> list[int]:
    """1小節内の onset tick（昇順・重複なし）。"""
    n = max(1, min(int(n), ticks_per_bar))
    all_ticks = list(range(ticks_per_bar))

    if placement == "sparse_random":
        return sorted(rng.sample(all_ticks, n))

    if placement == "offbeat":
        odds = [t for t in all_ticks if t % 2 == 1]
        evens = [t for t in all_ticks if t % 2 == 0]
        if n <= len(odds):
            # 均等に裏から取る
            step = len(odds) / n
            picked = [odds[min(len(odds) - 1, int(i * step))] for i in range(n)]
            # unique 化
            out: list[int] = []
            for t in picked:
                if t not in out:
                    out.append(t)
            for t in odds:
                if len(out) >= n:
                    break
                if t not in out:
                    out.append(t)
            return sorted(out[:n])
        return sorted(odds + evens[: n - len(odds)])

    if placement == "front":
        # 常に小節前半 (0..7)。N>8 は前半を埋め、余りを後半から補充
        front_pool = list(range(ticks_per_bar // 2))
        if n <= len(front_pool):
            step = len(front_pool) / n
            picked: list[int] = []
            for i in range(n):
                t = front_pool[min(len(front_pool) - 1, int(i * step))]
                if t not in picked:
                    picked.append(t)
            for t in front_pool:
                if len(picked) >= n:
                    break
                if t not in picked:
                    picked.append(t)
            return sorted(picked[:n])
        back_pool = list(range(ticks_per_bar // 2, ticks_per_bar))
        extra = n - len(front_pool)
        step = len(back_pool) / extra
        extras: list[int] = []
        for i in range(extra):
            t = back_pool[min(len(back_pool) - 1, int(i * step))]
            if t not in extras:
                extras.append(t)
        for t in back_pool:
            if len(extras) >= extra:
                break
            if t not in extras:
                extras.append(t)
        return sorted(front_pool + extras[:extra])

    if placement == "back":
        # 常に小節後半 (8..15)。N>8 は後半を埋め、余りを前半から補充
        back_pool = list(range(ticks_per_bar // 2, ticks_per_bar))
        if n <= len(back_pool):
            step = len(back_pool) / n
            picked = []
            for i in range(n):
                t = back_pool[min(len(back_pool) - 1, int(i * step))]
                if t not in picked:
                    picked.append(t)
            for t in back_pool:
                if len(picked) >= n:
                    break
                if t not in picked:
                    picked.append(t)
            return sorted(picked[:n])
        front_pool = list(range(ticks_per_bar // 2))
        extra = n - len(back_pool)
        step = len(front_pool) / extra
        extras = []
        for i in range(extra):
            t = front_pool[min(len(front_pool) - 1, int(i * step))]
            if t not in extras:
                extras.append(t)
        for t in front_pool:
            if len(extras) >= extra:
                break
            if t not in extras:
                extras.append(t)
        return sorted(extras[:extra] + back_pool)

    # even（既定）: グリッド均等配置 0, T/n, 2T/n, ...
    # 例: n=8, T=16 → 0,2,4,6,8,10,12,14
    if n == 1:
        return [0]
    picked = []
    for i in range(n):
        t = int(i * ticks_per_bar / n)
        t = max(0, min(ticks_per_bar - 1, t))
        if t not in picked:
            picked.append(t)
    for t in all_ticks:
        if len(picked) >= n:
            break
        if t not in picked:
            picked.append(t)
    return sorted(picked[:n])


def _strum_onset_duration(
    articulation: str,
    *,
    onset_index: int,
    onset_ticks: list[int],
    ticks_per_bar: int,
    rng: random.Random,
    rest_indices: frozenset[int],
) -> int | None:
    """onset 間ギャップに基づく duration。None は休符。

    - ストローク間: 可能なとき休符 2 tick（聞こえる切れ目）
    - 小節末: 伸ばしを既定。末の機械的 1tick 空きは稀
    """
    n = len(onset_ticks)
    tick = onset_ticks[onset_index]
    is_last = onset_index == n - 1
    if not is_last:
        gap = onset_ticks[onset_index + 1] - tick
    else:
        gap = max(1, ticks_per_bar - tick)

    # gap>=3 → 2 tick 休符、それ未満は 1 tick 休符が上限
    separated = max(gap - 2, 1) if gap >= 3 else max(gap - 1, 1)
    tight = max(gap - 1, 1)  # 旧: 常に 1 tick 空き
    short = max(1, min(2, max(gap // 2, 1)))

    if articulation == "solid":
        if is_last:
            if rng.random() < 0.15:
                return tight
            return gap
        return separated
    if articulation == "staccato":
        # 小節内は短く、末はだいたい埋める（毎小節末ギャップを避ける）
        if is_last and rng.random() < 0.75:
            return gap
        return short
    if articulation == "mixed":
        if is_last:
            if rng.random() < 0.2:
                return tight
            return gap
        return short if (onset_index % 2 == 1) else separated
    if articulation == "sustained":
        if onset_index % 2 != 0:
            return None
        if is_last:
            return gap
        return max(separated, min(ticks_per_bar - tick, max(gap * 2 - 2, separated)))
    if articulation == "rests":
        if onset_index in rest_indices:
            return None
        if is_last and rng.random() < 0.7:
            return gap
        return short if rng.random() < 0.35 else separated
    return separated if not is_last else gap


def _choose_rest_indices(rng: random.Random, n: int) -> frozenset[int]:
    if n <= 1:
        return frozenset()
    candidates = (
        frozenset({n - 1}),
        frozenset({i for i in range(n) if i % 2 == 1}) or frozenset({n - 1}),
        frozenset({n // 2}),
    )
    chosen = rng.choice(candidates)
    # 全部休みにしない
    if len(chosen) >= n:
        return frozenset({n - 1})
    return chosen


def _write_strum_bars(
    track: muspy.Track,
    *,
    pitch_sets: list[list[int]] | list,
    power_pitch_sets: list[list[int]] | None = None,
    power_chord_probability: float = 0.0,
    power_chord_onsets: set[int] | None = None,
    bars: int,
    bars_per_chord: int,
    onset_ticks: list[int],
    articulation: str,
    rng: random.Random,
    chords_per_bar: int = 1,
    raise_odd_loop_last: bool = True,
) -> None:
    rest_indices = (
        _choose_rest_indices(rng, len(onset_ticks))
        if articulation == "rests"
        else frozenset()
    )
    cpb = max(1, int(chords_per_bar))
    bpc = 1 if cpb >= 2 else max(1, int(bars_per_chord))
    for bar in range(bars):
        for oi, tick in enumerate(onset_ticks):
            slot = chord_slot(bar, tick, bars_per_chord=bpc, chords_per_bar=cpb)
            use_power = (
                power_pitch_sets is not None
                and rng.random() < power_chord_probability
            )
            source_pitch_sets = power_pitch_sets if use_power else list(pitch_sets)
            pitches = pitches_for_slot(
                source_pitch_sets,
                slot,
                raise_odd_loop_last=raise_odd_loop_last,
            )
            if not pitches:
                continue
            duration = _strum_onset_duration(
                articulation,
                onset_index=oi,
                onset_ticks=onset_ticks,
                ticks_per_bar=TICKS_PER_BAR,
                rng=rng,
                rest_indices=rest_indices,
            )
            if duration is None:
                continue
            # Don't ring across a half-bar chord change.
            if cpb >= 2 and tick < TICKS_PER_BAR // 2:
                duration = min(duration, (TICKS_PER_BAR // 2) - tick)
            onset_time = bar * TICKS_PER_BAR + tick
            if use_power and power_chord_onsets is not None:
                power_chord_onsets.add(onset_time)
            add_chord(
                track,
                pitches,
                time=onset_time,
                duration=max(1, duration),
            )


def generate_chord_strum(
    *,
    key: str,
    quality: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
    articulation: str = "solid",
    attacks_per_bar: int | None = None,
    placement: str | None = None,
    onset_ticks: list[int] | None = None,
    rng: random.Random | None = None,
) -> muspy.Music:
    rng = rng or random.Random()
    track = make_guitar_track("chord_strum")
    pitches = chord_pitches(key, quality)
    if onset_ticks is None:
        n = attacks_per_bar if attacks_per_bar is not None else choose_attacks_per_bar(rng)
        place = placement if placement is not None else choose_placement(rng)
        onset_ticks = resolve_onset_ticks(n, place, rng=rng)
    _write_strum_bars(
        track,
        pitch_sets=[pitches],
        bars=bars,
        bars_per_chord=1,
        onset_ticks=onset_ticks,
        articulation=articulation,
        rng=rng,
    )
    return build_music(track, bars=bars, tempo=float(bpm))


def generate_arpeggio(
    *,
    key: str,
    quality: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
) -> muspy.Music:
    track = make_guitar_track("arpeggio")
    pitches = chord_pitches(key, quality)
    notes_per_bar = len(pitches)
    step = TICKS_PER_BAR // max(notes_per_bar, 1)
    duration = max(step - 1, 2)

    for bar in range(bars):
        for index, pitch in enumerate(pitches):
            time = bar * TICKS_PER_BAR + index * step
            add_note(track, time=time, pitch=pitch, duration=duration)

    return build_music(track, bars=bars, tempo=float(bpm))


def generate_scale(
    *,
    key: str,
    mode: str,
    bpm: int,
    direction: str,
    bars: int = DEFAULT_BARS,
) -> muspy.Music:
    track = make_guitar_track(f"scale_{direction}")
    pitches = scale_pitches(key, mode)
    if direction == "scale_down":
        pitches = list(reversed(pitches))

    step = BEAT_TICKS
    duration = max(step - 1, 2)
    time = 0
    pitch_index = 0

    while time < bars * TICKS_PER_BAR:
        pitch = pitches[pitch_index % len(pitches)]
        add_note(track, time=time, pitch=pitch, duration=duration)
        pitch_index += 1
        time += step

    return build_music(track, bars=bars, tempo=float(bpm))


def generate_progression_strum(
    *,
    spec: ProgressionSpec,
    key: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
    bars_per_chord: int = 1,
    chords_per_bar: int = 1,
    raise_odd_loop_last: bool = True,
    articulation: str = "solid",
    attacks_per_bar: int | None = None,
    placement: str | None = None,
    onset_ticks: list[int] | None = None,
    power_chord_probability: float = BACKING_POWER_CHORD_PROBABILITY,
    power_chord_onsets: set[int] | None = None,
    rng: random.Random | None = None,
) -> muspy.Music:
    """進行ストロークを生成。既定で発音の70%をパワーコードにする。"""
    rng = rng or random.Random()
    track = make_guitar_track(f"prog_strum_{spec.name}")
    pitch_sets = progression_chord_pitch_sets(spec, key)
    power_pitch_sets = [
        power_chord_pitches(root)
        for root, _ in resolve_progression_chords(spec, key)
    ]
    if onset_ticks is None:
        n = attacks_per_bar if attacks_per_bar is not None else choose_attacks_per_bar(rng)
        place = placement if placement is not None else choose_placement(rng)
        onset_ticks = resolve_onset_ticks(n, place, rng=rng)
    _write_strum_bars(
        track,
        pitch_sets=pitch_sets,
        power_pitch_sets=power_pitch_sets,
        power_chord_probability=power_chord_probability,
        power_chord_onsets=power_chord_onsets,
        bars=bars,
        bars_per_chord=bars_per_chord,
        onset_ticks=onset_ticks,
        articulation=articulation,
        rng=rng,
        chords_per_bar=chords_per_bar,
        raise_odd_loop_last=raise_odd_loop_last,
    )
    return build_music(track, bars=bars, tempo=float(bpm))


def sample_backing_power_chord_onsets(
    *,
    bars: int,
    bpm: float,
    rng: random.Random,
    probability: float = BACKING_POWER_CHORD_PROBABILITY,
) -> set[int]:
    """lead条件用に、現行backing分布相当のパワーコードattack時刻を標本化する。"""
    attacks = sample_attacks_for_bpm(bpm, rng)
    placement = choose_placement(rng)
    ticks = resolve_onset_ticks(attacks, placement, rng=rng)
    return {
        bar * TICKS_PER_BAR + tick
        for bar in range(bars)
        for tick in ticks
        if rng.random() < probability
    }


def generate_progression_arpeggio(
    *,
    spec: ProgressionSpec,
    key: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
    bars_per_chord: int = 1,
    chords_per_bar: int = 1,
    raise_odd_loop_last: bool = True,
) -> muspy.Music:
    """進行に沿って、コードごとにアルペジオ伴奏を生成する。"""
    track = make_guitar_track(f"prog_arp_{spec.name}")
    pitch_sets = progression_chord_pitch_sets(spec, key)
    cpb = max(1, int(chords_per_bar))
    bpc = 1 if cpb >= 2 else max(1, int(bars_per_chord))

    for bar in range(bars):
        # One arpeggio pass per chord slice in the bar.
        slices = cpb if cpb >= 2 else 1
        slice_len = TICKS_PER_BAR // slices
        for s in range(slices):
            tick0 = s * slice_len
            slot = chord_slot(bar, tick0, bars_per_chord=bpc, chords_per_bar=cpb)
            pitches = pitches_for_slot(
                pitch_sets, slot, raise_odd_loop_last=raise_odd_loop_last
            )
            if not pitches:
                continue
            notes_per = len(pitches)
            step = max(1, slice_len // max(notes_per, 1))
            duration = max(step - 1, 1)
            for index, pitch in enumerate(pitches):
                time = bar * TICKS_PER_BAR + tick0 + index * step
                if time >= bar * TICKS_PER_BAR + tick0 + slice_len:
                    break
                add_note(track, time=time, pitch=pitch, duration=duration)

    return build_music(track, bars=bars, tempo=float(bpm))


_LEAD_RHYTHM_MOTIFS: tuple[tuple[tuple[int, int], ...], ...] = (
    # (2小節内tick, duration)。末尾を空けてフレーズ間に息継ぎを作る。
    ((0, 4), (6, 2), (10, 2), (12, 4), (20, 2), (23, 1), (26, 2)),
    ((0, 2), (3, 1), (6, 4), (12, 2), (16, 4), (22, 2), (27, 1)),
    ((0, 6), (8, 2), (11, 1), (14, 2), (20, 4), (26, 2)),
)

_LEAD_CONTOURS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 2, 1, 0),
    (0, 2, 1, 3, 4, 2, 1),
    (0, -1, 1, 2, 1, -1, 0),
)


def _nearest_chord_scale_index(
    scale: list[int],
    chord_pcs: set[int],
    around: int,
) -> int:
    candidates = [i for i, pitch in enumerate(scale) if pitch % 12 in chord_pcs]
    if not candidates:
        return max(0, min(len(scale) - 1, around))
    return min(candidates, key=lambda i: abs(i - around))


def _fit_contour(contour: tuple[int, ...], count: int) -> list[int]:
    """音数に合わせて輪郭を循環せず補間し、末尾は着地点へ戻す。"""
    if count <= len(contour):
        return list(contour[:count])
    out = list(contour)
    while len(out) < count:
        out.insert(-1, out[-2])
    return out


def generate_progression_lead(
    *,
    spec: ProgressionSpec,
    key: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
    bars_per_chord: int = 1,
    rng: random.Random | None = None,
    rest_prob: float = 0.12,
    leap_prob: float = 0.15,
    blocked_power_onsets: set[int] | frozenset[int] | None = None,
    power_chord_phrase_probability: float = LEAD_POWER_CHORD_PHRASE_PROBABILITY,
) -> muspy.Music:
    """反復・変奏・息継ぎのある、進行スケール上の単音リードを生成する。

    2小節モチーフを基本単位にし、A → A' → B → cadence の流れを作る。
    onset は四分・八分・付点・16分相当を混ぜ、各モチーフ末尾に休符を残す。
    A' はAの輪郭とリズムを再利用しつつ、終止音だけ現在コードへ着地させる。
    """
    rng = rng or random.Random()
    track = make_guitar_track(f"prog_lead_{spec.name}")
    scale = scale_pitches(key, spec.mode, base_octave=4)
    if not scale:
        scale = scale_pitches(key, spec.mode)
    pitch_sets = progression_chord_pitch_sets(spec, key)
    chord_pc_sets = [set(p % 12 for p in ps) for ps in pitch_sets]
    power_pitch_sets = [
        power_chord_pitches(root)
        for root, _ in resolve_progression_chords(spec, key)
    ]
    blocked = blocked_power_onsets or frozenset()

    phrase_ticks = TICKS_PER_BAR * 2
    center = len(scale) // 2
    base_rhythm = rng.choice(_LEAD_RHYTHM_MOTIFS)
    base_contour = rng.choice(_LEAD_CONTOURS)

    phrase_count = (bars + 1) // 2
    for phrase_index in range(phrase_count):
        phrase_start = phrase_index * phrase_ticks
        if phrase_start >= bars * TICKS_PER_BAR:
            break

        role = phrase_index % 4
        if role in (0, 1):  # A / A': 同じ形を認識できる程度に反復
            rhythm = list(base_rhythm)
            contour = _fit_contour(base_contour, len(rhythm))
            if role == 1 and len(contour) >= 3:
                contour[-2] += rng.choice((-1, 1))
        elif role == 2:  # B: 別の輪郭
            rhythm = list(rng.choice(_LEAD_RHYTHM_MOTIFS))
            contour = _fit_contour(rng.choice(_LEAD_CONTOURS), len(rhythm))
            if rng.random() < leap_prob and len(contour) >= 4:
                contour[len(contour) // 2] += rng.choice((-2, 2))
        else:  # cadence: 音数を減らし、長い着地と息継ぎ
            rhythm = [(0, 4), (6, 2), (12, 4), (20, 6)]
            contour = [0, 1, -1, 0]

        first_bar = phrase_start // TICKS_PER_BAR
        first_chord = chord_pc_sets[
            (first_bar // bars_per_chord) % len(chord_pc_sets)
        ]
        anchor = _nearest_chord_scale_index(scale, first_chord, center)

        kept: list[tuple[int, int, int]] = []
        for event_index, ((offset, duration), delta) in enumerate(
            zip(rhythm, contour)
        ):
            pos = phrase_start + offset
            if pos >= bars * TICKS_PER_BAR:
                continue
            # 内部音だけを少量抜き、モチーフの頭と終止は必ず残す。
            if (
                0 < event_index < len(rhythm) - 1
                and rng.random() < rest_prob
            ):
                continue
            note_index = max(0, min(len(scale) - 1, anchor + delta))
            kept.append((pos, duration, note_index))

        if not kept:
            continue

        # フレーズ終止は、その時点のコードトーンへ最短で着地させる。
        last_pos, last_duration, last_index = kept[-1]
        last_bar = last_pos // TICKS_PER_BAR
        last_chord = chord_pc_sets[
            (last_bar // bars_per_chord) % len(chord_pc_sets)
        ]
        kept[-1] = (
            last_pos,
            last_duration,
            _nearest_chord_scale_index(scale, last_chord, last_index),
        )

        power_pos: int | None = None
        if rng.random() < power_chord_phrase_probability:
            # backingのpower chordと同時attackしない。長さ2tick以上を優先。
            candidates = [
                pos
                for pos, duration, _ in kept
                if pos not in blocked and duration >= 2
            ]
            if candidates:
                power_pos = candidates[-1] if role == 3 else rng.choice(candidates)

        for pos, duration, note_index in kept:
            if pos == power_pos:
                bar = pos // TICKS_PER_BAR
                chord_index = (bar // bars_per_chord) % len(power_pitch_sets)
                for pitch in power_pitch_sets[chord_index]:
                    add_note(
                        track,
                        time=pos,
                        pitch=pitch,
                        duration=duration,
                        velocity=DEFAULT_VELOCITY,
                    )
            else:
                add_note(
                    track,
                    time=pos,
                    pitch=scale[note_index],
                    duration=duration,
                    velocity=DEFAULT_VELOCITY,
                )

    return build_music(track, bars=bars, tempo=float(bpm))


def choose_pattern(rng: random.Random) -> str:
    patterns = list(PATTERN_WEIGHTS.keys())
    weights = [PATTERN_WEIGHTS[p] for p in patterns]
    return rng.choices(patterns, weights=weights, k=1)[0]


def choose_bars(rng: random.Random, *, fixed_bars: int | None = None) -> int:
    if fixed_bars is not None:
        return fixed_bars
    return rng.choice(BAR_LENGTH_CHOICES)


def choose_progression(rng: random.Random) -> ProgressionSpec:
    weights = [spec.weight for spec in PROGRESSIONS]
    return rng.choices(list(PROGRESSIONS), weights=weights, k=1)[0]


def choose_key_for_progression(rng: random.Random, spec: ProgressionSpec) -> str:
    from .constants import KEYS

    return rng.choice(KEYS)


def generate_random_phrase(
    *,
    rng: random.Random,
    bpm: int | None = None,
    bars: int | None = None,
) -> tuple[muspy.Music, dict]:
    from .constants import BPM_RANGE, CHORD_QUALITIES, KEYS, SCALE_MODES

    pattern = choose_pattern(rng)
    bars = choose_bars(rng, fixed_bars=bars)
    bpm_value = bpm if bpm is not None else rng.randint(*BPM_RANGE)

    if pattern in ("progression_strum", "progression_arpeggio"):
        spec = choose_progression(rng)
        key = choose_key_for_progression(rng, spec)
        # ~25% half-bar changes; then bars_per_chord stays 1
        chords_per_bar = rng.choice((1, 1, 1, 2))
        bars_per_chord = 1 if chords_per_bar >= 2 else rng.choice((1, 1, 1, 2))
        raise_odd = rng.random() < 0.85
        if pattern == "progression_strum":
            articulation = choose_strum_articulation(rng)
            attacks = sample_attacks_for_bpm(bpm_value, rng)
            placement = choose_placement(rng)
            onset_ticks = resolve_onset_ticks(attacks, placement, rng=rng)
            music = generate_progression_strum(
                spec=spec,
                key=key,
                bpm=bpm_value,
                bars=bars,
                bars_per_chord=bars_per_chord,
                chords_per_bar=chords_per_bar,
                raise_odd_loop_last=raise_odd,
                articulation=articulation,
                onset_ticks=onset_ticks,
                rng=rng,
            )
            return music, {
                "pattern": pattern,
                "progression": spec.name,
                "family": spec.family,
                "mode": spec.mode,
                "key": key,
                "bars_per_chord": bars_per_chord,
                "chords_per_bar": chords_per_bar,
                "raise_odd_loop_last": raise_odd,
                "articulation": articulation,
                "attacks_per_bar": attacks,
                "placement": placement,
                "onset_ticks": onset_ticks,
                "power_chord_probability": BACKING_POWER_CHORD_PROBABILITY,
                "bpm": bpm_value,
                "bars": bars,
            }
        music = generate_progression_arpeggio(
            spec=spec,
            key=key,
            bpm=bpm_value,
            bars=bars,
            bars_per_chord=bars_per_chord,
            chords_per_bar=chords_per_bar,
            raise_odd_loop_last=raise_odd,
        )
        pitch_sets = progression_chord_pitch_sets(spec, key)
        arp_notes = len(pitch_sets[0]) if pitch_sets else 0
        return music, {
            "pattern": pattern,
            "progression": spec.name,
            "family": spec.family,
            "mode": spec.mode,
            "key": key,
            "bars_per_chord": bars_per_chord,
            "chords_per_bar": chords_per_bar,
            "raise_odd_loop_last": raise_odd,
            "arp_notes_per_bar": arp_notes,
            "bpm": bpm_value,
            "bars": bars,
        }

    key = rng.choice(KEYS)

    if pattern == "chord_strum":
        quality = rng.choice(CHORD_QUALITIES)
        articulation = choose_strum_articulation(rng)
        attacks = sample_attacks_for_bpm(bpm_value, rng)
        placement = choose_placement(rng)
        onset_ticks = resolve_onset_ticks(attacks, placement, rng=rng)
        music = generate_chord_strum(
            key=key,
            quality=quality,
            bpm=bpm_value,
            bars=bars,
            articulation=articulation,
            onset_ticks=onset_ticks,
            rng=rng,
        )
        meta = {
            "pattern": pattern,
            "key": key,
            "quality": quality,
            "articulation": articulation,
            "attacks_per_bar": attacks,
            "placement": placement,
            "onset_ticks": onset_ticks,
            "bpm": bpm_value,
            "bars": bars,
        }
    elif pattern == "arpeggio":
        quality = rng.choice(CHORD_QUALITIES)
        music = generate_arpeggio(key=key, quality=quality, bpm=bpm_value, bars=bars)
        meta = {
            "pattern": pattern,
            "key": key,
            "quality": quality,
            "arp_notes_per_bar": len(chord_pitches(key, quality)),
            "bpm": bpm_value,
            "bars": bars,
        }
    else:
        mode = rng.choice(SCALE_MODES)
        music = generate_scale(
            key=key,
            mode=mode,
            bpm=bpm_value,
            direction=pattern,
            bars=bars,
        )
        meta = {
            "pattern": pattern,
            "key": key,
            "mode": mode,
            "bpm": bpm_value,
            "bars": bars,
        }

    return music, meta
