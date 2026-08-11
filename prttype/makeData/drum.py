"""ドラムパターン合成（12 型カタログ・骨格固定）。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import muspy

from .builder import add_note, build_music
from .constants import (
    BEAT_TICKS,
    BEAT_TYPES,
    DEFAULT_BARS,
    DEFAULT_VELOCITY,
    DRUM_CLOSED_HAT,
    DRUM_CRASH_1,
    DRUM_CRASH_2,
    DRUM_KICK,
    DRUM_OPEN_HAT,
    DRUM_SIDE_STICK,
    DRUM_SNARE,
    DRUM_TOM_FILLS,
    TICKS_PER_BAR,
)

BeatType = Literal[
    "eight_basic",
    "four_floor",
    "sixteen_basic",
    "sixteen_funk",
    "halftime",
    "halftime_shuffle",
    "ballad_sparse",
    "shuffle_eight",
    "disco",
    "tresillo",
    "reggae",
    "metal_double",
]

HAT_8 = (0, 2, 4, 6, 8, 10, 12, 14)
HAT_4 = (0, 4, 8, 12)
HAT_16 = tuple(range(16))
HAT_SHUFFLE = (0, 3, 4, 7, 8, 11, 12, 15)
HAT_DISCO_OH = (2, 6, 10, 14)


@dataclass(frozen=True)
class _BeatGrid:
    kick: tuple[int, ...]
    snare: tuple[int, ...]
    hat: tuple[int, ...]
    hat_pitch: int = DRUM_CLOSED_HAT
    snare_pitch: int = DRUM_SNARE
    kick_vel: tuple[int, int] = (85, 105)
    snare_vel: tuple[int, int] = (80, 100)
    hat_vel: tuple[int, int] = (65, 85)
    fill_prob: float = 0.85
    crash_prob: float = 0.7
    open_hat_tail_prob: float = 0.0
    ghost_snare: bool = False
    disco_open_hat: bool = False
    metal_crash: bool = False


_BEAT_GRIDS: dict[str, _BeatGrid] = {
    "eight_basic": _BeatGrid(
        kick=(0, 8),
        snare=(4, 12),
        hat=HAT_8,
        open_hat_tail_prob=0.35,
    ),
    "four_floor": _BeatGrid(
        kick=(0, 4, 8, 12),
        snare=(4, 12),
        hat=HAT_8,
        kick_vel=(95, 115),
        crash_prob=0.85,
    ),
    "sixteen_basic": _BeatGrid(
        kick=(0, 8),
        snare=(4, 12),
        hat=HAT_16,
        hat_vel=(55, 75),
    ),
    "sixteen_funk": _BeatGrid(
        kick=(0, 3, 8, 10),
        snare=(4, 12),
        hat=HAT_16,
        hat_vel=(55, 75),
        ghost_snare=True,
        fill_prob=0.7,
    ),
    "halftime": _BeatGrid(
        kick=(0,),
        snare=(8,),
        hat=HAT_8,
        fill_prob=0.15,
        crash_prob=0.4,
    ),
    "halftime_shuffle": _BeatGrid(
        kick=(0,),
        snare=(8,),
        hat=HAT_SHUFFLE,
        fill_prob=0.15,
        crash_prob=0.4,
    ),
    "ballad_sparse": _BeatGrid(
        kick=(0,),
        snare=(4, 12),
        hat=HAT_4,
        snare_pitch=DRUM_SIDE_STICK,
        kick_vel=(70, 85),
        snare_vel=(50, 65),
        hat_vel=(50, 65),
        fill_prob=0.0,
        crash_prob=0.25,
    ),
    "shuffle_eight": _BeatGrid(
        kick=(0, 8),
        snare=(4, 12),
        hat=HAT_SHUFFLE,
    ),
    "disco": _BeatGrid(
        kick=(0, 4, 8, 12),
        snare=(4, 12),
        hat=HAT_DISCO_OH,
        hat_pitch=DRUM_OPEN_HAT,
        kick_vel=(95, 115),
        disco_open_hat=True,
        crash_prob=0.8,
    ),
    "tresillo": _BeatGrid(
        kick=(0, 3, 6, 8, 11, 14),
        snare=(4, 12),
        hat=HAT_8,
        kick_vel=(90, 110),
    ),
    "reggae": _BeatGrid(
        kick=(8,),
        snare=(8,),
        hat=HAT_4,
        snare_pitch=DRUM_SIDE_STICK,
        kick_vel=(80, 95),
        snare_vel=(70, 85),
        hat_vel=(55, 70),
        fill_prob=0.0,
        crash_prob=0.2,
    ),
    "metal_double": _BeatGrid(
        kick=HAT_8,
        snare=(4, 12),
        hat=HAT_8,
        kick_vel=(100, 120),
        snare_vel=(95, 115),
        fill_prob=0.9,
        crash_prob=1.0,
        metal_crash=True,
    ),
}


def make_drum_track(name: str = "Drums") -> muspy.Track:
    return muspy.Track(program=0, is_drum=True, name=name)


def extract_kick_times(music: muspy.Music) -> set[int]:
    """ドラムトラックからキック onset tick の集合を返す。"""
    times: set[int] = set()
    for track in music.tracks:
        if not track.is_drum:
            continue
        for note in track.notes:
            if int(note.pitch) == DRUM_KICK:
                times.add(int(note.time))
    return times


def beat_type_to_id(beat_type: str) -> int:
    if beat_type not in BEAT_TYPES:
        raise ValueError(f"unknown beat_type: {beat_type!r}")
    return BEAT_TYPES.index(beat_type)


def _hit(
    track: muspy.Track,
    *,
    time: int,
    pitch: int,
    velocity: int = DEFAULT_VELOCITY,
    duration: int = 1,
) -> None:
    add_note(track, time=time, pitch=pitch, duration=duration, velocity=velocity)


def _add_tom_fill(
    track: muspy.Track,
    *,
    bar_t: int,
    dense: bool,
    rng: random.Random,
) -> None:
    """8小節ブロック最終小節の後半に tom fill。"""
    start = bar_t + (BEAT_TICKS if dense else 2 * BEAT_TICKS)
    toms = list(DRUM_TOM_FILLS)
    step = 1 if dense else 2
    vel = 100 if dense else 85
    t = start
    while t < bar_t + TICKS_PER_BAR:
        _hit(track, time=t, pitch=rng.choice(toms), velocity=vel)
        t += step


def generate_drum_pattern(
    *,
    bpm: float = 120.0,
    bars: int = DEFAULT_BARS,
    beat_type: str = "eight_basic",
    section_start_crash: bool = True,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> muspy.Music:
    """GM キットのドラムパターンを生成する。

    kick/snare 骨格は beat_type ごとに固定。velocity・ghost・crash・fill のみ確率変動。
    """
    if beat_type not in _BEAT_GRIDS:
        raise ValueError(f"unknown beat_type: {beat_type!r}")
    grid = _BEAT_GRIDS[beat_type]
    rng = rng or random.Random(seed)
    track = make_drum_track()

    for bar in range(bars):
        bar_t = bar * TICKS_PER_BAR
        is_last = bar == bars - 1
        is_first = bar == 0

        if is_first and section_start_crash:
            if grid.metal_crash or rng.random() < grid.crash_prob:
                _hit(
                    track,
                    time=bar_t,
                    pitch=rng.choice((DRUM_CRASH_1, DRUM_CRASH_2)),
                    velocity=110 if grid.metal_crash else 95,
                )

        for tick in grid.kick:
            _hit(
                track,
                time=bar_t + tick,
                pitch=DRUM_KICK,
                velocity=rng.randint(*grid.kick_vel),
            )

        for tick in grid.snare:
            _hit(
                track,
                time=bar_t + tick,
                pitch=grid.snare_pitch,
                velocity=rng.randint(*grid.snare_vel),
            )

        if grid.ghost_snare:
            for beat in (1, 3):
                if rng.random() < 0.55:
                    _hit(
                        track,
                        time=bar_t + beat * BEAT_TICKS + 2,
                        pitch=rng.choice((DRUM_SNARE, DRUM_SIDE_STICK)),
                        velocity=rng.randint(35, 50),
                    )

        if grid.disco_open_hat:
            for tick in grid.hat:
                _hit(
                    track,
                    time=bar_t + tick,
                    pitch=DRUM_OPEN_HAT,
                    velocity=rng.randint(75, 95),
                )
            # 表拍は closed hat も薄く
            for tick in HAT_4:
                _hit(
                    track,
                    time=bar_t + tick,
                    pitch=DRUM_CLOSED_HAT,
                    velocity=rng.randint(60, 75),
                )
        else:
            hat_ticks = list(grid.hat)
            for i, tick in enumerate(hat_ticks):
                pitch = grid.hat_pitch
                if (
                    grid.open_hat_tail_prob > 0
                    and tick == hat_ticks[-1]
                    and rng.random() < grid.open_hat_tail_prob
                ):
                    pitch = DRUM_OPEN_HAT
                _hit(
                    track,
                    time=bar_t + tick,
                    pitch=pitch,
                    velocity=rng.randint(*grid.hat_vel),
                )

        if is_last and grid.fill_prob > 0 and rng.random() < grid.fill_prob:
            dense = beat_type in ("metal_double", "four_floor", "sixteen_funk", "disco")
            _add_tom_fill(track, bar_t=bar_t, dense=dense, rng=rng)

    return build_music(track, bars=bars, tempo=float(bpm))
