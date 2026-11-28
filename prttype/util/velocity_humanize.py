"""拍アクセント＋曲シード型の velocity 後処理（学習なし）。

patch_to_midi が全ノート 80 固定なので、保存後に強弱を振る。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import muspy

VelocityStyle = Literal["backbeat_rock", "flat_accents", "crescendo"]
VELOCITY_STYLES: tuple[VelocityStyle, ...] = (
    "backbeat_rock",
    "flat_accents",
    "crescendo",
)

# GM drum
_KICK = {35, 36}
_SNARE = {37, 38, 40}
_HAT = {42, 44, 46}
_CRASH = {49, 57}


def _clamp(v: int) -> int:
    return int(max(1, min(127, v)))


def _beat_slot(time: int, resolution: int) -> str:
    """resolution=4 → 1拍4tick。down / and / off を返す。"""
    tpb = max(int(resolution), 1)
    bar = 4 * tpb
    pos = int(time) % bar
    beat = pos // tpb
    sub = pos % tpb
    if sub == 0:
        return "down" if beat in (0, 2) else "back"  # 1,3拍はバックビート寄り
    if sub == tpb // 2:
        return "and"
    return "off"


def _tonal_base(style: VelocityStyle, slot: str, *, bar_frac: float, rng: random.Random) -> int:
    if style == "backbeat_rock":
        table = {"down": (88, 102), "back": (96, 112), "and": (68, 82), "off": (52, 68)}
    elif style == "flat_accents":
        table = {"down": (78, 88), "back": (80, 90), "and": (74, 84), "off": (70, 80)}
        # たまにだけ突く
        if slot in ("down", "back") and rng.random() < 0.18:
            return _clamp(rng.randint(100, 118))
    else:  # crescendo: 区間位置で底上げ
        table = {"down": (72, 86), "back": (76, 90), "and": (60, 74), "off": (48, 62)}
    lo, hi = table[slot]
    base = rng.randint(lo, hi)
    if style == "crescendo":
        base += int(22 * bar_frac)
    return _clamp(base)


def _drum_vel(pitch: int, slot: str, style: VelocityStyle, rng: random.Random) -> int:
    if pitch in _CRASH:
        return _clamp(rng.randint(100, 122))
    if pitch in _KICK:
        lo, hi = (92, 112) if slot == "down" else (78, 98)
    elif pitch in _SNARE:
        lo, hi = (98, 118) if slot in ("back", "down") else (70, 90)
    elif pitch in _HAT:
        lo, hi = (58, 78) if slot in ("down", "back") else (42, 62)
    else:
        lo, hi = (70, 95)
    v = rng.randint(lo, hi)
    if style == "flat_accents":
        v = int(0.55 * v + 0.45 * 80)
    return _clamp(v)


def choose_velocity_style(rng: random.Random) -> VelocityStyle:
    return rng.choice(list(VELOCITY_STYLES))


def humanize_music_velocities(
    music: muspy.Music,
    *,
    seed: int | None = None,
    style: VelocityStyle | None = None,
) -> VelocityStyle:
    """muspy.Music の全トラック velocity をスタイルに応じて振り直す。"""
    rng = random.Random(seed)
    chosen: VelocityStyle = style if style in VELOCITY_STYLES else choose_velocity_style(rng)
    res = max(int(music.resolution or 4), 1)
    max_t = 0
    for tr in music.tracks:
        for n in tr.notes:
            max_t = max(max_t, int(n.time))
    max_t = max(max_t, 1)

    for tr in music.tracks:
        name = (tr.name or "").lower()
        is_drum = bool(tr.is_drum) or name == "drums"
        notes = sorted(tr.notes, key=lambda n: (n.time, n.pitch))
        for i, n in enumerate(notes):
            slot = _beat_slot(int(n.time), res)
            bar_frac = float(n.time) / float(max_t)
            if is_drum:
                n.velocity = _drum_vel(int(n.pitch), slot, chosen, rng)
                continue
            v = _tonal_base(chosen, slot, bar_frac=bar_frac, rng=rng)
            # フレーズ先頭っぽい（直前が空き）を少し押す
            if i == 0 or (int(n.time) - int(notes[i - 1].time) >= 2 * res):
                v = _clamp(v + rng.randint(4, 12))
            # lead は一段前に（聴感で ~1.5 倍がちょうどよい）
            if name == "lead":
                v = _clamp(int(round(v * 1.5)))
            elif name == "backing":
                v = _clamp(int(round(v * 0.92)))
            elif name == "bass":
                v = _clamp(int(round(v * 0.95)))
            n.velocity = v
    return chosen


def _bpm_from_pretty(src: object) -> float:
    if src.get_tempo_changes()[1].size:
        return float(src.get_tempo_changes()[1][0])
    return 120.0


def _time_to_tick(t: float, *, bpm: float, resolution: int) -> int:
    """秒 → プロジェクト互換 tick（1拍 = resolution tick）。"""
    beat_sec = 60.0 / max(float(bpm), 1e-6)
    return int(round(float(t) / beat_sec * max(int(resolution), 1)))


def humanize_midi_file(
    midi_path: Path | str,
    *,
    seed: int | None = None,
    style: VelocityStyle | None = None,
    out_path: Path | str | None = None,
) -> tuple[Path, VelocityStyle]:
    """velocity だけ振り直す。pitch bend 等は pretty_midi 経由で保持する。

    以前の muspy read/write は pitch bend を落としていた。
    """
    import pretty_midi as pm

    path = Path(midi_path)
    src = pm.PrettyMIDI(str(path))
    rng = random.Random(seed)
    chosen: VelocityStyle = style if style in VELOCITY_STYLES else choose_velocity_style(rng)
    bpm = _bpm_from_pretty(src)
    res = 4  # form / patch 系の既定 resolution
    max_t = 0
    for inst in src.instruments:
        for n in inst.notes:
            max_t = max(max_t, _time_to_tick(n.start, bpm=bpm, resolution=res))
    max_t = max(max_t, 1)

    for inst in src.instruments:
        name = (inst.name or "").lower()
        is_drum = bool(inst.is_drum) or name == "drums"
        notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
        for i, n in enumerate(notes):
            tick = _time_to_tick(n.start, bpm=bpm, resolution=res)
            slot = _beat_slot(tick, res)
            bar_frac = float(tick) / float(max_t)
            if is_drum:
                n.velocity = _drum_vel(int(n.pitch), slot, chosen, rng)
                continue
            v = _tonal_base(chosen, slot, bar_frac=bar_frac, rng=rng)
            if i == 0 or (
                _time_to_tick(n.start, bpm=bpm, resolution=res)
                - _time_to_tick(notes[i - 1].start, bpm=bpm, resolution=res)
                >= 2 * res
            ):
                v = _clamp(v + rng.randint(4, 12))
            if name == "lead":
                v = _clamp(int(round(v * 1.5)))
            elif name == "backing":
                v = _clamp(int(round(v * 0.92)))
            elif name == "bass":
                v = _clamp(int(round(v * 0.95)))
            n.velocity = v

    out = Path(out_path) if out_path is not None else path
    out.parent.mkdir(parents=True, exist_ok=True)
    src.write(str(out))
    return out, chosen


__all__ = [
    "VELOCITY_STYLES",
    "VelocityStyle",
    "choose_velocity_style",
    "humanize_music_velocities",
    "humanize_midi_file",
]
