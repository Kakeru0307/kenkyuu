"""Band lock postprocess: drum grid snap, lead gap placement (no retrain)."""

from __future__ import annotations

import muspy

from makeData.constants import BEAT_TICKS, TICKS_PER_BAR
from makeData.drum import extract_kick_times


def collect_onsets(
    music: muspy.Music | None,
    *,
    drums_only: bool = False,
    tonal_only: bool = False,
) -> set[int]:
    if music is None:
        return set()
    out: set[int] = set()
    for track in music.tracks:
        if drums_only and not track.is_drum:
            continue
        if tonal_only and track.is_drum:
            continue
        for note in track.notes:
            out.add(int(note.time))
    return out


def build_drum_grid(
    drum_music: muspy.Music | None,
    *,
    bars: int,
    subdiv_ticks: int = 2,
) -> set[int]:
    """ドラム onset + 定間隔格子。subdiv 既定=8分（resolution=4 で 2tick）。"""
    grid = collect_onsets(drum_music, drums_only=True)
    step = max(1, int(subdiv_ticks))
    total = max(1, int(bars)) * TICKS_PER_BAR
    grid.update(range(0, total, step))
    # キックは必ず格子に含める（extract と二重でも可）
    grid |= extract_kick_times(drum_music) if drum_music is not None else set()
    return grid


def _nearest_grid(t: int, grid: set[int], *, max_shift: int) -> int | None:
    best: int | None = None
    best_d = max_shift + 1
    for g in grid:
        d = abs(int(g) - int(t))
        if d < best_d or (d == best_d and best is not None and g < best):
            best_d = d
            best = int(g)
    if best is None or best_d > max_shift:
        return None
    return best


def snap_music_to_grid(
    music: muspy.Music,
    grid: set[int],
    *,
    max_shift: int = 2,
    skip_drums: bool = True,
) -> muspy.Music:
    """非ドラムノートの onset を格子へ吸着（max_shift 以内のみ）。"""
    if not grid:
        return music
    for track in music.tracks:
        if skip_drums and track.is_drum:
            continue
        for note in track.notes:
            snapped = _nearest_grid(int(note.time), grid, max_shift=max_shift)
            if snapped is not None:
                note.time = snapped
        track.notes = sorted(track.notes, key=lambda n: (int(n.time), int(n.pitch)))
    return music


def shift_lead_to_gaps(
    lead_music: muspy.Music,
    blocked: set[int],
    *,
    collide_window: int = 0,
    search_radius: int = 2,
    bars: int | None = None,
) -> muspy.Music:
    """伴奏 onset と重なるリードを近傍の空き tick へずらす。"""
    if not blocked:
        return lead_music
    max_t = None if bars is None else max(0, int(bars) * TICKS_PER_BAR - 1)
    occupied = set(blocked)

    def free(t: int) -> bool:
        if t < 0:
            return False
        if max_t is not None and t > max_t:
            return False
        return all(abs(t - b) > collide_window for b in occupied)

    for track in lead_music.tracks:
        if track.is_drum:
            continue
        new_notes: list[muspy.Note] = []
        for note in sorted(track.notes, key=lambda n: (int(n.time), int(n.pitch))):
            t = int(note.time)
            if free(t):
                occupied.add(t)
                new_notes.append(note)
                continue
            placed = False
            for d in range(1, search_radius + 1):
                for cand in (t + d, t - d):
                    if free(cand):
                        note.time = cand
                        occupied.add(cand)
                        new_notes.append(note)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                # 空きが無ければそのまま（密度は thin 側）
                occupied.add(t)
                new_notes.append(note)
        track.notes = sorted(new_notes, key=lambda n: (int(n.time), int(n.pitch)))
    return lead_music


def boost_kick_aligned_bass(
    bass_music: muspy.Music,
    kick_times: set[int] | frozenset[int],
    *,
    velocity: int = 100,
    window: int = 0,
) -> muspy.Music:
    """キック同時刻のベースを前に出す。"""
    if not kick_times:
        return bass_music
    kicks = {int(t) for t in kick_times}
    for track in bass_music.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            t = int(note.time)
            if any(abs(t - k) <= window for k in kicks):
                note.velocity = max(int(note.velocity), int(velocity))
    return bass_music


def prune_off_kick_bass(
    bass_music: muspy.Music,
    kick_times: set[int] | frozenset[int],
    *,
    keep_beat_ticks: bool = True,
) -> muspy.Music:
    """キック近傍以外の弱いベースを間引き（拍頭は残す）。密度をキック軸へ。"""
    if not kick_times:
        return bass_music
    kicks = {int(t) for t in kick_times}
    for track in bass_music.tracks:
        if track.is_drum:
            continue
        kept: list[muspy.Note] = []
        for note in track.notes:
            t = int(note.time)
            on_kick = any(abs(t - k) <= 1 for k in kicks)
            on_beat = keep_beat_ticks and (t % BEAT_TICKS == 0)
            if on_kick or on_beat:
                kept.append(note)
            elif int(note.velocity) >= 100:
                kept.append(note)
        track.notes = sorted(kept, key=lambda n: (int(n.time), int(n.pitch)))
    return bass_music


__all__ = [
    "boost_kick_aligned_bass",
    "build_drum_grid",
    "collect_onsets",
    "prune_off_kick_bass",
    "shift_lead_to_gaps",
    "snap_music_to_grid",
]
