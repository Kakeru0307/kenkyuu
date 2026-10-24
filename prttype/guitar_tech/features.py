"""Per-note context features for technique classification."""

from __future__ import annotations

import numpy as np

from .carriers import NoteEvent

# pitch, dur, vel, iv_prev, iv_next, gap_prev, gap_next, rel_pos
NOTE_FEATURE_DIM = 8
FEATURE_DIM = NOTE_FEATURE_DIM  # apply モデル用エイリアス


def note_features(notes: list[NoteEvent]) -> np.ndarray:
    n = len(notes)
    if n == 0:
        return np.zeros((0, NOTE_FEATURE_DIM), dtype=np.float32)
    phrase_dur = max(notes[-1].end - notes[0].start, 1e-3)
    xs = np.zeros((n, NOTE_FEATURE_DIM), dtype=np.float32)
    for i, note in enumerate(notes):
        prev = notes[i - 1] if i > 0 else None
        nxt = notes[i + 1] if i + 1 < n else None
        xs[i, 0] = note.pitch / 127.0
        xs[i, 1] = min(note.duration / 2.0, 1.0)
        xs[i, 2] = note.velocity / 127.0
        xs[i, 3] = ((note.pitch - prev.pitch) / 12.0) if prev else 0.0
        xs[i, 4] = ((nxt.pitch - note.pitch) / 12.0) if nxt else 0.0
        xs[i, 5] = (
            min(max(0.0, note.start - prev.end) / 1.0, 1.0) if prev else 0.0
        )
        xs[i, 6] = (
            min(max(0.0, nxt.start - note.end) / 1.0, 1.0) if nxt else 0.0
        )
        xs[i, 7] = (note.start - notes[0].start) / phrase_dur
    return xs


def pretty_midi_to_notes(midi) -> list[NoteEvent]:
    notes: list[NoteEvent] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append(
                NoteEvent(
                    pitch=int(n.pitch),
                    start=float(n.start),
                    end=float(n.end),
                    velocity=int(n.velocity),
                )
            )
    notes.sort(key=lambda x: (x.start, x.pitch))
    return notes


def top_voice_indices(notes: list[NoteEvent]) -> list[int]:
    """同一 onset（1ms 丸め）ごとに最高 pitch のインデックスだけ残す。"""
    if not notes:
        return []
    by_start: dict[float, list[int]] = {}
    for i, n in enumerate(notes):
        key = round(n.start, 3)
        by_start.setdefault(key, []).append(i)
    chosen: list[int] = []
    for key in sorted(by_start):
        idxs = by_start[key]
        chosen.append(max(idxs, key=lambda j: notes[j].pitch))
    return chosen


def note_has_overlap(notes: list[NoteEvent], i: int) -> bool:
    n = notes[i]
    for j, m in enumerate(notes):
        if j == i:
            continue
        if m.start < n.end - 1e-6 and n.start < m.end - 1e-6:
            return True
    return False
