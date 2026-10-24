"""MIDI ↔ fixed-length frame tensor for technique clips."""

from __future__ import annotations

import numpy as np
import pretty_midi as pm

from .constants import FEATURE_DIM, N_FRAMES, PB_MAX


def midi_to_frames(
    midi_path: str | pm.PrettyMIDI,
    *,
    n_frames: int = N_FRAMES,
    transpose: int = 0,
    pb_scale: float = 1.0,
) -> np.ndarray:
    """Return float32 (n_frames, 3): pitch_norm, gate, pb_norm in [-1, 1] / [0,1]."""
    midi = (
        midi_path
        if isinstance(midi_path, pm.PrettyMIDI)
        else pm.PrettyMIDI(str(midi_path))
    )
    end = max(midi.get_end_time(), 1e-3)
    frames = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)
    # pitch / gate from notes
    for inst in midi.instruments:
        for note in inst.notes:
            p = int(note.pitch) + int(transpose)
            p = max(0, min(127, p))
            t0 = int(note.start / end * n_frames)
            t1 = max(t0 + 1, int(note.end / end * n_frames))
            t1 = min(n_frames, t1)
            for t in range(t0, t1):
                frames[t, 0] = p / 127.0
                frames[t, 1] = 1.0
    # pitch bend (use first non-drum instrument)
    pb_times: list[float] = []
    pb_vals: list[float] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for pb in inst.pitch_bends:
            pb_times.append(pb.time)
            pb_vals.append(float(pb.pitch) * float(pb_scale))
        if pb_times:
            break
    if pb_times:
        pb_times_a = np.asarray(pb_times, dtype=np.float64)
        pb_vals_a = np.clip(np.asarray(pb_vals, dtype=np.float64), -PB_MAX, PB_MAX)
        for t in range(n_frames):
            sec = (t + 0.5) / n_frames * end
            # stepwise hold last bend at/before sec
            idx = int(np.searchsorted(pb_times_a, sec, side="right") - 1)
            if idx < 0:
                val = 0.0
            else:
                val = pb_vals_a[idx]
            frames[t, 2] = float(np.clip(val / PB_MAX, -1.0, 1.0))
    return frames


def frames_to_midi(
    frames: np.ndarray,
    *,
    duration_sec: float = 4.0,
    program: int = 29,
    velocity: int = 96,
) -> pm.PrettyMIDI:
    """Decode (T,3) frames to a single-track PrettyMIDI."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != FEATURE_DIM:
        raise ValueError(f"frames shape must be (T,{FEATURE_DIM}), got {frames.shape}")
    n = frames.shape[0]
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=int(program), name="guitar_tech")
    dt = duration_sec / max(1, n)

    # merge consecutive gate frames with same pitch into notes
    t = 0
    while t < n:
        if frames[t, 1] < 0.55:
            t += 1
            continue
        pitch = int(round(float(frames[t, 0]) * 127.0))
        pitch = max(0, min(127, pitch))
        t0 = t
        t += 1
        while t < n and frames[t, 1] >= 0.55:
            p2 = int(round(float(frames[t, 0]) * 127.0))
            if p2 != pitch:
                break
            t += 1
        inst.notes.append(
            pm.Note(
                velocity=int(velocity),
                pitch=pitch,
                start=t0 * dt,
                end=max(t0 * dt + dt * 0.5, t * dt),
            )
        )

    # pitch bends every frame (skip if near zero streak)
    last = None
    for i in range(n):
        pb = int(round(float(frames[i, 2]) * PB_MAX))
        pb = max(-8192, min(8191, pb))
        if last is not None and abs(pb - last) < 8:
            continue
        inst.pitch_bends.append(pm.PitchBend(pitch=pb, time=i * dt))
        last = pb

    midi.instruments.append(inst)
    return midi
