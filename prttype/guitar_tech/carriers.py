"""Bare monophonic carrier phrases for apply-dataset synthesis."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pretty_midi as pm

# C major pentatonic + neighbors (pitch classes); transposed later
_SCALE_PCS = (0, 2, 4, 5, 7, 9, 10)


@dataclass
class NoteEvent:
    pitch: int
    start: float
    end: float
    velocity: int

    @property
    def duration(self) -> float:
        return max(1e-4, self.end - self.start)


def generate_carrier(
    rng: random.Random,
    *,
    duration_sec: float | None = None,
    program: int = 29,
) -> tuple[list[NoteEvent], pm.PrettyMIDI]:
    dur = float(duration_sec if duration_sec is not None else rng.uniform(3.5, 5.5))
    root = rng.choice([48, 50, 52, 53, 55, 57, 60])  # various registers
    n_notes = rng.randint(8, 18)
    density = rng.uniform(0.55, 0.95)

    times: list[float] = []
    t = 0.05
    step = dur / max(n_notes + 1, 2)
    for _ in range(n_notes):
        if rng.random() > density:
            t += step * rng.uniform(0.5, 1.2)
            continue
        times.append(t)
        t += step * rng.uniform(0.6, 1.4)
        if t >= dur - 0.05:
            break
    if not times:
        times = [0.1, 0.5, 1.0]

    notes: list[NoteEvent] = []
    pitch = root + rng.choice(_SCALE_PCS)
    for i, start in enumerate(times):
        if i + 1 < len(times):
            max_end = times[i + 1] - 0.01
        else:
            max_end = min(dur - 0.02, start + rng.uniform(0.15, 0.9))
        # mix short / long
        if rng.random() < 0.25:
            end = min(max_end, start + rng.uniform(0.4, 1.2))  # long
        else:
            end = min(max_end, start + rng.uniform(0.08, 0.35))
        end = max(end, start + 0.04)
        interval = rng.choice([-3, -2, -1, 0, 0, 1, 1, 2, 3, 4])
        pitch = int(max(40, min(84, pitch + interval)))
        # snap toward scale
        pc = (pitch - root) % 12
        if pc not in _SCALE_PCS:
            pitch = root + min(_SCALE_PCS, key=lambda x: abs(x - pc))
            while pitch < 40:
                pitch += 12
            while pitch > 84:
                pitch -= 12
        vel = int(rng.randint(70, 110))
        notes.append(NoteEvent(pitch=pitch, start=start, end=end, velocity=vel))

    midi = notes_to_midi(notes, program=program)
    return notes, midi


def notes_to_midi(notes: list[NoteEvent], *, program: int = 29) -> pm.PrettyMIDI:
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=program, name="carrier")
    for n in notes:
        inst.notes.append(
            pm.Note(
                velocity=int(n.velocity),
                pitch=int(n.pitch),
                start=float(n.start),
                end=float(n.end),
            )
        )
    midi.instruments.append(inst)
    return midi
