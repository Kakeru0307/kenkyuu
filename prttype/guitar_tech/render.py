"""Teacher renderer: bare notes + tech/params → target MIDI + labels."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import pretty_midi as pm

from .carriers import NoteEvent
from .constants import PB_MAX, PB_TECHS
from .features import note_has_overlap
from .params import TECH_WEIGHTS, sample_params

# semitone → pitch bend (assuming ±2 semitone PB range → ±8192)
_PB_PER_SEMITONE = PB_MAX / 2.0


@dataclass
class NoteLabel:
    note_index: int
    tech: str
    params: dict[str, Any]


def _clamp_pb(v: float) -> int:
    return int(max(-8192, min(8191, round(v))))


def _pb_semitone(semi: float) -> int:
    return _clamp_pb(semi * _PB_PER_SEMITONE)


def _reset_pb(inst: pm.Instrument, t: float) -> None:
    inst.pitch_bends.append(pm.PitchBend(pitch=0, time=float(t)))


def copy_timing(src: pm.PrettyMIDI, dst: pm.PrettyMIDI) -> None:
    """Copy tempo map / time signatures so DAW beat grid matches the source.

    Must also copy ``resolution``; otherwise write() rebuilds set_tempo from
    mismatched tick_scale×resolution and can overflow.
    """
    dst.resolution = int(src.resolution)
    if hasattr(src, "_tick_scales") and src._tick_scales:
        dst._tick_scales = [(int(tick), float(scale)) for tick, scale in src._tick_scales]
    dst.time_signature_changes = list(src.time_signature_changes)


def assign_techs(notes: list[NoteEvent], rng: random.Random) -> list[NoteLabel]:
    labels: list[NoteLabel] = []
    names = list(TECH_WEIGHTS.keys())
    weights = [TECH_WEIGHTS[n] for n in names]
    for i, n in enumerate(notes):
        # target ~50-70% none overall
        if rng.random() < 0.58:
            labels.append(NoteLabel(i, "none", {}))
            continue
        tech = rng.choices(names, weights=weights, k=1)[0]
        # eligibility soft filters
        if tech == "vibrato" and n.duration < 0.28:
            tech = "staccato" if n.duration < 0.15 else "none"
        if tech == "tremolo" and n.duration < 0.25:
            tech = "staccato"
        # slide notes-mode needs a following note; hammer/pull are self-contained
        if tech == "slide" and i + 1 >= len(notes):
            tech = "none"
        if tech == "palm_mute" and n.pitch > 64 and rng.random() < 0.5:
            tech = "none"
        params = sample_params(tech, rng) if tech != "none" else {}
        # slide notes mode without next → fall back to pb
        if tech == "slide" and params.get("mode") == "notes" and i + 1 >= len(notes):
            params["mode"] = "pb"
        uses_pb = tech in PB_TECHS and not (
            tech == "slide" and params.get("mode") == "notes"
        )
        if uses_pb and note_has_overlap(notes, i):
            tech, params = "none", {}
        labels.append(NoteLabel(i, tech, params))
    return labels


def render_target(
    notes: list[NoteEvent],
    labels: list[NoteLabel],
    *,
    program: int = 29,
    rng: random.Random | None = None,
    source: pm.PrettyMIDI | None = None,
    instrument_name: str | None = None,
) -> pm.PrettyMIDI:
    rng = rng or random.Random(0)
    midi = pm.PrettyMIDI()
    if source is not None:
        copy_timing(source, midi)
    inst_name = instrument_name or "Guitar"
    if instrument_name is None and source is not None:
        for src_inst in source.instruments:
            if not src_inst.is_drum:
                inst_name = src_inst.name or inst_name
                break
    inst = pm.Instrument(program=program, name=inst_name)
    label_by_i = {lb.note_index: lb for lb in labels}

    for i, n in enumerate(notes):
        lb = label_by_i.get(i) or NoteLabel(i, "none", {})
        tech, p = lb.tech, lb.params
        # PB is channel-global; never bend under overlapping notes
        uses_pb = tech in PB_TECHS and not (
            tech == "slide" and p.get("mode") == "notes"
        )
        if uses_pb and note_has_overlap(notes, i):
            tech, p = "none", {}

        if tech == "tremolo":
            _render_tremolo(inst, n, p, rng)
            continue
        if tech == "trill":
            _render_trill(inst, n, p, rng)
            continue
        if tech == "ghost":
            pitch = n.pitch - 12 if p.get("pitch_mode") == "drop12" else n.pitch
            length = min(n.duration, p["len_ms"] / 1000.0)
            inst.notes.append(
                pm.Note(
                    velocity=int(p["vel"]),
                    pitch=int(max(0, min(127, pitch))),
                    start=n.start,
                    end=n.start + max(0.005, length),
                )
            )
            continue
        if tech == "staccato":
            length = n.duration * p["len_ratio"]
            length = max(p["min_ms"] / 1000.0, min(p["max_ms"] / 1000.0, length))
            vel = int(max(1, min(127, n.velocity * p["vel_scale"])))
            inst.notes.append(
                pm.Note(velocity=vel, pitch=n.pitch, start=n.start, end=n.start + length)
            )
            continue
        if tech == "palm_mute":
            bias = 1.0 - p["low_bias"] * max(0.0, (60 - n.pitch) / 24.0)
            vel = int(max(1, min(127, n.velocity * p["vel_scale"] * bias)))
            end = n.start + n.duration * p["len_ratio"]
            inst.notes.append(
                pm.Note(velocity=vel, pitch=n.pitch, start=n.start, end=max(n.start + 0.02, end))
            )
            continue
        if tech == "hammer":
            grace_end = n.start + p["grace_ms"] / 1000.0
            grace_pitch = int(max(0, min(127, n.pitch - int(p["interval"]))))
            inst.notes.append(
                pm.Note(
                    velocity=int(n.velocity * p["grace_vel_scale"]),
                    pitch=grace_pitch,
                    start=n.start,
                    end=grace_end,
                )
            )
            inst.notes.append(
                pm.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=max(n.start, grace_end - p["overlap_ms"] / 1000.0),
                    end=n.end,
                )
            )
            continue
        if tech == "pull":
            high = int(max(0, min(127, n.pitch + int(round(p["from_semitone"])))))
            mid = n.start + p["from_ms"] / 1000.0
            inst.notes.append(
                pm.Note(
                    velocity=int(n.velocity * p["from_vel_scale"]),
                    pitch=high,
                    start=n.start,
                    end=min(mid, n.end),
                )
            )
            inst.notes.append(
                pm.Note(
                    velocity=int(n.velocity * p["dest_vel_scale"]),
                    pitch=n.pitch,
                    start=min(mid, n.end - 0.01),
                    end=n.end,
                )
            )
            continue
        if tech == "tap":
            vel = int(max(1, min(127, n.velocity * p["tap_vel_scale"])))
            if p.get("pattern") == "run":
                t = n.start
                for k in range(int(p["run_notes"])):
                    length = p["tap_len_ms"] / 1000.0
                    inst.notes.append(
                        pm.Note(
                            velocity=vel,
                            pitch=int(max(0, min(127, n.pitch + k))),
                            start=t,
                            end=min(n.end, t + length),
                        )
                    )
                    t += length
                    if t >= n.end:
                        break
            else:
                length = min(n.duration, p["tap_len_ms"] / 1000.0)
                inst.notes.append(
                    pm.Note(velocity=vel, pitch=n.pitch, start=n.start, end=n.start + length)
                )
            continue
        if tech == "slide" and p.get("mode") == "notes" and i + 1 < len(notes):
            nxt = notes[i + 1]
            direction = 1 if nxt.pitch >= n.pitch else -1
            start_p = int(n.pitch - direction * int(round(p["from_semitone"])))
            start_p = max(0, min(127, start_p))
            slide_end = min(n.end - 0.01, n.start + p["slide_ms"] / 1000.0)
            slide_end = max(n.start + 0.01, slide_end)
            inst.notes.append(
                pm.Note(velocity=n.velocity, pitch=start_p, start=n.start, end=slide_end)
            )
            if slide_end < n.end:
                inst.notes.append(
                    pm.Note(velocity=n.velocity, pitch=n.pitch, start=slide_end, end=n.end)
                )
            continue

        # default: keep note, maybe add PB for bend-like techs
        inst.notes.append(
            pm.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end)
        )
        if tech == "vibrato":
            _add_vibrato_pb(inst, n, p)
        elif tech in ("choke", "choke_strong"):
            _add_bend_up_pb(inst, n, p)
        elif tech == "prebend":
            _add_prebend_pb(inst, n, p)
        elif tech == "bend_release":
            _add_bend_release_pb(inst, n, p)
        elif tech == "arming":
            _add_arming_pb(inst, n, p)
        elif tech == "slide":
            # notes-mode without next (or mode=pb): pitch-bend slide
            _add_slide_pb(inst, n, p, notes[i + 1] if i + 1 < len(notes) else None)

    # Stable PB order for players that assume time-sorted events
    inst.pitch_bends.sort(key=lambda b: (b.time, b.pitch))
    midi.instruments.append(inst)
    return midi


def _add_vibrato_pb(inst: pm.Instrument, n: NoteEvent, p: dict[str, Any]) -> None:
    delay = n.duration * float(p["delay"])
    fade = float(p["fade_in_ms"]) / 1000.0
    depth = float(p["depth"]) * _PB_PER_SEMITONE  # depth as fraction of 2st
    rate = float(p["rate_hz"])
    phase = float(p["phase"])
    _reset_pb(inst, n.start)
    t = n.start
    dt = 1.0 / 60.0
    while t < n.end - 1e-6:
        local = t - n.start - delay
        if local < 0:
            val = 0.0
        else:
            env = 1.0
            if fade > 1e-6:
                env = min(1.0, local / fade)
            val = depth * env * math.sin(2 * math.pi * rate * local + phase)
        inst.pitch_bends.append(pm.PitchBend(pitch=_clamp_pb(val), time=t))
        t += dt
    _reset_pb(inst, n.end)


def _add_bend_up_pb(inst: pm.Instrument, n: NoteEvent, p: dict[str, Any]) -> None:
    rise = float(p["rise_ms"]) / 1000.0
    hold_frac = float(p["hold_frac"])
    target = float(p["target_semitone"])
    overshoot = float(p.get("overshoot", 0.0))
    peak = target + overshoot
    t_rise_end = n.start + min(rise, n.duration * 0.5)
    t_hold_end = min(n.end, n.start + n.duration * (0.3 + 0.7 * hold_frac))
    _reset_pb(inst, n.start)
    steps = 12
    for s in range(steps + 1):
        t = n.start + (t_rise_end - n.start) * s / steps
        semi = peak * s / steps
        inst.pitch_bends.append(pm.PitchBend(pitch=_pb_semitone(semi), time=t))
    inst.pitch_bends.append(pm.PitchBend(pitch=_pb_semitone(target), time=t_hold_end))
    # release so the next note does not inherit residual bend
    _reset_pb(inst, n.end)


def _add_prebend_pb(inst: pm.Instrument, n: NoteEvent, p: dict[str, Any]) -> None:
    semi = float(p["pre_semitone"])
    hold = float(p["pre_hold_ms"]) / 1000.0
    rel = float(p["release_ms"]) / 1000.0
    inst.pitch_bends.append(pm.PitchBend(pitch=_pb_semitone(semi), time=n.start))
    t_hold = min(n.end, n.start + hold)
    inst.pitch_bends.append(pm.PitchBend(pitch=_pb_semitone(semi), time=t_hold))
    if rel > 1e-6:
        t_rel = min(n.end, t_hold + rel)
        inst.pitch_bends.append(pm.PitchBend(pitch=0, time=t_rel))
    _reset_pb(inst, n.end)


def _add_bend_release_pb(inst: pm.Instrument, n: NoteEvent, p: dict[str, Any]) -> None:
    peak = float(p["peak_semitone"])
    rise = float(p["rise_ms"]) / 1000.0
    release = float(p["release_ms"]) / 1000.0
    t1 = min(n.end, n.start + rise)
    t2 = min(n.end, t1 + release)
    _reset_pb(inst, n.start)
    inst.pitch_bends.append(pm.PitchBend(pitch=_pb_semitone(peak), time=t1))
    inst.pitch_bends.append(pm.PitchBend(pitch=0, time=t2))
    _reset_pb(inst, n.end)


def _add_arming_pb(inst: pm.Instrument, n: NoteEvent, p: dict[str, Any]) -> None:
    depth = float(p["depth"]) * _PB_PER_SEMITONE
    dip = float(p["dip_ms"]) / 1000.0
    ret = float(p["return_ms"]) / 1000.0
    n_dips = int(p["n_dips"])
    t = n.start
    sign = -1.0
    _reset_pb(inst, n.start)
    for _ in range(n_dips):
        if t >= n.end:
            break
        t_dip = min(n.end, t + dip)
        inst.pitch_bends.append(pm.PitchBend(pitch=0, time=t))
        inst.pitch_bends.append(pm.PitchBend(pitch=_clamp_pb(sign * depth), time=t_dip))
        t_ret = min(n.end, t_dip + ret)
        inst.pitch_bends.append(pm.PitchBend(pitch=0, time=t_ret))
        t = t_ret
        if p.get("direction") == "dip":
            sign = -sign
    _reset_pb(inst, n.end)


def _add_slide_pb(
    inst: pm.Instrument,
    n: NoteEvent,
    p: dict[str, Any],
    nxt: NoteEvent | None,
) -> None:
    semi = float(p["from_semitone"])
    if nxt is not None and nxt.pitch < n.pitch:
        semi = -semi
    elif nxt is None:
        semi = -abs(semi)
    else:
        semi = abs(semi) if nxt.pitch >= n.pitch else -abs(semi)
    # start from offset and land at 0 when reaching note pitch
    slide = float(p["slide_ms"]) / 1000.0
    t1 = min(n.end, n.start + slide)
    inst.pitch_bends.append(pm.PitchBend(pitch=_pb_semitone(-semi), time=n.start))
    inst.pitch_bends.append(pm.PitchBend(pitch=0, time=t1))
    _reset_pb(inst, n.end)


def _render_tremolo(
    inst: pm.Instrument, n: NoteEvent, p: dict[str, Any], rng: random.Random
) -> None:
    length = p["note_len_ms"] / 1000.0
    gap = p["gap_ms"] / 1000.0
    t = n.start
    k = 0
    while t < n.end - 0.005:
        vel = int(p["vel_mean"] + rng.uniform(-p["vel_jitter"], p["vel_jitter"]))
        if p.get("accent_every") and p["accent_every"] > 0 and k % int(p["accent_every"]) == 0:
            vel = min(127, vel + 12)
        vel = max(1, min(127, vel))
        end = min(n.end, t + length)
        inst.notes.append(pm.Note(velocity=vel, pitch=n.pitch, start=t, end=end))
        t = end + gap
        k += 1


def _render_trill(
    inst: pm.Instrument, n: NoteEvent, p: dict[str, Any], rng: random.Random
) -> None:
    alt = int(p["alt_semitone"])
    alt_ms = p["alt_ms"] / 1000.0
    n_alts = int(p["n_alts"])
    t = n.start
    for k in range(n_alts * 2):
        if t >= n.end:
            break
        pitch = n.pitch if k % 2 == 0 else int(max(0, min(127, n.pitch + alt)))
        vel = int(
            max(1, min(127, n.velocity + rng.uniform(-p["vel_jitter"], p["vel_jitter"])))
        )
        end = min(n.end, t + alt_ms)
        inst.notes.append(pm.Note(velocity=vel, pitch=pitch, start=t, end=end))
        t = end


def labels_to_jsonable(labels: list[NoteLabel]) -> list[dict[str, Any]]:
    return [asdict(x) for x in labels]


def carrier_notes_json(notes: list[NoteEvent]) -> list[dict[str, Any]]:
    return [
        {
            "pitch": n.pitch,
            "start": round(n.start, 5),
            "end": round(n.end, 5),
            "velocity": n.velocity,
        }
        for n in notes
    ]
