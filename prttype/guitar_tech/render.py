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
from .eligibility import apply_eligibility, filter_tech_for_note
from .params import TECH_WEIGHTS, canonicalize_tech, sample_params

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


_MIN_NOTE_SEC = 0.005


def _append_note(
    inst: pm.Instrument,
    *,
    velocity: int,
    pitch: int,
    start: float,
    end: float,
) -> None:
    """PrettyMIDI は end > start 必須。短すぎ／逆転はスキップまたは最小長に伸ばす。"""
    s = float(start)
    e = float(end)
    if e <= s:
        e = s + _MIN_NOTE_SEC
    if e <= s:
        return
    inst.notes.append(
        pm.Note(
            velocity=int(max(1, min(127, velocity))),
            pitch=int(max(0, min(127, pitch))),
            start=s,
            end=e,
        )
    )


# form / patch 系は resolution=4（16分格子）。tremolo 等の 25–70ms は
# その PPQ だと start/end が同一 tick に潰れ、SMF write で note-off が消えて
# 同 pitch が曲末まで伸びる。奏法書き出しだけ高 PPQ にする。
TECH_WRITE_RESOLUTION = 480


def copy_timing(src: pm.PrettyMIDI, dst: pm.PrettyMIDI) -> None:
    """Copy tempo map / time signatures so DAW beat grid matches the source.

    Must also copy ``resolution``; otherwise write() rebuilds set_tempo from
    mismatched tick_scale×resolution and can overflow.
    """
    dst.resolution = int(src.resolution)
    if hasattr(src, "_tick_scales") and src._tick_scales:
        dst._tick_scales = [(int(tick), float(scale)) for tick, scale in src._tick_scales]
    dst.time_signature_changes = list(src.time_signature_changes)


def sanitize_same_pitch_overlaps(
    inst: pm.Instrument,
    *,
    min_dur: float = _MIN_NOTE_SEC,
) -> int:
    """Cut earlier note so same-pitch notes never overlap (SMF is 1 voice/pitch)."""
    from collections import defaultdict

    by_pitch: dict[int, list[pm.Note]] = defaultdict(list)
    for n in inst.notes:
        by_pitch[int(n.pitch)].append(n)
    cuts = 0
    for ns in by_pitch.values():
        ns.sort(key=lambda n: (float(n.start), float(n.end)))
        for a, b in zip(ns, ns[1:]):
            if float(b.start) < float(a.end) - 1e-9:
                a.end = max(float(a.start) + min_dur, float(b.start))
                cuts += 1
            if float(a.end) <= float(a.start):
                a.end = float(a.start) + min_dur
    return cuts


def remaster_for_tech_write(
    src: pm.PrettyMIDI,
    *,
    resolution: int = TECH_WRITE_RESOLUTION,
) -> pm.PrettyMIDI:
    """Rebuild at higher PPQ so sub-16th notes survive PrettyMIDI SMF I/O.

    Absolute times (seconds) are preserved; only tick quantization changes.
    """
    times, tempos = src.get_tempo_changes()
    initial = float(tempos[0]) if len(tempos) else 120.0
    out = pm.PrettyMIDI(resolution=int(resolution), initial_tempo=initial)
    # mid-song tempo changes (tick scales in the new PPQ)
    for t, tempo in zip(times[1:], tempos[1:]):
        tick = int(out.time_to_tick(float(t)))
        scale = 60.0 / (float(tempo) * float(resolution))
        out._tick_scales.append((tick, scale))
    out.time_signature_changes = [
        pm.TimeSignature(ts.numerator, ts.denominator, float(ts.time))
        for ts in src.time_signature_changes
    ]
    for inst in src.instruments:
        sanitize_same_pitch_overlaps(inst)
        ni = pm.Instrument(
            program=int(inst.program),
            is_drum=bool(inst.is_drum),
            name=inst.name,
        )
        for n in inst.notes:
            if float(n.end) <= float(n.start):
                continue
            ni.notes.append(
                pm.Note(
                    velocity=int(n.velocity),
                    pitch=int(n.pitch),
                    start=float(n.start),
                    end=float(n.end),
                )
            )
        for pb in inst.pitch_bends:
            ni.pitch_bends.append(pm.PitchBend(pitch=int(pb.pitch), time=float(pb.time)))
        for cc in getattr(inst, "control_changes", []) or []:
            ni.control_changes.append(
                pm.ControlChange(
                    number=int(cc.number),
                    value=int(cc.value),
                    time=float(cc.time),
                )
            )
        ni.pitch_bends.sort(key=lambda b: (b.time, b.pitch))
        out.instruments.append(ni)
    return out


def assign_techs(notes: list[NoteEvent], rng: random.Random) -> list[NoteLabel]:
    """Synthetic fill only — keep sparse (≈70% normal)."""
    labels: list[NoteLabel] = []
    names = list(TECH_WEIGHTS.keys())
    weights = [TECH_WEIGHTS[n] for n in names]
    for i, n in enumerate(notes):
        if rng.random() < 0.70:
            labels.append(NoteLabel(i, "normal", {"vel_scale": 1.0}))
            continue
        tech = rng.choices(names, weights=weights, k=1)[0]
        if tech == "mute" and n.pitch > 64 and rng.random() < 0.5:
            tech = "normal"
        params = sample_params(tech, rng) if tech != "normal" else {"vel_scale": 1.0}
        if tech == "slide" and params.get("mode") == "notes" and i + 1 >= len(notes):
            params["mode"] = "pb"
        tech = filter_tech_for_note(tech, notes, i, params=params)
        if tech == "normal":
            params = {"vel_scale": float(params.get("vel_scale", 1.0))}
        labels.append(NoteLabel(i, tech, params))
    apply_eligibility(labels, notes)
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
        lb = label_by_i.get(i) or NoteLabel(i, "normal", {"vel_scale": 1.0})
        tech = canonicalize_tech(lb.tech)
        # fill missing parametric keys; keep teacher vel_scale / pb_curve
        p = sample_params(tech, rng)
        p.update(lb.params or {})
        uses_pb = tech in PB_TECHS and not (
            tech == "slide" and p.get("mode") == "notes"
        )
        if uses_pb and note_has_overlap(notes, i):
            # slide は前ノートからこの音へ滑り込ませる（それ以外の PB は従来どおり不可）
            if tech != "slide" or i <= 0:
                tech, p = "normal", {"vel_scale": float(p.get("vel_scale", 1.0))}
            else:
                p["mode"] = "pb"

        vel_scale = float(p.get("vel_scale", 1.0))
        base_vel = int(max(1, min(127, round(n.velocity * vel_scale))))

        # 次が「前→後」slide なら、この音を次の開始で切って PB の土台を空ける
        note_end = float(n.end)
        nxt_lb = label_by_i.get(i + 1) if i + 1 < len(notes) else None
        if (
            nxt_lb is not None
            and canonicalize_tech(nxt_lb.tech) == "slide"
            and notes[i + 1].start < n.end - 1e-6
        ):
            note_end = max(n.start + _MIN_NOTE_SEC, min(n.end, float(notes[i + 1].start)))

        if note_end <= n.start:
            _append_note(
                inst, velocity=base_vel, pitch=n.pitch, start=n.start, end=n.start + _MIN_NOTE_SEC
            )
            continue

        if tech == "tremolo":
            _render_tremolo(inst, n, p, rng)
            continue
        if tech == "ghost":
            pitch = n.pitch - 12 if p.get("pitch_mode") == "drop12" else n.pitch
            length = min(n.duration, float(p.get("len_ms", 15.0)) / 1000.0)
            vel = int(p["vel"]) if "vel" in p else max(1, min(55, base_vel))
            _append_note(
                inst,
                velocity=vel,
                pitch=pitch,
                start=n.start,
                end=n.start + max(_MIN_NOTE_SEC, length),
            )
            continue
        if tech == "mute":
            bias = 1.0 - float(p.get("low_bias", 0.0)) * max(0.0, (60 - n.pitch) / 24.0)
            vel = int(max(1, min(127, n.velocity * float(p.get("vel_scale", 0.55)) * bias)))
            end = n.start + n.duration * float(p.get("len_ratio", 0.65))
            _append_note(
                inst, velocity=vel, pitch=n.pitch, start=n.start, end=max(n.start + 0.02, end)
            )
            continue
        if tech == "hammer":
            # 短ノートで grace > 本体になると end<=start になるので clamp
            grace_ms = min(float(p["grace_ms"]), max(1.0, n.duration * 1000.0 * 0.45))
            grace_end = n.start + grace_ms / 1000.0
            grace_pitch = int(max(0, min(127, n.pitch - int(p["interval"]))))
            _append_note(
                inst,
                velocity=int(n.velocity * p["grace_vel_scale"]),
                pitch=grace_pitch,
                start=n.start,
                end=min(grace_end, n.end),
            )
            main_start = max(n.start, grace_end - float(p["overlap_ms"]) / 1000.0)
            main_start = min(main_start, n.end - _MIN_NOTE_SEC)
            _append_note(
                inst,
                velocity=n.velocity,
                pitch=n.pitch,
                start=main_start,
                end=n.end,
            )
            continue
        if tech == "pull":
            high = int(max(0, min(127, n.pitch + int(round(p["from_semitone"])))))
            from_ms = min(float(p["from_ms"]), max(1.0, n.duration * 1000.0 * 0.45))
            mid = n.start + from_ms / 1000.0
            mid = min(mid, n.end - _MIN_NOTE_SEC)
            mid = max(mid, n.start + _MIN_NOTE_SEC)
            _append_note(
                inst,
                velocity=int(n.velocity * p["from_vel_scale"]),
                pitch=high,
                start=n.start,
                end=mid,
            )
            _append_note(
                inst,
                velocity=int(n.velocity * p["dest_vel_scale"]),
                pitch=n.pitch,
                start=mid,
                end=n.end,
            )
            continue
        if tech == "tap":
            vel = int(max(1, min(127, n.velocity * p["tap_vel_scale"])))
            if p.get("pattern") == "run":
                t = n.start
                for k in range(int(p["run_notes"])):
                    length = p["tap_len_ms"] / 1000.0
                    _append_note(
                        inst,
                        velocity=vel,
                        pitch=n.pitch + k,
                        start=t,
                        end=min(n.end, t + length),
                    )
                    t += length
                    if t >= n.end:
                        break
            else:
                length = max(_MIN_NOTE_SEC, min(n.duration, p["tap_len_ms"] / 1000.0))
                _append_note(
                    inst, velocity=vel, pitch=n.pitch, start=n.start, end=n.start + length
                )
            continue
        if tech == "slide" and p.get("mode") == "notes" and i + 1 < len(notes) and i <= 0:
            # notes モードは非重なり・先頭付近向け。重なり前→後は PB 経路へ
            nxt = notes[i + 1]
            direction = 1 if nxt.pitch >= n.pitch else -1
            start_p = int(n.pitch - direction * int(round(p["from_semitone"])))
            start_p = max(0, min(127, start_p))
            slide_end = min(note_end - _MIN_NOTE_SEC, n.start + p["slide_ms"] / 1000.0)
            slide_end = max(n.start + _MIN_NOTE_SEC, slide_end)
            _append_note(
                inst, velocity=n.velocity, pitch=start_p, start=n.start, end=slide_end
            )
            if slide_end < note_end:
                _append_note(
                    inst, velocity=n.velocity, pitch=n.pitch, start=slide_end, end=note_end
                )
            continue

        # default: keep note (with vel_scale), maybe add PB
        _append_note(
            inst, velocity=base_vel, pitch=n.pitch, start=n.start, end=note_end
        )
        # PB 描画用に tip まで切ったノート長を渡す
        n_pb = NoteEvent(pitch=n.pitch, start=n.start, end=note_end, velocity=n.velocity)
        if tech == "vibrato":
            _add_vibrato_pb(inst, n_pb, p)
        elif tech in ("bend", "choke_strong"):
            _add_bend_up_pb(inst, n_pb, p)
        elif tech == "prebend":
            _add_prebend_pb(inst, n_pb, p)
        elif tech == "bend_release":
            _add_bend_release_pb(inst, n_pb, p)
        elif tech == "arming":
            _add_arming_pb(inst, n_pb, p)
        elif tech == "slide":
            _add_slide_pb(
                inst,
                n_pb,
                p,
                nxt=notes[i + 1] if i + 1 < len(notes) else None,
                prev=notes[i - 1] if i > 0 else None,
            )

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
    nxt: NoteEvent | None = None,
    prev: NoteEvent | None = None,
) -> None:
    """滑り込み PB。prev があれば前の音高 → この音高へ（±2半音に clamp）。"""
    if prev is not None:
        delta = float(n.pitch - prev.pitch)  # +なら下から上がってくる
        delta = max(-2.0, min(2.0, delta))
        if abs(delta) < 0.15:
            return
        # heard = pitch + pb_semitones。開始を prev 相当に → pb0 = prev - n = -delta
        semi0 = -delta
    else:
        semi = float(p.get("from_semitone", 1.0))
        if nxt is not None and nxt.pitch < n.pitch:
            semi = -abs(semi)
        elif nxt is None:
            semi = -abs(semi)
        else:
            semi = abs(semi) if nxt.pitch >= n.pitch else -abs(semi)
        semi0 = -semi
    slide = float(p["slide_ms"]) / 1000.0
    t1 = min(n.end, n.start + slide)
    inst.pitch_bends.append(pm.PitchBend(pitch=_pb_semitone(semi0), time=n.start))
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
