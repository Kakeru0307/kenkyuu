"""Shared placement eligibility filters (train labels + inference)."""

from __future__ import annotations

from .carriers import NoteEvent
from .constants import PB_TECHS
from .features import note_has_overlap
from .params import canonicalize_tech


def filter_tech_for_note(
    tech: str,
    notes: list[NoteEvent],
    index: int,
    *,
    params: dict | None = None,
) -> str:
    """Return tech or ``normal`` if this note cannot carry the technique."""
    tech = canonicalize_tech(tech)
    if tech == "normal":
        return "normal"
    n = notes[index]
    if tech == "vibrato" and n.duration < 0.28:
        return "normal"
    if tech == "tremolo" and n.duration < 0.25:
        return "normal"
    if tech == "slide":
        # 前→後、または次ノートへの滑り込み。どちらも無ければ不可
        if index <= 0 and index + 1 >= len(notes):
            return "normal"
        # 重なりでも「前の音からこの音へ」なら許可（描画側で前を切って PB）
        if note_has_overlap(notes, index) and index <= 0:
            return "normal"
        return tech
    if tech == "mute" and n.pitch > 64:
        # high notes: mute less typical; keep if explicitly labeled but soft-gate in synth only
        pass
    uses_pb = tech in PB_TECHS and not (
        tech == "slide" and (params or {}).get("mode") == "notes"
    )
    if uses_pb and note_has_overlap(notes, index):
        return "normal"
    return tech


def apply_eligibility(
    labels: list,
    notes: list[NoteEvent],
) -> list:
    """In-place sanitize NoteLabel list; returns same list."""
    for lb in labels:
        params = getattr(lb, "params", None) or {}
        tech = filter_tech_for_note(lb.tech, notes, int(lb.note_index), params=params)
        if tech != lb.tech:
            lb.tech = tech
            if tech == "normal":
                lb.params = {"vel_scale": float(params.get("vel_scale", 1.0))}
    return labels
