"""Lead 小節メロの反復／変化スコア（区間 role 依存）。

thin の和声点（±40 前後）と同桁。サビはフック加点、verse/bridge はコピペ減点。
"""

from __future__ import annotations

import copy
from collections import defaultdict

import muspy

from makeData.constants import KEY_TO_SEMITONE, SCALE_INTERVALS, TICKS_PER_BAR
from song_form import SectionRole, SongForm

# role → (copy_w, hook_w) — 点数用。自動修正はコピペ崩しのみ（フック寄せしない）
ROLE_MOTIF_WEIGHTS: dict[SectionRole, tuple[float, float]] = {
    "intro": (-1.0, 0.2),
    "verse": (-1.5, 0.0),
    "pre_chorus": (-0.8, 0.4),
    "chorus": (-0.3, 1.0),
    "bridge": (-2.0, -0.5),
    "inst": (-2.0, -0.5),
    "outro": (-0.5, 0.8),
}

SIM_SAME = 0.85
SIM_NEAR = 0.60
SIM_FAR = 0.40
PREFIX = 8
HOOK_ONSETS = 4
# 自動修正: 直前小節と酷似以上のときだけ（フック強制はうるさくなるのでしない）
MAX_NOTES_NUDGED = 2
MAX_FIX_ROUNDS = 1


def bar_roles_from_form(form: SongForm) -> list[SectionRole]:
    roles: list[SectionRole] = []
    for section in form.sections:
        roles.extend([section.role] * int(section.bars))
    return roles


def _lead_track(music: muspy.Music) -> muspy.Track | None:
    for track in music.tracks:
        if track.is_drum:
            continue
        if (track.name or "").lower() == "lead":
            return track
    for track in music.tracks:
        if not track.is_drum:
            return track
    return None


def _notes_by_bar(track: muspy.Track) -> dict[int, list[muspy.Note]]:
    by_bar: dict[int, list[muspy.Note]] = defaultdict(list)
    for note in track.notes:
        bar = int(note.time) // TICKS_PER_BAR
        if bar >= 0:
            by_bar[bar].append(note)
    for bar in by_bar:
        by_bar[bar].sort(key=lambda n: (int(n.time), int(n.pitch)))
    return by_bar


def _pc_seq(notes: list[muspy.Note], *, n: int | None = None) -> tuple[int, ...]:
    pcs = [int(n.pitch) % 12 for n in notes]
    if n is not None:
        pcs = pcs[:n]
    return tuple(pcs)


def _similarity(a: tuple[int, ...], b: tuple[int, ...], *, n: int = PREFIX) -> float:
    if not a or not b:
        return 0.0
    aa = a[:n]
    bb = b[:n]
    m = min(len(aa), len(bb))
    if m <= 0:
        return 0.0
    hit = sum(1 for i in range(m) if aa[i] == bb[i])
    return hit / float(m)


def motif_component(
    *,
    sim_prev: float,
    sim_hook: float,
    copy_w: float,
    hook_w: float,
) -> float:
    """copy_w < 0 でコピペ減点、hook_w > 0 でサビ動機加点。"""
    score = 0.0
    # copy_w=-1.5 & 同一 → -60。chorus copy_w=-0.3 → -12。
    if sim_prev >= SIM_SAME:
        score += copy_w * 40.0
    elif sim_prev >= SIM_NEAR:
        score += copy_w * 20.0
    elif sim_prev < SIM_FAR and copy_w < 0:
        score += (-copy_w) * 12.0

    if hook_w != 0.0:
        if sim_hook >= SIM_SAME:
            score += hook_w * 35.0
        elif sim_hook >= SIM_NEAR:
            score += hook_w * 18.0
        elif sim_hook < SIM_FAR and hook_w > 0:
            score += hook_w * (-10.0)
    return score


def _scale_pitches(key: str, mode: str) -> list[int]:
    tonic = KEY_TO_SEMITONE[key]
    return sorted({(tonic + iv) % 12 for iv in SCALE_INTERVALS[mode]})


def _break_copy_notes(
    notes: list[muspy.Note],
    *,
    scale_pcs: list[int],
    prev_seq: tuple[int, ...],
    max_nudge: int = MAX_NOTES_NUDGED,
) -> int:
    """直前小節と同じ PC の弱拍音だけ、最大 max_nudge 個をスケール隣へ。フック寄せはしない。"""
    if not notes or not scale_pcs or not prev_seq:
        return 0
    changed = 0
    for i, note in enumerate(notes):
        if changed >= max_nudge:
            break
        tick_in_bar = int(note.time) % TICKS_PER_BAR
        strong = tick_in_bar in (0, TICKS_PER_BAR // 2)
        if strong and i == 0:
            continue
        pc = int(note.pitch) % 12
        if i >= len(prev_seq) or pc != prev_seq[i]:
            continue
        idx = scale_pcs.index(pc) if pc in scale_pcs else 0
        new_pc = scale_pcs[(idx + 1) % len(scale_pcs)]
        note.pitch = int(note.pitch) - pc + new_pc
        changed += 1
    return changed


def apply_lead_motif_scores(
    music: muspy.Music,
    form: SongForm,
    *,
    key: str,
    mode: str,
    fix: bool = True,
) -> dict[str, float | int | list]:
    """Lead 小節にモチーフ点を付け、酷似コピペだけ静かに崩す。

    サビへのフック強制書き換えはしない（うるさくなるため）。
    """
    track = _lead_track(music)
    if track is None or not track.notes:
        return {"bars": 0, "fixed": 0, "mean_score": 0.0}

    roles = bar_roles_from_form(form)
    by_bar = _notes_by_bar(track)
    total_bars = max(len(roles), max(by_bar.keys(), default=-1) + 1)
    scale_pcs = _scale_pitches(key, mode)

    hook: tuple[int, ...] = ()
    for bar, role in enumerate(roles):
        if role == "chorus" and by_bar.get(bar):
            hook = _pc_seq(by_bar[bar], n=HOOK_ONSETS)
            break

    scores: list[float] = []
    fixed = 0
    details: list[tuple[int, str, float, float, float]] = []

    for bar in range(total_bars):
        role: SectionRole = roles[bar] if bar < len(roles) else "verse"
        copy_w, hook_w = ROLE_MOTIF_WEIGHTS.get(role, (-1.0, 0.0))
        notes = by_bar.get(bar, [])
        seq = _pc_seq(notes, n=PREFIX)
        prev_seq = _pc_seq(by_bar.get(bar - 1, []), n=PREFIX) if bar > 0 else ()
        sim_prev = _similarity(seq, prev_seq) if prev_seq else 0.0
        sim_hook = _similarity(seq, hook) if hook else 0.0
        score = motif_component(
            sim_prev=sim_prev, sim_hook=sim_hook, copy_w=copy_w, hook_w=hook_w
        )

        # 自動修正は「変化してほしい区間での酷似コピペ」のみ
        should_break = (
            fix
            and notes
            and copy_w < 0
            and sim_prev >= SIM_NEAR
            and role not in ("chorus",)  # サビの反復は触らない
        )
        if should_break:
            for _ in range(MAX_FIX_ROUNDS):
                n_ch = _break_copy_notes(
                    notes, scale_pcs=scale_pcs, prev_seq=prev_seq
                )
                if n_ch <= 0:
                    break
                seq = _pc_seq(notes, n=PREFIX)
                sim_prev = _similarity(seq, prev_seq) if prev_seq else 0.0
                sim_hook = _similarity(seq, hook) if hook else 0.0
                score = motif_component(
                    sim_prev=sim_prev,
                    sim_hook=sim_hook,
                    copy_w=copy_w,
                    hook_w=hook_w,
                )
                if sim_prev < SIM_NEAR:
                    break
            fixed += 1

        scores.append(score)
        details.append((bar, role, sim_prev, sim_hook, score))

    track.notes = sorted(track.notes, key=lambda n: (int(n.time), int(n.pitch)))
    mean_score = float(sum(scores) / max(1, len(scores)))
    return {
        "bars": len(scores),
        "fixed": fixed,
        "mean_score": mean_score,
        "hook": list(hook),
        "details": details,
    }


def clone_music(music: muspy.Music) -> muspy.Music:
    return copy.deepcopy(music)
