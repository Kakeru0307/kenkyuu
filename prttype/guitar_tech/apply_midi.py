"""Apply learned technique classifier + parametric renderer to a MIDI file."""

from __future__ import annotations

import argparse
import json
import random
import secrets
import shutil
from pathlib import Path

import pretty_midi as pm
import torch

from .apply_model import TechApplyNet
from .carriers import NoteEvent
from .constants import CHECKPOINT_DIR, DEFAULT_PROGRAM, OUTPUT_DIR, PB_TECHS
from .features import (
    NOTE_FEATURE_DIM,
    note_features,
    note_has_overlap,
    pretty_midi_to_notes,
    top_voice_indices,
)
from .eligibility import apply_eligibility
from .params import (
    TECH_NAMES,
    TRACK_VEL_GAINS,
    allowed_techs_for_role,
    canonicalize_tech,
    sample_params,
)
from .render import NoteLabel, labels_to_jsonable, render_target


def load_apply_model(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[TechApplyNet, list[str]]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in blob:
        raise ValueError(
            f"{ckpt_path} は奏法付与 ckpt ではありません"
            "（model_state_dict なし。guitar_tech_apply_last.pt を指定してください）"
        )
    meta = blob.get("meta") or {}
    names = list(meta.get("tech_names") or TECH_NAMES)
    if not any(canonicalize_tech(n) == "normal" for n in names):
        raise ValueError(f"{ckpt_path}: meta.tech_names に normal/none がありません")
    feat_dim = int(meta.get("feature_dim", NOTE_FEATURE_DIM))
    if feat_dim != NOTE_FEATURE_DIM:
        raise ValueError(
            f"{ckpt_path}: feature_dim={feat_dim} と NOTE_FEATURE_DIM={NOTE_FEATURE_DIM} が不一致"
        )
    predict_vel = bool(meta.get("predict_vel", "vel_head.weight" in blob["model_state_dict"]))
    model = TechApplyNet(
        len(names),
        in_dim=feat_dim,
        hidden=int(meta.get("hidden", 128)),
        predict_vel=predict_vel,
    )
    try:
        model.load_state_dict(blob["model_state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"{ckpt_path} の state_dict が TechApplyNet と合いません"
            "（旧 vocab / frame AE の ckpt では使えません。再学習が必要です）"
        ) from exc
    model.to(device)
    model.eval()
    return model, names


def _normal_index(tech_names: list[str]) -> int:
    for key in ("normal", "none"):
        try:
            return tech_names.index(key)
        except ValueError:
            continue
    return 0


@torch.inference_mode()
def predict_labels(
    notes: list[NoteEvent],
    model: TechApplyNet,
    tech_names: list[str],
    *,
    device: torch.device,
    rng: random.Random,
    none_bias: float = 0.0,
    allowed_techs: frozenset[str] | None = None,
) -> list[NoteLabel]:
    if not notes:
        return []
    feats = torch.from_numpy(note_features(notes)).float().to(device)
    out = model(feats)
    if isinstance(out, tuple):
        logits, vel_pred = out
        vel_pred = vel_pred.cpu().tolist()
    else:
        logits = out
        vel_pred = [1.0] * len(notes)
    if none_bias:
        logits = logits.clone()
        ni = _normal_index(tech_names)
        logits[:, ni] = logits[:, ni] + float(none_bias)
    if allowed_techs is not None:
        logits = logits.clone()
        for i, name in enumerate(tech_names):
            if canonicalize_tech(name) not in allowed_techs:
                logits[:, i] = -1e9
    pred = logits.argmax(dim=-1).cpu().tolist()
    labels: list[NoteLabel] = []
    for i, tid in enumerate(pred):
        tech = tech_names[int(tid)] if 0 <= int(tid) < len(tech_names) else "normal"
        tech = canonicalize_tech(tech)
        if allowed_techs is not None and tech not in allowed_techs:
            tech = "normal"
        if tech in PB_TECHS and note_has_overlap(notes, i):
            tech = "normal"
        params = sample_params(tech, rng) if tech != "normal" else {}
        vs = float(vel_pred[i]) if i < len(vel_pred) else 1.0
        params["vel_scale"] = max(0.15, min(1.5, vs))
        labels.append(NoteLabel(note_index=i, tech=tech, params=params))
    apply_eligibility(labels, notes)
    return labels


def _target_track_indexes(
    src: pm.PrettyMIDI,
    *,
    track_name: str | None,
    role: str,
) -> list[int]:
    """Pick instrument indexes to rewrite. Prefer exact name; keep other tracks intact."""
    name = track_name or ("Backing" if role == "backing" else "Lead")
    named = [
        i
        for i, inst in enumerate(src.instruments)
        if (not inst.is_drum) and (inst.name or "") == name
    ]
    if named:
        return named
    # single non-drum file (lead-only / backing-only generators)
    nondrum = [i for i, inst in enumerate(src.instruments) if not inst.is_drum]
    if len(nondrum) == 1:
        return nondrum
    # name mismatch on multi-track: do not touch anything
    return []


def assign_labels_for_midi(
    midi_path: Path | str,
    *,
    checkpoint: Path | None = None,
    seed: int | None = None,
    none_bias: float = 0.0,
    top_voice: bool = True,
    role: str = "lead",
    render: bool = True,
    out_path: Path | None = None,
    instrument_name: str | None = None,
) -> tuple[list[NoteLabel], Path | None]:
    """Label authority API: note-level tech + vel on one track; other tracks kept."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = checkpoint or (CHECKPOINT_DIR / "guitar_tech_apply_last.pt")
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"apply checkpoint missing: {ckpt}")
    model, tech_names = load_apply_model(Path(ckpt), device)
    src = pm.PrettyMIDI(str(midi_path))
    if seed is None:
        seed = secrets.randbelow(2**31)
    rng = random.Random(seed)
    allowed = allowed_techs_for_role(role)
    track_name = instrument_name or ("Backing" if role == "backing" else "Lead")
    target_idxs = _target_track_indexes(src, track_name=track_name, role=role)

    all_labels: list[NoteLabel] = []
    for idx in target_idxs:
        inst = src.instruments[idx]
        notes = _notes_from_instrument(inst)
        if not notes:
            continue
        if top_voice:
            voice_idx = top_voice_indices(notes)
            voice_notes = [notes[i] for i in voice_idx]
            voice_labels = predict_labels(
                voice_notes,
                model,
                tech_names,
                device=device,
                rng=rng,
                none_bias=none_bias,
                allowed_techs=allowed,
            )
            labels = [
                NoteLabel(i, "normal", {"vel_scale": 1.0}) for i in range(len(notes))
            ]
            for j, lb in enumerate(voice_labels):
                gi = voice_idx[j]
                tech = lb.tech
                if tech in PB_TECHS and note_has_overlap(notes, gi):
                    tech = "normal"
                labels[gi] = NoteLabel(
                    gi, tech, lb.params if tech != "normal" else {"vel_scale": 1.0}
                )
        else:
            labels = predict_labels(
                notes,
                model,
                tech_names,
                device=device,
                rng=rng,
                none_bias=none_bias,
                allowed_techs=allowed,
            )
        all_labels.extend(labels)
        if render:
            rendered = render_target(
                notes,
                labels,
                program=int(inst.program),
                rng=rng,
                source=src,
                instrument_name=inst.name or track_name,
            )
            if rendered.instruments:
                src.instruments[idx] = rendered.instruments[0]

    meta_path = Path(midi_path).with_suffix(".tech.json")
    meta_path.write_text(
        json.dumps(
            {
                "source": str(midi_path),
                "checkpoint": str(ckpt),
                "seed": seed,
                "role": role,
                "track_name": track_name,
                "target_indexes": target_idxs,
                "labels": labels_to_jsonable(all_labels),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    out: Path | None = None
    if render:
        out = Path(out_path) if out_path else Path(midi_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        src.write(str(out))
    return all_labels, out


def _role_from_instrument(inst: pm.Instrument) -> str:
    name = (inst.name or "").lower()
    if inst.is_drum or "drum" in name:
        return "drums"
    if "bass" in name or int(inst.program) in range(32, 40):
        return "bass"
    if "back" in name or "rhythm" in name or "comp" in name:
        return "backing"
    if "lead" in name or "solo" in name:
        return "lead"
    # electric guitar programs default: treat as lead if overdriven-ish else backing
    if int(inst.program) in (29, 30):
        return "lead"
    if int(inst.program) in range(24, 32):
        return "backing"
    return "lead"


def balance_track_velocities(
    midi_path: Path | str,
    *,
    gains: dict[str, float] | None = None,
    out_path: Path | None = None,
) -> Path:
    """Lead-referenced track gains: backing/bass/drums quieter."""
    gains = gains or TRACK_VEL_GAINS
    path = Path(midi_path)
    src = pm.PrettyMIDI(str(path))
    for inst in src.instruments:
        role = _role_from_instrument(inst)
        g = float(gains.get(role, 1.0))
        if abs(g - 1.0) < 1e-6:
            continue
        for n in inst.notes:
            n.velocity = int(max(1, min(127, round(n.velocity * g))))
    out = Path(out_path) if out_path else path
    out.parent.mkdir(parents=True, exist_ok=True)
    src.write(str(out))
    return out


def apply_techniques_to_midi(
    midi_path: Path | str,
    *,
    checkpoint: Path | None = None,
    out_path: Path | None = None,
    seed: int | None = None,
    none_bias: float = 0.0,
    program: int | None = None,
    top_voice: bool = True,
    role: str = "lead",
    instrument_name: str | None = None,
) -> Path:
    del program  # per-track program kept from source instrument
    out = Path(out_path) if out_path else (
        OUTPUT_DIR / f"{Path(midi_path).stem}_tech.mid"
    )
    labels, written = assign_labels_for_midi(
        midi_path,
        checkpoint=checkpoint,
        seed=seed,
        none_bias=none_bias,
        top_voice=top_voice,
        role=role,
        render=True,
        out_path=out,
        instrument_name=instrument_name,
    )
    assert written is not None
    meta_path = written.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "source": str(midi_path),
                "checkpoint": str(checkpoint or (CHECKPOINT_DIR / "guitar_tech_apply_last.pt")),
                "seed": seed,
                "top_voice": top_voice,
                "role": role,
                "labels": labels_to_jsonable(labels),
                "tech_counts": {
                    t: sum(1 for lb in labels if lb.tech == t)
                    for t in sorted({lb.tech for lb in labels})
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return written


def _notes_from_instrument(inst: pm.Instrument) -> list[NoteEvent]:
    notes = [
        NoteEvent(
            pitch=int(n.pitch),
            start=float(n.start),
            end=float(n.end),
            velocity=int(n.velocity),
        )
        for n in inst.notes
    ]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def labels_for_section_techs(
    notes: list[NoteEvent],
    sections: list[tuple[float, float, str]],
    rng: random.Random,
) -> list[NoteLabel]:
    """ノート開始時刻が含まれる区間の奏法を付ける。分類モデルは使わない。"""
    labels: list[NoteLabel] = []
    for i, note in enumerate(notes):
        tech = "normal"
        for start, end, name in sections:
            if start <= note.start < end:
                tech = canonicalize_tech(name) if canonicalize_tech(name) in TECH_NAMES else "normal"
                break
        params = sample_params(tech, rng) if tech != "normal" else {"vel_scale": 1.0}
        if tech == "slide" and params.get("mode") == "notes" and i + 1 >= len(notes):
            params["mode"] = "pb"
        labels.append(NoteLabel(note_index=i, tech=tech, params=params))
    apply_eligibility(labels, notes)
    return labels


def apply_section_techs_to_track(
    midi_path: Path | str,
    sections: list[tuple[float, float, str]],
    *,
    track_name: str = "Lead",
    seed: int | None = None,
    out_path: Path | None = None,
) -> Path:
    """区間 (開始秒, 終了秒, 奏法名) を、指定名のトラックへ書き込む。

    他のトラックは残す。奏法名は guitar_tech.params.TECH_NAMES。
    """
    path = Path(midi_path)
    src = pm.PrettyMIDI(str(path))
    lead_indexes = [
        i
        for i, inst in enumerate(src.instruments)
        if inst.name == track_name and not inst.is_drum
    ]
    if not lead_indexes:
        return path
    if seed is None:
        seed = secrets.randbelow(2**31)
    rng = random.Random(seed)
    for idx in lead_indexes:
        inst = src.instruments[idx]
        notes = _notes_from_instrument(inst)
        if not notes:
            continue
        labels = labels_for_section_techs(notes, sections, rng)
        rendered = render_target(
            notes,
            labels,
            program=int(inst.program),
            rng=rng,
            source=src,
            instrument_name=track_name,
        )
        if rendered.instruments:
            src.instruments[idx] = rendered.instruments[0]
    out = Path(out_path) if out_path else path
    out.parent.mkdir(parents=True, exist_ok=True)
    src.write(str(out))
    return out


def apply_techniques_overwrite(
    midi_path: Path | str,
    *,
    checkpoint: Path | None = None,
    seed: int | None = None,
    none_bias: float = 0.0,
    keep_bare: bool = True,
    top_voice: bool = True,
    role: str = "lead",
    instrument_name: str | None = None,
) -> Path:
    """生成 MIDI に奏法をかけ、同じパスへ上書きする（本線の常時付与用）。"""
    path = Path(midi_path)
    if keep_bare:
        bare = path.with_name(path.stem + "_bare.mid")
        shutil.copy2(path, bare)
    return apply_techniques_to_midi(
        path,
        checkpoint=checkpoint,
        out_path=path,
        seed=seed,
        none_bias=none_bias,
        top_voice=top_voice,
        role=role,
        instrument_name=instrument_name,
    )


def apply_techniques_to_muspy_music(
    music: "muspy.Music",
    *,
    checkpoint: Path | None = None,
    seed: int | None = None,
    none_bias: float = 0.0,
    top_voice: bool = True,
    role: str = "lead",
    track_name: str | None = None,
):
    """muspy.Music → 一時 MIDI で奏法付与 → muspy.Music（form 用）。"""
    import tempfile

    import muspy
    from patch_to_midi import save_music

    name = track_name or ("Backing" if role == "backing" else "Lead")
    for t in music.tracks:
        if not t.is_drum:
            t.name = name

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "part.mid"
        save_music(music, p)
        apply_techniques_to_midi(
            p,
            checkpoint=checkpoint,
            out_path=p,
            seed=seed,
            none_bias=none_bias,
            top_voice=top_voice,
            role=role,
            instrument_name=name,
        )
        out = muspy.read_midi(p)
        for t in out.tracks:
            if not t.is_drum:
                t.name = name
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply guitar techniques to MIDI")
    parser.add_argument("midi", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "guitar_tech_apply_last.pt",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--none-bias",
        type=float,
        default=0.0,
        help="none クラス logit に足すバイアス（大きいほど控えめ）",
    )
    parser.add_argument(
        "--all-voices",
        action="store_true",
        help="最上声部だけなく全ノートに分類（和音では PB 奏法は自動スキップ）",
    )
    args = parser.parse_args()
    path = apply_techniques_to_midi(
        args.midi,
        checkpoint=args.checkpoint,
        out_path=args.out,
        seed=args.seed,
        none_bias=args.none_bias,
        top_voice=not args.all_voices,
    )
    print(f"wrote {path}")
    print(f"meta {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
