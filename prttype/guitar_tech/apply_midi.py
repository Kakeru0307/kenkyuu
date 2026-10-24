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
from .params import TECH_NAMES, allowed_techs_for_role, sample_params
from .render import NoteLabel, render_target


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
    if "none" not in names:
        raise ValueError(f"{ckpt_path}: meta.tech_names に none がありません")
    feat_dim = int(meta.get("feature_dim", NOTE_FEATURE_DIM))
    if feat_dim != NOTE_FEATURE_DIM:
        raise ValueError(
            f"{ckpt_path}: feature_dim={feat_dim} と NOTE_FEATURE_DIM={NOTE_FEATURE_DIM} が不一致"
        )
    model = TechApplyNet(
        len(names),
        in_dim=feat_dim,
        hidden=int(meta.get("hidden", 128)),
    )
    try:
        model.load_state_dict(blob["model_state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"{ckpt_path} の state_dict が TechApplyNet と合いません"
            "（frame AE の guitar_tech_last.pt では使えません）"
        ) from exc
    model.to(device)
    model.eval()
    return model, names


def _none_index(tech_names: list[str]) -> int:
    try:
        return tech_names.index("none")
    except ValueError:
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
    logits = model(feats)
    if none_bias:
        logits = logits.clone()
        ni = _none_index(tech_names)
        logits[:, ni] = logits[:, ni] + float(none_bias)
    # 役割で禁止されているクラスはマスクしてから argmax（バイアス操作ではない）
    if allowed_techs is not None:
        logits = logits.clone()
        for i, name in enumerate(tech_names):
            if name not in allowed_techs:
                logits[:, i] = -1e9
    pred = logits.argmax(dim=-1).cpu().tolist()
    labels: list[NoteLabel] = []
    for i, tid in enumerate(pred):
        tech = tech_names[int(tid)] if 0 <= int(tid) < len(tech_names) else "none"
        if allowed_techs is not None and tech not in allowed_techs:
            tech = "none"
        if tech in PB_TECHS and note_has_overlap(notes, i):
            tech = "none"
        params = sample_params(tech, rng) if tech != "none" else {}
        labels.append(NoteLabel(note_index=i, tech=tech, params=params))
    return labels


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = checkpoint or (CHECKPOINT_DIR / "guitar_tech_apply_last.pt")
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"apply checkpoint missing: {ckpt}")

    model, tech_names = load_apply_model(Path(ckpt), device)
    src = pm.PrettyMIDI(str(midi_path))
    all_notes = pretty_midi_to_notes(src)
    if seed is None:
        seed = secrets.randbelow(2**31)
    rng = random.Random(seed)
    allowed = allowed_techs_for_role(role)

    if top_voice and all_notes:
        voice_idx = top_voice_indices(all_notes)
        voice_notes = [all_notes[i] for i in voice_idx]
        voice_labels = predict_labels(
            voice_notes,
            model,
            tech_names,
            device=device,
            rng=rng,
            none_bias=none_bias,
            allowed_techs=allowed,
        )
        labels = [NoteLabel(i, "none", {}) for i in range(len(all_notes))]
        for j, lb in enumerate(voice_labels):
            gi = voice_idx[j]
            tech = lb.tech
            # chord tones still occupy the PB channel
            if tech in PB_TECHS and note_has_overlap(all_notes, gi):
                tech = "none"
            labels[gi] = NoteLabel(
                gi, tech, lb.params if tech != "none" else {}
            )
        notes = all_notes
    else:
        notes = all_notes
        labels = predict_labels(
            notes,
            model,
            tech_names,
            device=device,
            rng=rng,
            none_bias=none_bias,
            allowed_techs=allowed,
        )

    prog = DEFAULT_PROGRAM if program is None else int(program)
    if program is None:
        for inst in src.instruments:
            if not inst.is_drum:
                prog = int(inst.program)
                break
    name = instrument_name
    if name is None:
        for inst in src.instruments:
            if not inst.is_drum and inst.name:
                name = inst.name
                break
    if name is None:
        name = "Backing" if role == "backing" else "Lead"
    # ensure source name survives render
    for inst in src.instruments:
        if not inst.is_drum:
            inst.name = name
    out_midi = render_target(
        notes, labels, program=prog, rng=rng, source=src, instrument_name=name
    )

    out = Path(out_path) if out_path else (
        OUTPUT_DIR / f"{Path(midi_path).stem}_tech.mid"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out_midi.write(str(out))

    meta_path = out.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "source": str(midi_path),
                "checkpoint": str(ckpt),
                "seed": seed,
                "top_voice": top_voice,
                "role": role,
                "labels": [
                    {"note_index": lb.note_index, "tech": lb.tech, "params": lb.params}
                    for lb in labels
                ],
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
    return out


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
        tech = "none"
        for start, end, name in sections:
            if start <= note.start < end:
                tech = name if name in TECH_NAMES else "none"
                break
        if tech == "vibrato" and note.duration < 0.28:
            tech = "none"
        if tech == "slide" and i + 1 >= len(notes):
            tech = "none"
        params = sample_params(tech, rng) if tech != "none" else {}
        if tech == "slide" and params.get("mode") == "notes" and i + 1 >= len(notes):
            params["mode"] = "pb"
        uses_pb = tech in PB_TECHS and not (
            tech == "slide" and params.get("mode") == "notes"
        )
        if uses_pb and note_has_overlap(notes, i):
            tech, params = "none", {}
        labels.append(NoteLabel(note_index=i, tech=tech, params=params))
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
