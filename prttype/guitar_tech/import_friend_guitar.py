"""Import commissioned friend guitar MIDI (folder name → tech).

音高シフトで種を増やす（元クリップの旋律・奏法ラベルはそのまま）。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pretty_midi as pm

from .carriers import NoteEvent
from .constants import TECH_ROOT
from .eligibility import apply_eligibility
from .features import pretty_midi_to_notes
from .import_egsolo_tech import vel_scales_from_notes
from .params import TECH_NAMES
from .render import NoteLabel, labels_to_jsonable, render_target

# 0 = 原調。半音シフトでレジスタ多様性を増やす（PB は相対なので据え置き）
DEFAULT_TRANSPOSE_SEMITONES: tuple[int, ...] = (0, -5, -2, 2, 5, 7)

# Japanese folder names from 依頼仕様
FOLDER_TO_TECH: dict[str, str] = {
    "スライド": "slide",
    "ハンマリング": "hammer",
    "プリング": "pull",
    "パームミュート": "mute",
    "ゴーストノート": "ghost",
    "トリル": "vibrato",
    "トレモロピッキング": "tremolo",
    "タッピング": "tap",
    "ナチュラルハーモニクス": "harmonic",
    "浅いチョーキング": "bend",
    "チョーキング": "bend",
    "強いチョーキング": "choke_strong",
    "プリベンド": "prebend",
    "ベンドリリース": "bend_release",
    "ビブラート": "vibrato",
    "アーミング": "arming",
}
# skip: ピンチハーモニクス, スタッカート
SKIP_FOLDERS = frozenset({"ピンチハーモニクス", "スタッカート"})


def _pb_curve_params(src: pm.PrettyMIDI) -> dict:
    """Store raw pitch-bend points for later sampling / stats."""
    pts: list[tuple[float, int]] = []
    for inst in src.instruments:
        for b in inst.pitch_bends:
            pts.append((float(b.time), int(b.pitch)))
    pts.sort()
    if len(pts) > 64:
        step = max(1, len(pts) // 64)
        pts = pts[::step]
    return {"pb_curve": pts} if pts else {}


def _transpose_notes(notes: list[NoteEvent], semitones: int) -> list[NoteEvent] | None:
    """全ノートを半音シフト。1音でも 0–127 外ならそのシフトは破棄。"""
    if semitones == 0:
        return [
            NoteEvent(n.pitch, n.start, n.end, n.velocity) for n in notes
        ]
    out: list[NoteEvent] = []
    for n in notes:
        pitch = int(n.pitch) + int(semitones)
        if pitch < 0 or pitch > 127:
            return None
        out.append(NoteEvent(pitch, n.start, n.end, n.velocity))
    return out


def _notes_to_carrier_midi(
    notes: list[NoteEvent],
    *,
    program: int = 29,
    source: pm.PrettyMIDI | None = None,
) -> pm.PrettyMIDI:
    midi = pm.PrettyMIDI()
    if source is not None:
        midi.resolution = int(source.resolution)
        if hasattr(source, "_tick_scales") and source._tick_scales:
            midi._tick_scales = [
                (int(tick), float(scale)) for tick, scale in source._tick_scales
            ]
        midi.time_signature_changes = list(source.time_signature_changes)
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
    if source is not None:
        for src_inst in source.instruments:
            for b in src_inst.pitch_bends:
                inst.pitch_bends.append(
                    pm.PitchBend(pitch=int(b.pitch), time=float(b.time))
                )
            break
    midi.instruments.append(inst)
    return midi


def import_friend_guitar(
    *,
    root: Path = TECH_ROOT,
    out_dir: Path,
    write_targets: bool = True,
    transpose_semitones: tuple[int, ...] = DEFAULT_TRANSPOSE_SEMITONES,
) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    out_dir = Path(out_dir)
    carrier_dir = out_dir / "carrier"
    target_dir = out_dir / "target"
    labels_dir = out_dir / "labels"
    for d in (carrier_dir, target_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    tech_counts: Counter[str] = Counter()
    n_notes = 0
    written = 0
    n_source_clips = 0
    manifest = out_dir / "manifest_friend.jsonl"
    shifts = tuple(int(s) for s in transpose_semitones)

    with manifest.open("w", encoding="utf-8") as mf:
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            name = folder.name
            if name in SKIP_FOLDERS:
                continue
            tech = FOLDER_TO_TECH.get(name)
            if tech is None:
                continue
            for mid in sorted(folder.glob("*.mid")):
                src = pm.PrettyMIDI(str(mid))
                base_notes = pretty_midi_to_notes(src)
                if not base_notes:
                    continue
                n_source_clips += 1
                mean_v = sum(n.velocity for n in base_notes) / len(base_notes)
                pb = _pb_curve_params(src)

                for semi in shifts:
                    notes = _transpose_notes(base_notes, semi)
                    if notes is None:
                        continue
                    scales = [
                        max(0.15, min(1.5, float(n.velocity) / max(1.0, mean_v)))
                        for n in notes
                    ]
                    labels = [
                        NoteLabel(i, tech, {"vel_scale": scales[i], **pb})
                        for i in range(len(notes))
                    ]
                    apply_eligibility(labels, notes)
                    tag = f"t{semi:+d}" if semi != 0 else "t+0"
                    pair_id = f"friend_{name}_{mid.stem}_{tag}"
                    carrier = _notes_to_carrier_midi(notes, program=29, source=src)
                    carrier.write(str(carrier_dir / f"{pair_id}.mid"))
                    if write_targets:
                        target = render_target(notes, labels, program=29, source=src)
                        target.write(str(target_dir / f"{pair_id}.mid"))
                    payload = {
                        "id": pair_id,
                        "source": "friend",
                        "folder": name,
                        "transpose": semi,
                        "notes": [
                            {
                                "pitch": n.pitch,
                                "start": round(n.start, 5),
                                "end": round(n.end, 5),
                                "velocity": n.velocity,
                            }
                            for n in notes
                        ],
                        "labels": labels_to_jsonable(labels),
                        "tech_names": list(TECH_NAMES),
                    }
                    (labels_dir / f"{pair_id}.json").write_text(
                        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                    )
                    for lb in labels:
                        tech_counts[lb.tech] += 1
                        n_notes += 1
                    mf.write(
                        json.dumps(
                            {
                                "id": pair_id,
                                "carrier": f"carrier/{pair_id}.mid",
                                "labels": f"labels/{pair_id}.json",
                                "source": "friend",
                                "tech": tech,
                                "transpose": semi,
                                "n_notes": len(notes),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1

    summary = {
        "source": "friend",
        "n_source_clips": n_source_clips,
        "n_pairs": written,
        "n_note_labels": n_notes,
        "transpose_semitones": list(shifts),
        "tech_note_counts": dict(tech_counts),
    }
    (out_dir / "summary_friend.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    import argparse

    from .constants import APPLY_DATASET_DIR

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=TECH_ROOT)
    p.add_argument("--out-dir", type=Path, default=APPLY_DATASET_DIR)
    p.add_argument(
        "--transpose",
        type=str,
        default=",".join(str(s) for s in DEFAULT_TRANSPOSE_SEMITONES),
        help="comma-separated semitone shifts (include 0 for original)",
    )
    args = p.parse_args()
    shifts = tuple(int(x.strip()) for x in args.transpose.split(",") if x.strip())
    print(
        import_friend_guitar(
            root=args.root,
            out_dir=args.out_dir,
            transpose_semitones=shifts or DEFAULT_TRANSPOSE_SEMITONES,
        )
    )


if __name__ == "__main__":
    main()
