"""Import EG-Solo note_midi + tech_midi into apply label JSON pairs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pretty_midi as pm

from .carriers import NoteEvent
from .constants import EGSOLO_PITCH_TO_TECH, EGSOLO_ROOT
from .eligibility import apply_eligibility
from .features import pretty_midi_to_notes
from .params import TECH_NAMES
from .render import NoteLabel, labels_to_jsonable, render_target


def _stem_key(name: str) -> str:
    """20_11_note.mid / 20_11.mid → 20_11"""
    stem = Path(name).stem
    if stem.endswith("_note"):
        stem = stem[: -len("_note")]
    return stem


def _tech_at_onset(
    tech_notes: list[pm.Note],
    t: float,
    *,
    tol: float = 0.04,
) -> str:
    hits = [n for n in tech_notes if abs(float(n.start) - t) <= tol]
    if not hits:
        # nearest within window
        if not tech_notes:
            return "normal"
        nearest = min(tech_notes, key=lambda n: abs(float(n.start) - t))
        if abs(float(nearest.start) - t) > tol * 2:
            return "normal"
        hits = [nearest]
    # prefer non-normal when multiple (meta pitches ignored)
    mapped: list[str] = []
    for n in hits:
        tech = EGSOLO_PITCH_TO_TECH.get(int(n.pitch))
        if tech is not None:
            mapped.append(tech)
    if not mapped:
        return "normal"
    non_normal = [t for t in mapped if t != "normal"]
    return non_normal[0] if non_normal else "normal"


def vel_scales_from_notes(
    notes: list[NoteEvent],
    techs: list[str],
    *,
    global_normal_mean: float | None = None,
) -> list[float]:
    normal_vels = [n.velocity for n, t in zip(notes, techs) if t == "normal"]
    if normal_vels:
        v_ref = sum(normal_vels) / len(normal_vels)
    elif global_normal_mean and global_normal_mean > 1e-6:
        v_ref = global_normal_mean
    else:
        v_ref = sum(n.velocity for n in notes) / max(1, len(notes))
    v_ref = max(1.0, float(v_ref))
    out: list[float] = []
    for n in notes:
        scale = float(n.velocity) / v_ref
        out.append(max(0.15, min(1.5, scale)))
    return out


def pair_note_tech(
    note_midi: Path,
    tech_midi: Path,
    *,
    global_normal_mean: float | None = None,
) -> tuple[list[NoteEvent], list[NoteLabel]]:
    src = pm.PrettyMIDI(str(note_midi))
    notes = pretty_midi_to_notes(src)
    tech_pm = pm.PrettyMIDI(str(tech_midi))
    tech_notes: list[pm.Note] = []
    for inst in tech_pm.instruments:
        tech_notes.extend(inst.notes)
    tech_notes.sort(key=lambda n: (n.start, n.pitch))

    techs = [_tech_at_onset(tech_notes, float(n.start)) for n in notes]
    scales = vel_scales_from_notes(notes, techs, global_normal_mean=global_normal_mean)
    labels = [
        NoteLabel(i, techs[i], {"vel_scale": scales[i]})
        for i in range(len(notes))
    ]
    apply_eligibility(labels, notes)
    return notes, labels


def import_egsolo(
    *,
    egsolo_root: Path = EGSOLO_ROOT,
    out_dir: Path,
    write_targets: bool = True,
) -> dict:
    note_dir = egsolo_root / "note_midi"
    tech_dir = egsolo_root / "tech_midi"
    if not note_dir.is_dir() or not tech_dir.is_dir():
        raise FileNotFoundError(f"need note_midi + tech_midi under {egsolo_root}")

    tech_by_key = {_stem_key(p.name): p for p in tech_dir.glob("*.mid")}
    note_files = sorted(note_dir.glob("*.mid"))
    pairs: list[tuple[Path, Path, str]] = []
    for nf in note_files:
        key = _stem_key(nf.name)
        tf = tech_by_key.get(key)
        if tf is None:
            continue
        pairs.append((nf, tf, key))
    if not pairs:
        raise RuntimeError(f"no note/tech pairs in {egsolo_root}")

    # first pass: global normal mean velocity
    normal_vels: list[float] = []
    for nf, tf, _ in pairs:
        notes, labels = pair_note_tech(nf, tf)
        for n, lb in zip(notes, labels):
            if lb.tech == "normal":
                normal_vels.append(float(n.velocity))
    gmean = (sum(normal_vels) / len(normal_vels)) if normal_vels else 80.0

    out_dir = Path(out_dir)
    carrier_dir = out_dir / "carrier"
    target_dir = out_dir / "target"
    labels_dir = out_dir / "labels"
    for d in (carrier_dir, target_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    tech_counts: Counter[str] = Counter()
    n_notes = 0
    written = 0
    manifest = out_dir / "manifest_egsolo.jsonl"
    with manifest.open("w", encoding="utf-8") as mf:
        for nf, tf, key in pairs:
            notes, labels = pair_note_tech(nf, tf, global_normal_mean=gmean)
            if not notes:
                continue
            pair_id = f"egsolo_{key}"
            src = pm.PrettyMIDI(str(nf))
            # rewrite carrier with source timing
            carrier_path = carrier_dir / f"{pair_id}.mid"
            src.write(str(carrier_path))
            if write_targets:
                target = render_target(notes, labels, program=29, source=src)
                target.write(str(target_dir / f"{pair_id}.mid"))
            payload = {
                "id": pair_id,
                "source": "egsolo",
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
                        "source": "egsolo",
                        "n_notes": len(notes),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    summary = {
        "source": "egsolo",
        "n_pairs": written,
        "n_note_labels": n_notes,
        "normal_ratio": tech_counts.get("normal", 0) / max(1, n_notes),
        "tech_note_counts": dict(tech_counts),
        "global_normal_mean_vel": gmean,
    }
    (out_dir / "summary_egsolo.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    import argparse

    from .constants import APPLY_DATASET_DIR

    p = argparse.ArgumentParser()
    p.add_argument("--egsolo-root", type=Path, default=EGSOLO_ROOT)
    p.add_argument("--out-dir", type=Path, default=APPLY_DATASET_DIR)
    args = p.parse_args()
    s = import_egsolo(egsolo_root=args.egsolo_root, out_dir=args.out_dir)
    print(s)


if __name__ == "__main__":
    main()
