"""Import Guitar-TECHS technique MIDIs (folder/file name → tech)."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pretty_midi as pm

from .constants import GUITAR_TECHS_ROOT
from .eligibility import apply_eligibility
from .features import pretty_midi_to_notes
from .params import TECH_NAMES
from .render import NoteLabel, labels_to_jsonable, render_target

_FILE_TECH: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"singlenote|alternate|single.?notes?", re.I), "normal"),
    (re.compile(r"palm.?mute", re.I), "mute"),
    (re.compile(r"vibrato", re.I), "vibrato"),
    (re.compile(r"pinch", re.I), "SKIP"),
    (re.compile(r"harmonic", re.I), "harmonic"),
    (re.compile(r"bend", re.I), "bend"),
]


def _tech_from_path(path: Path) -> str | None:
    blob = f"{path.parent.name}/{path.name}"
    for pat, tech in _FILE_TECH:
        if pat.search(blob):
            return None if tech == "SKIP" else tech
    return None


def import_guitar_techs(
    *,
    root: Path = GUITAR_TECHS_ROOT,
    out_dir: Path,
    write_targets: bool = True,
    max_per_tech: int = 40,
) -> dict:
    """Coverage fill — keep per-tech count capped (dense single-tech files)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    out_dir = Path(out_dir)
    carrier_dir = out_dir / "carrier"
    target_dir = out_dir / "target"
    labels_dir = out_dir / "labels"
    for d in (carrier_dir, target_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    # collect midis under *techniques* and singlenotes
    candidates: list[tuple[Path, str]] = []
    for mid in root.rglob("*.mid"):
        parts = {p.lower() for p in mid.parts}
        if not any("technique" in p or "singlenote" in p for p in parts):
            # also accept midi_*.mid directly under techniques
            if "techniques" not in str(mid.parent).lower() and "singlenotes" not in str(
                mid.parent
            ).lower():
                continue
        tech = _tech_from_path(mid)
        if tech is None:
            continue
        candidates.append((mid, tech))

    # cap per tech
    by_tech: dict[str, list[Path]] = {}
    for mid, tech in candidates:
        by_tech.setdefault(tech, []).append(mid)
    selected: list[tuple[Path, str]] = []
    for tech, files in by_tech.items():
        selected.extend((f, tech) for f in files[:max_per_tech])

    tech_counts: Counter[str] = Counter()
    n_notes = 0
    written = 0
    manifest = out_dir / "manifest_techs.jsonl"
    with manifest.open("w", encoding="utf-8") as mf:
        for mid, tech in selected:
            src = pm.PrettyMIDI(str(mid))
            notes = pretty_midi_to_notes(src)
            if not notes:
                continue
            mean_v = sum(n.velocity for n in notes) / len(notes)
            scales = [
                max(0.15, min(1.5, float(n.velocity) / max(1.0, mean_v)))
                for n in notes
            ]
            labels = [
                NoteLabel(i, tech, {"vel_scale": scales[i]})
                for i in range(len(notes))
            ]
            apply_eligibility(labels, notes)
            pair_id = f"techs_{mid.stem}"[:80]
            src.write(str(carrier_dir / f"{pair_id}.mid"))
            if write_targets:
                target = render_target(notes, labels, program=29, source=src)
                target.write(str(target_dir / f"{pair_id}.mid"))
            payload = {
                "id": pair_id,
                "source": "guitar_techs",
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
                        "source": "guitar_techs",
                        "tech": tech,
                        "n_notes": len(notes),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    summary = {
        "source": "guitar_techs",
        "n_pairs": written,
        "n_note_labels": n_notes,
        "tech_note_counts": dict(tech_counts),
    }
    (out_dir / "summary_techs.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    import argparse

    from .constants import APPLY_DATASET_DIR

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=GUITAR_TECHS_ROOT)
    p.add_argument("--out-dir", type=Path, default=APPLY_DATASET_DIR)
    args = p.parse_args()
    print(import_guitar_techs(root=args.root, out_dir=args.out_dir))


if __name__ == "__main__":
    main()
