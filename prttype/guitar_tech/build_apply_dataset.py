"""Build apply-learning pairs: carrier × randomized technique teacher.

Default scale: 1500 carriers × 3 applies ≈ 4500 pairs.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .carriers import generate_carrier, notes_to_midi
from .constants import APPLY_DATASET_DIR, DEFAULT_PROGRAM
from .params import TECH_NAMES
from .render import assign_techs, labels_to_jsonable, render_target


def build(
    *,
    out_dir: Path = APPLY_DATASET_DIR,
    n_carriers: int = 1500,
    applies_per_carrier: int = 3,
    seed: int = 0,
) -> Path:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    carrier_dir = out_dir / "carrier"
    target_dir = out_dir / "target"
    labels_dir = out_dir / "labels"
    for d in (carrier_dir, target_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    tech_note_counts: Counter[str] = Counter()
    n_notes_total = 0
    n_pairs = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for c_i in range(n_carriers):
            notes, _ = generate_carrier(rng, program=DEFAULT_PROGRAM)
            carrier_midi = notes_to_midi(notes, program=DEFAULT_PROGRAM)
            carrier_name = f"c{c_i:05d}.mid"
            carrier_rel = f"carrier/{carrier_name}"
            carrier_midi.write(str(carrier_dir / carrier_name))

            for a_i in range(applies_per_carrier):
                pair_id = f"{c_i:05d}_{a_i:02d}"
                labels = assign_techs(notes, rng)
                target = render_target(notes, labels, program=DEFAULT_PROGRAM, rng=rng)
                target_rel = f"target/{pair_id}.mid"
                labels_rel = f"labels/{pair_id}.json"
                target.write(str(target_dir / f"{pair_id}.mid"))

                for lb in labels:
                    tech_note_counts[lb.tech] += 1
                    n_notes_total += 1

                label_payload = {
                    "id": pair_id,
                    "carrier_id": f"c{c_i:05d}",
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
                    json.dumps(label_payload, ensure_ascii=False),
                    encoding="utf-8",
                )

                row = {
                    "id": pair_id,
                    "carrier": carrier_rel,
                    "target": target_rel,
                    "labels": labels_rel,
                    "n_notes": len(notes),
                    "tech_counts": dict(Counter(lb.tech for lb in labels)),
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_pairs += 1

            if (c_i + 1) % 250 == 0:
                print(f"carriers {c_i + 1}/{n_carriers} pairs={n_pairs}")

    none_n = tech_note_counts.get("none", 0)
    summary = {
        "n_carriers": n_carriers,
        "applies_per_carrier": applies_per_carrier,
        "n_pairs": n_pairs,
        "n_note_labels": n_notes_total,
        "none_ratio": (none_n / n_notes_total) if n_notes_total else 0.0,
        "tech_note_counts": dict(tech_note_counts),
        "seed": seed,
        "program": DEFAULT_PROGRAM,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {out_dir}: pairs={n_pairs} none_ratio={summary['none_ratio']:.2f} "
        f"techs={len([k for k,v in tech_note_counts.items() if k!='none' and v>0])}"
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build guitar tech apply dataset")
    parser.add_argument("--out-dir", type=Path, default=APPLY_DATASET_DIR)
    parser.add_argument("--n-carriers", type=int, default=1500)
    parser.add_argument("--applies-per-carrier", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(
        out_dir=args.out_dir,
        n_carriers=args.n_carriers,
        applies_per_carrier=args.applies_per_carrier,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
