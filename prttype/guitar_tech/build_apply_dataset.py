"""Build apply dataset: EG-Solo + Guitar-TECHS + friend MIDI."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .constants import (
    APPLY_DATASET_DIR,
    EGSOLO_ROOT,
    GUITAR_TECHS_ROOT,
    TECH_ROOT,
)
from .import_egsolo_tech import import_egsolo
from .import_friend_guitar import import_friend_guitar
from .import_guitar_techs import import_guitar_techs


def build(
    *,
    out_dir: Path = APPLY_DATASET_DIR,
    skip_egsolo: bool = False,
    skip_techs: bool = False,
    skip_friend: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}

    if not skip_egsolo and (EGSOLO_ROOT / "tech_midi").is_dir():
        try:
            summaries["egsolo"] = import_egsolo(egsolo_root=EGSOLO_ROOT, out_dir=out_dir)
            print("egsolo", summaries["egsolo"])
        except Exception as exc:
            print(f"egsolo import skipped: {exc}")
    if not skip_techs and GUITAR_TECHS_ROOT.is_dir():
        try:
            summaries["guitar_techs"] = import_guitar_techs(
                root=GUITAR_TECHS_ROOT, out_dir=out_dir, max_per_tech=30
            )
            print("techs", summaries["guitar_techs"])
        except Exception as exc:
            print(f"techs import skipped: {exc}")
    if not skip_friend and TECH_ROOT.is_dir():
        try:
            summaries["friend"] = import_friend_guitar(root=TECH_ROOT, out_dir=out_dir)
            print("friend", summaries["friend"])
        except Exception as exc:
            print(f"friend import skipped: {exc}")

    total_notes = 0
    tech_counts: Counter[str] = Counter()
    n_pairs = 0
    for s in summaries.values():
        n_pairs += int(s.get("n_pairs", 0))
        total_notes += int(s.get("n_note_labels", 0))
        tech_counts.update(s.get("tech_note_counts") or {})
    normal_n = tech_counts.get("normal", 0)
    summary = {
        "n_pairs": n_pairs,
        "n_note_labels": total_notes,
        "normal_ratio": (normal_n / total_notes) if total_notes else 0.0,
        "tech_note_counts": dict(tech_counts),
        "sources": summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {out_dir}: pairs={n_pairs} normal_ratio={summary['normal_ratio']:.2f}"
    )
    if n_pairs == 0:
        raise RuntimeError(
            f"{out_dir}: pairs=0（EG-Solo / Guitar-TECHS / 依頼ギターのいずれも取り込めなかった）"
        )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build guitar tech apply dataset (EG-Solo + TECHS + friend only)"
    )
    parser.add_argument("--out-dir", type=Path, default=APPLY_DATASET_DIR)
    parser.add_argument("--skip-egsolo", action="store_true")
    parser.add_argument("--skip-techs", action="store_true")
    parser.add_argument("--skip-friend", action="store_true")
    args = parser.parse_args()
    build(
        out_dir=args.out_dir,
        skip_egsolo=args.skip_egsolo,
        skip_techs=args.skip_techs,
        skip_friend=args.skip_friend,
    )


if __name__ == "__main__":
    main()
