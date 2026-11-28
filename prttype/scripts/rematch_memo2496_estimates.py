"""Rematch Memo2496 estimates.jsonl against current progression catalog (no Chordino).

Usage (WSL or Windows with path to estimates):
  python scripts/rematch_memo2496_estimates.py \\
    --estimates /home/kakeru0307/github/memo/estimates.jsonl \\
    --out-jsonl /home/kakeru0307/github/memo/structure_pairs.jsonl \\
    --variants-out /home/kakeru0307/github/memo/variant_candidates.jsonl \\
    --estimates-out /home/kakeru0307/github/memo/estimates_rematched.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from makeData.progressions import e3_canonical_mode  # noqa: E402
from memo2496_to_pairs import match_progression  # noqa: E402


def rematch_row(row: dict) -> dict:
    meta = dict(row.get("_meta") or {})
    st = dict(row.get("structure") or {})
    mask = dict(row.get("_label_mask") or {})
    loop_raw = meta.get("loop")
    if not loop_raw:
        row = dict(row)
        row["structure"] = st
        row["_label_mask"] = mask
        row["_meta"] = meta
        return row

    loop = [(int(c["degree"]), str(c.get("quality", "maj"))) for c in loop_raw]
    name, dist = match_progression(loop)
    meta["edit_distance"] = dist
    meta["near"] = name
    if name is not None and dist == 0:
        st["progression"] = name
        st["mode"] = e3_canonical_mode(name, st.get("mode"))
        mask["prog"] = True
        mask["mode"] = True
        meta["match"] = name
        meta["variant"] = False
    elif name is not None and dist == 1:
        st.pop("progression", None)
        mask["prog"] = False
        meta["match"] = name
        meta["variant"] = True
        meta["near"] = name
    else:
        st.pop("progression", None)
        mask["prog"] = False
        meta["match"] = name
        meta["variant"] = True
        meta["near"] = name

    out = dict(row)
    out["structure"] = st
    out["_label_mask"] = mask
    out["_meta"] = meta
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimates", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--variants-out", type=Path, default=None)
    ap.add_argument("--estimates-out", type=Path, default=None)
    ap.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="structure_pairs をここにもコピー（例: colab_train/data/memo2496_pairs/）",
    )
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in args.estimates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n_pair = n_var = n_prog = n_fail = 0
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.variants_out:
        args.variants_out.parent.mkdir(parents=True, exist_ok=True)
    if args.estimates_out:
        args.estimates_out.parent.mkdir(parents=True, exist_ok=True)

    null = os.devnull
    with args.out_jsonl.open("w", encoding="utf-8") as fout, (
        args.variants_out.open("w", encoding="utf-8") if args.variants_out else open(null, "w")
    ) as fvar, (
        args.estimates_out.open("w", encoding="utf-8") if args.estimates_out else open(null, "w")
    ) as fest:
        for row in rows:
            if row.get("error") or (row.get("_meta") or {}).get("failed"):
                n_fail += 1
                fest.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            row = rematch_row(row)
            fest.write(json.dumps(row, ensure_ascii=False) + "\n")
            meta = row.get("_meta") or {}
            if meta.get("variant") and meta.get("loop"):
                fvar.write(
                    json.dumps(
                        {
                            "clip": row.get("clip"),
                            "loop": meta.get("loop"),
                            "near": meta.get("near"),
                            "edit_distance": meta.get("edit_distance"),
                            "key": (row.get("structure") or {}).get("key"),
                            "mode": (row.get("structure") or {}).get("mode"),
                            "bpm": (row.get("structure") or {}).get("bpm"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_var += 1
            if row.get("va") is None:
                continue
            fout.write(
                json.dumps(
                    {
                        "source": "memo2496",
                        "clip": row.get("clip"),
                        "va": row.get("va"),
                        "va_source": row.get("va_source"),
                        "structure": row.get("structure"),
                        "_label_mask": row.get("_label_mask"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_pair += 1
            if (row.get("_label_mask") or {}).get("prog"):
                n_prog += 1

    print(
        f"pairs={n_pair} with_prog={n_prog} variants={n_var} failed={n_fail} -> {args.out_jsonl}"
    )
    if args.copy_to is not None:
        dest = args.copy_to
        if dest.is_dir():
            dest = dest / "structure_pairs.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out_jsonl, dest)
        print(f"copied -> {dest}")


if __name__ == "__main__":
    main()
