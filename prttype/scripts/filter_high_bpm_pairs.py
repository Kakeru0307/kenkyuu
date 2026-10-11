"""既存 structure pairs から高 BPM だけ抜いて専用フォルダへ書く。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PRTTTYPE = SCRIPT_DIR.parent


def row_va(row: dict) -> tuple[float, float] | None:
    va = row.get("va")
    if isinstance(va, dict):
        return float(va["valence"]), float(va["arousal"])
    if isinstance(va, (list, tuple)) and len(va) >= 2:
        return float(va[0]), float(va[1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpm-min", type=float, default=150.0)
    parser.add_argument("--arousal-min", type=float, default=None, help="指定時は A 下限も満たす行だけ")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PRTTTYPE / "data" / "high_bpm_pairs",
    )
    args = parser.parse_args()

    sources = [
        PRTTTYPE / "data" / "pmemo_pairs" / "structure_pairs.jsonl",
        PRTTTYPE / "data" / "fme24_pairs" / "structure_pairs.jsonl",
        PRTTTYPE / "data" / "vgmidi_pairs" / "structure_pairs.jsonl",
        PRTTTYPE / "data" / "prior_pairs" / "manifests" / "accept.jsonl",
    ]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "structure_pairs.jsonl"

    rows: list[dict] = []
    stats: Counter[str] = Counter()

    for path in sources:
        if not path.is_file():
            stats[f"missing:{path.name}"] += 1
            continue
        default_src = "accept" if path.name == "accept.jsonl" else path.parent.name.replace("_pairs", "")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            st = dict(raw.get("structure") or {})
            if st.get("bpm") is None:
                stats["skip_no_bpm"] += 1
                continue
            bpm = float(st["bpm"])
            if bpm < args.bpm_min:
                stats["skip_low_bpm"] += 1
                continue
            va = row_va(raw)
            if va is None:
                stats["skip_no_va"] += 1
                continue
            v, a = va
            if args.arousal_min is not None and a < args.arousal_min:
                stats["skip_low_arousal"] += 1
                continue
            orig = str(raw.get("source") or default_src).strip().lower() or default_src
            structure: dict = {"bpm": round(bpm, 1), "bars": int(st.get("bars") or 8)}
            for k in ("key", "mode", "progression", "energy", "bars_per_chord"):
                if st.get(k) is not None:
                    structure[k] = st[k]
            # 専用ソース名にして WeightedRandomSampler で高速帯を別バケットとして均等化
            rows.append(
                {
                    "source": "high_bpm",
                    "source_orig": orig,
                    "clip": raw.get("clip") or raw.get("id") or raw.get("midi_path") or "",
                    "va": {"valence": v, "arousal": a},
                    "va_source": raw.get("va_source") or orig,
                    "structure": structure,
                    "filter": f"bpm>={args.bpm_min:g}",
                }
            )
            stats[f"ok:{orig}"] += 1

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 便利用: 高 BPM ∩ 高め A
    hi = [r for r in rows if r["va"]["arousal"] >= 0.5]
    out_hi = out_dir / "structure_pairs_highA.jsonl"
    with out_hi.open("w", encoding="utf-8") as f:
        for row in hi:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"bpm_min={args.bpm_min} arousal_min={args.arousal_min}")
    print(f"highBPM n={len(rows)} -> {out_path}")
    print(f"highBPM & A>=0.5 n={len(hi)} -> {out_hi}")
    print("by_source", {k: stats[k] for k in sorted(stats) if k.startswith("ok:")})
    if rows:
        bpms = [r["structure"]["bpm"] for r in rows]
        print(
            "bpm mean/min/max",
            round(sum(bpms) / len(bpms), 1),
            min(bpms),
            max(bpms),
        )


if __name__ == "__main__":
    main()
