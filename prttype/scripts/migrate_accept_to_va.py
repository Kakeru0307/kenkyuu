"""既存 prior_pairs accept/candidates に va フィールドを追記する。

emotion_wrime から wrime_to_va() で計算する。
emotion_wrime が空の行は va=(0.0, 0.0)。

Usage:
  python scripts/migrate_accept_to_va.py
  python scripts/migrate_accept_to_va.py --jsonl data/prior_pairs/manifests/candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_va import wrime_to_va


def migrate(path: Path) -> int:
    if not path.is_file():
        raise SystemExit(f"not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    n = 0
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        v, a = wrime_to_va(row.get("emotion_wrime"))
        row["va"] = {"valence": v, "arousal": a}
        out.append(json.dumps(row, ensure_ascii=False))
        n += 1
    path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="prior_pairs に va フィールドを追記")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "data" / "prior_pairs" / "manifests" / "accept.jsonl",
    )
    args = parser.parse_args()
    path = args.jsonl if args.jsonl.is_absolute() else (ROOT / args.jsonl)
    n = migrate(path)
    print(f"updated {n} rows → {path}")


if __name__ == "__main__":
    main()
