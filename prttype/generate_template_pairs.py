"""template prior 学習データ生成。

VA ルールベース重み（_template_weights）で各テンプレの選択確率を計算し、
VA / BPM をランダムサンプリングして「正解テンプレ」を確率的に割り当てる。
生成データは data/template_pairs/ に JSONL 形式で保存する。

使い方:
    python generate_template_pairs.py --count 5000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "data" / "template_pairs" / "pairs.jsonl"

BPM_MIN, BPM_MAX = 60, 160


def _bpm_to_unit(bpm: float) -> float:
    return (bpm - BPM_MIN) / (BPM_MAX - BPM_MIN)


def generate(count: int, out_path: Path, seed: int = 0) -> None:
    from song_form import _TEMPLATES, _template_weights

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    ids = [t[0] for t in _TEMPLATES]
    records: list[dict] = []

    for _ in range(count):
        valence = rng.uniform(-1.0, 1.0)
        arousal = rng.uniform(-1.0, 1.0)
        bpm = rng.uniform(BPM_MIN, BPM_MAX)
        bpm_unit = _bpm_to_unit(bpm)

        base = [t[1] for t in _TEMPLATES]
        weights = _template_weights((valence, arousal), base)
        chosen = rng.choices(ids, weights=weights, k=1)[0]
        label = ids.index(chosen)
        records.append({
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "bpm_unit": round(bpm_unit, 4),
            "template": chosen,
            "label": label,
        })

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"生成: {len(records)} 件 → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="template prior 学習データ生成")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate(args.count, args.out, args.seed)


if __name__ == "__main__":
    main()
