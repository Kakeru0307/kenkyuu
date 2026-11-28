"""THETA (sakai25) → 文→VA JSONL。

text_ja + 聴取者の x_text,y_text 平均を使う。
入力: リポジトリを clone したディレクトリ（data_pairs.txt + perceived_emotion.txt）

出力: data/condition_encoder/theta_va.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "condition_encoder" / "theta_va.jsonl"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        row = {header[i]: parts[i] if i < len(parts) else "" for i in range(len(header))}
        rows.append(row)
    return rows


def build(theta_dir: Path) -> list[dict]:
    pairs_path = theta_dir / "data_pairs.txt"
    emo_path = theta_dir / "perceived_emotion.txt"
    if not pairs_path.is_file() or not emo_path.is_file():
        raise FileNotFoundError(
            f"THETA に data_pairs.txt / perceived_emotion.txt が必要: {theta_dir}"
        )
    pairs = {r["song_id"]: r for r in _read_tsv(pairs_path) if r.get("song_id")}
    sums: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in _read_tsv(emo_path):
        sid = r.get("song_id") or r.get("song_id".lower())
        if not sid:
            continue
        try:
            x = float(r["x_text"])
            y = float(r["y_text"])
        except (KeyError, ValueError):
            continue
        sums[sid].append((x, y))

    out: list[dict] = []
    for sid, ratings in sums.items():
        pair = pairs.get(sid)
        if not pair:
            continue
        text = (pair.get("text_ja") or "").strip()
        if not text:
            continue
        v = sum(t[0] for t in ratings) / len(ratings)
        a = sum(t[1] for t in ratings) / len(ratings)
        v = max(-1.0, min(1.0, v))
        a = max(-1.0, min(1.0, a))
        out.append(
            {
                "text": text,
                "va": {"valence": v, "arousal": a},
                "source": "theta",
                "song_id": sid,
                "n_raters": len(ratings),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--theta-dir",
        type=Path,
        required=True,
        help="takamichi-lab/sakai25_music_dataset のルート",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    rows = build(args.theta_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
