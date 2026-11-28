"""Jamendo autotagging TSV → 文＋音楽条件スコア JSONL。

自作スコア禁止。タグ→軸は axis_tag_map.TAG_TO_AXIS のみ。
文はタグをそのまま短い英語句にする（タグ由来。新規意味の発明ではない）。

入力例: mtg-jamendo-dataset の autotagging_moodtheme.tsv / autotagging_genre.tsv
列: TRACK_ID と tags（スペース区切りで genre---rock 等）

出力: data/condition_encoder/music_axes.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.music_condition.axis_tag_map import (  # noqa: E402
    MUSIC_AXES,
    empty_scores,
    tags_to_scores,
)

DEFAULT_OUT = ROOT / "data" / "condition_encoder" / "music_axes.jsonl"


def _parse_tags_field(raw: str) -> list[str]:
    # Jamendo: "genre---rock mood/theme---energetic" or comma-separated
    parts = []
    for tok in raw.replace(",", " ").split():
        tok = tok.strip()
        if tok:
            parts.append(tok)
    return parts


def _text_from_tags(tags: list[str]) -> str:
    names = []
    for t in tags:
        name = t.split("---")[-1] if "---" in t else t
        names.append(name.replace("_", " "))
    # 実タグ列挙のみ
    return " ".join(names) + " music"


def load_jamendo_tsv(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("TRACK"):
            # header-ish
            if "\t" in line and "TRACK" in line.upper():
                continue
            if not line:
                continue
        # formats vary: id\ttag1 tag2  OR path\ttags
        if "\t" in line:
            left, right = line.split("\t", 1)
        else:
            continue
        tags = _parse_tags_field(right)
        if not tags:
            continue
        scores = tags_to_scores(tags)
        if all(v <= 0.0 for v in scores.values()):
            continue
        rows.append(
            {
                "text": _text_from_tags(tags),
                "scores": scores,
                "source": "jamendo",
                "track": left,
                "tags": tags,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tsv",
        type=Path,
        nargs="+",
        required=True,
        help="Jamendo autotagging_*.tsv を1つ以上",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-rows", type=int, default=0, help="0=無制限")
    args = ap.parse_args()

    all_rows: list[dict] = []
    for tsv in args.tsv:
        all_rows.extend(load_jamendo_tsv(tsv))
    if args.max_rows > 0:
        all_rows = all_rows[: args.max_rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"axes={list(MUSIC_AXES)}")
    print(f"wrote {len(all_rows)} rows → {args.out}")
    # 空 scores テンプレ確認用
    _ = empty_scores()


if __name__ == "__main__":
    main()
