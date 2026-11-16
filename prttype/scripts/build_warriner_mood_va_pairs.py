"""Warriner VAD × 音楽 mood 語 → 短い文→VA JSONL。

VA は辞書の人手評定のみ使う（発明しない）。
文は「{lemma} music」/「{lemma}な曲」の定型。語の VA をそのまま載せる。

--warriner: Word / V.Mean.Sum / A.Mean.Sum 列のある TSV/CSV
  （Warriner et al. 2013 Ratings_Warriner_et_al.csv 等）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.music_condition.axis_tag_map import MUSIC_MOOD_LEMMAS  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "condition_encoder" / "warriner_mood_va.jsonl"


def _load_warriner(path: Path) -> dict[str, tuple[float, float]]:
    """lemma → (V,A) in [-1,1]。元スケールはおおむね 1..9。"""
    with path.open(encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(f, dialect=dialect)
        fieldmap = {k.lower().strip(): k for k in (reader.fieldnames or [])}

        def col(*cands: str) -> str | None:
            for c in cands:
                if c.lower() in fieldmap:
                    return fieldmap[c.lower()]
            return None

        word_k = col("Word", "word", "Lemma", "lemma")
        v_k = col("V.Mean.Sum", "Valence", "valence", "V.Mean")
        a_k = col("A.Mean.Sum", "Arousal", "arousal", "A.Mean")
        if not word_k or not v_k or not a_k:
            raise SystemExit(
                f"Warriner 列が見つからない: {list(fieldmap)} "
                "(Word, V.Mean.Sum, A.Mean.Sum が必要)"
            )
        out: dict[str, tuple[float, float]] = {}
        for row in reader:
            w = str(row[word_k]).strip().lower()
            if not w:
                continue
            try:
                v_raw = float(row[v_k])
                a_raw = float(row[a_k])
            except ValueError:
                continue
            # 1..9 → -1..1
            v = max(-1.0, min(1.0, (v_raw - 5.0) / 4.0))
            a = max(-1.0, min(1.0, (a_raw - 5.0) / 4.0))
            out[w] = (v, a)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warriner", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    lexicon = _load_warriner(args.warriner)
    rows: list[dict] = []
    missing: list[str] = []
    for lemma in MUSIC_MOOD_LEMMAS:
        if lemma not in lexicon:
            missing.append(lemma)
            continue
        v, a = lexicon[lemma]
        for text in (f"{lemma} music", f"{lemma}な曲", f"a {lemma} song"):
            rows.append(
                {
                    "text": text,
                    "va": {"valence": v, "arousal": a},
                    "source": "warriner_mood",
                    "lemma": lemma,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows → {args.out}")
    if missing:
        print(f"missing lemmas (skipped): {missing}")


if __name__ == "__main__":
    main()
