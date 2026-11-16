"""WRIME → 文→VA JSONL（客観ラベル優先）。

極性 → V: [-2,2] → [-1,1]
8感情強度 → A: 高覚醒感情の平均 − 低覚醒感情の平均 を [-1,1] にクリップ

出力: data/condition_encoder/wrime_va.jsonl
  {"text": "...", "va": {"valence": v, "arousal": a}, "source": "wrime"}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "condition_encoder" / "wrime_va.jsonl"

# Plutchik 8感情の覚醒寄与（固定。docs と同期）
_HIGH_A = ("joy", "anger", "surprise", "fear")
_LOW_A = ("sadness", "disgust", "trust", "anticipation")
_EMOTION_KEYS = _HIGH_A + _LOW_A


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def wrime_row_to_va(row: dict) -> tuple[float, float] | None:
    """1行から (V,A)。客観 Avg.* を優先、無ければ Writer_*。"""
    def _get_f(keys: list[str]) -> float | None:
        for k in keys:
            if k in row and str(row[k]).strip() != "":
                try:
                    return float(row[k])
                except ValueError:
                    continue
        return None

    polarity = _get_f(
        [
            "Avg. Readers_Polarity",
            "Avg.Readers_Polarity",
            "Reader_Polarity",
            "Writer_Polarity",
        ]
    )
    if polarity is None:
        return None
    v = max(-1.0, min(1.0, polarity / 2.0))

    highs: list[float] = []
    lows: list[float] = []
    for emo in _HIGH_A:
        x = _get_f(
            [
                f"Avg. Readers_{emo.capitalize()}",
                f"Avg.Readers_{emo.capitalize()}",
                f"Reader_{emo.capitalize()}",
                f"Writer_{emo.capitalize()}",
                f"Avg. Readers_{emo}",
                f"Writer_{emo}",
            ]
        )
        if x is not None:
            highs.append(x)
    for emo in _LOW_A:
        x = _get_f(
            [
                f"Avg. Readers_{emo.capitalize()}",
                f"Avg.Readers_{emo.capitalize()}",
                f"Reader_{emo.capitalize()}",
                f"Writer_{emo.capitalize()}",
                f"Avg. Readers_{emo}",
                f"Writer_{emo}",
            ]
        )
        if x is not None:
            lows.append(x)
    if not highs and not lows:
        a = 0.0
    else:
        # WRIME 強度はおおむね 0..3。差を [-1,1] に収める
        raw = _avg(highs) - _avg(lows)
        a = max(-1.0, min(1.0, raw / 3.0))
    return v, a


def _text_from_row(row: dict) -> str:
    for k in ("Sentence", "sentence", "text", "Text"):
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    return ""


def load_wrime_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_from_hf(split: str = "train") -> list[dict]:
    """HuggingFace shunk031/wrime から読む（ネット必須）。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets が必要: pip install datasets") from exc
    ds = load_dataset("shunk031/wrime", name="ver2", split=split)
    rows: list[dict] = []
    for ex in ds:
        text = str(ex.get("sentence") or "").strip()
        if not text:
            continue
        # ver2: readers 平均が dict
        readers = ex.get("avg_readers") or ex.get("readers") or {}
        writer = ex.get("writer") or {}
        polarity = None
        if isinstance(readers, dict) and "polarity" in readers:
            polarity = float(readers["polarity"])
        elif isinstance(writer, dict) and "polarity" in writer:
            polarity = float(writer["polarity"])
        if polarity is None:
            continue
        v = max(-1.0, min(1.0, polarity / 2.0))

        def _emo_avg(src: dict, keys: tuple[str, ...]) -> float:
            vals = []
            for k in keys:
                if k in src:
                    vals.append(float(src[k]))
            return _avg(vals)

        src = readers if isinstance(readers, dict) else writer
        if not isinstance(src, dict):
            src = {}
        raw_a = _emo_avg(src, _HIGH_A) - _emo_avg(src, _LOW_A)
        a = max(-1.0, min(1.0, raw_a / 3.0))
        rows.append(
            {
                "text": text,
                "va": {"valence": v, "arousal": a},
                "source": "wrime",
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=None, help="WRIME CSV（あれば優先）")
    ap.add_argument("--hf", action="store_true", help="HuggingFace shunk031/wrime ver2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out_rows: list[dict] = []
    if args.csv is not None:
        for row in load_wrime_csv(args.csv):
            text = _text_from_row(row)
            va = wrime_row_to_va(row)
            if not text or va is None:
                continue
            v, a = va
            out_rows.append(
                {"text": text, "va": {"valence": v, "arousal": a}, "source": "wrime"}
            )
    elif args.hf:
        out_rows = build_from_hf(args.split)
    else:
        raise SystemExit("--csv か --hf を指定してください")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
