"""VA×BPM 整合向けに structure pairs を整理する。

本線の学習セット（件数揃え）:
  structure_pairs_balanced_116.jsonl
    - 高VA×BPM≥150 … 最大 116 件（足りなければ全件）
    - 高V低A … 116 件
    - 低V高A … 116 件
    - 低V低A … 116 件
  → 計 最大 464 件。source は bal_* で均等サンプリング可能。

Usage:
  python scripts/organize_va_bpm_pairs.py
  python scripts/organize_va_bpm_pairs.py --per-group 116 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent

VA_SUM_HIGH = 0.4
VA_SUM_LOW = -0.4
VA_BOOST_HIGH = 0.5
BPM_SLOW = 90.0
BPM_VERY_FAST = 150.0


def row_va(row: dict[str, Any]) -> tuple[float, float] | None:
    va = row.get("va")
    if isinstance(va, dict):
        return float(va.get("valence", 0.0)), float(va.get("arousal", 0.0))
    if isinstance(va, (list, tuple)) and len(va) >= 2:
        return float(va[0]), float(va[1])
    return None


def va_quadrant(v: float, a: float) -> str:
    return ("highV" if v >= 0 else "lowV") + "_" + ("highA" if a >= 0 else "lowA")


def bpm_band(bpm: float) -> str:
    if bpm < 90:
        return "slow"
    if bpm < 120:
        return "mid"
    if bpm < 150:
        return "fast"
    return "very_fast"


def classify(v: float, a: float, bpm: float) -> str:
    s = v + a
    bb = bpm_band(bpm)
    if s >= VA_SUM_HIGH and bb == "slow":
        return "drop_high_va_slow"
    # 低V低A×超速だけ落とす。低V高A×速は通す（緊張帯の速い例を殺さない）
    if v < 0.0 and a < 0.0 and bb == "very_fast":
        return "drop_lowV_lowA_very_fast"
    return "keep"


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    v, a = row_va(row) or (0.0, 0.0)
    st = row.get("structure") or {}
    return (
        round(v, 4),
        round(a, 4),
        round(float(st.get("bpm") or 0), 1),
        st.get("progression"),
        st.get("key"),
        st.get("mode"),
        str(row.get("clip") or row.get("id") or ""),
    )


def load_rows(path: Path, default_source: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if path.name == "accept.jsonl":
            gate = (row.get("gate") or {}).get("status")
            if gate not in (None, "accept"):
                continue
        st = row.get("structure") or {}
        if st.get("bpm") is None:
            continue
        va = row_va(row)
        if va is None:
            continue
        src = str(row.get("source") or default_source).strip().lower()
        if src == "high_bpm":
            continue
        row = dict(row)
        row["source"] = src
        row["structure"] = dict(st)
        out.append(row)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_n(
    pool: list[dict[str, Any]],
    n: int,
    *,
    rng: random.Random,
    prefer_key=None,
) -> list[dict[str, Any]]:
    """prefer_key が低いほど優先。同点は乱択。"""
    if len(pool) <= n:
        out = list(pool)
        rng.shuffle(out)
        return out
    if prefer_key is None:
        return rng.sample(pool, n)
    ranked = sorted(pool, key=lambda r: (prefer_key(r), rng.random()))
    return ranked[:n]


def prefer_calm(row: dict[str, Any]) -> float:
    """低め BPM を優先（高V低A・低V低A向け）。"""
    return float(row["structure"]["bpm"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize VA×BPM balanced training pairs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "va_bpm_curated")
    parser.add_argument("--per-group", type=int, default=116)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out_dir: Path = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quad_dir = out_dir / "by_quadrant"
    quad_dir.mkdir(parents=True, exist_ok=True)
    n = int(args.per_group)
    rng = random.Random(args.seed)

    inputs = [
        (ROOT / "data" / "prior_pairs" / "manifests" / "accept.jsonl", "accept"),
        (ROOT / "data" / "vgmidi_pairs" / "structure_pairs.jsonl", "vgmidi"),
        (ROOT / "data" / "pmemo_pairs" / "structure_pairs.jsonl", "pmemo"),
        (ROOT / "data" / "fme24_pairs" / "structure_pairs.jsonl", "fme24"),
    ]

    all_rows: list[dict[str, Any]] = []
    for path, src in inputs:
        if not path.is_file():
            print(f"[skip missing] {path}")
            continue
        chunk = load_rows(path, src)
        print(f"{src}: {len(chunk)} rows")
        all_rows.extend(chunk)

    core: list[dict[str, Any]] = []
    drop_reason: Counter[str] = Counter()
    by_quad: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pool_high_fast: list[dict[str, Any]] = []
    seen_hf: set[tuple[Any, ...]] = set()
    seen_q: dict[str, set[tuple[Any, ...]]] = defaultdict(set)

    for row in all_rows:
        v, a = row_va(row) or (0.0, 0.0)
        bpm = float(row["structure"]["bpm"])
        q = va_quadrant(v, a)
        tag = classify(v, a, bpm)
        if tag != "keep":
            drop_reason[tag] += 1
            continue
        core.append(row)
        key = row_key(row)
        if key not in seen_q[q]:
            seen_q[q].add(key)
            by_quad[q].append(row)
        s = v + a
        if s >= VA_BOOST_HIGH and bpm >= BPM_VERY_FAST and key not in seen_hf:
            seen_hf.add(key)
            pool_high_fast.append(row)

    # --- 学習本線: 各グループ n 件 ---
    # 1) 高VA×≥150（目標 n。足りなければ全件）
    g_high_fast = sample_n(pool_high_fast, min(n, len(pool_high_fast)), rng=rng)
    used: set[tuple[Any, ...]] = {row_key(r) for r in g_high_fast}

    def pool_minus(q: str) -> list[dict[str, Any]]:
        return [r for r in by_quad[q] if row_key(r) not in used]

    g_hVlA = sample_n(pool_minus("highV_lowA"), n, rng=rng, prefer_key=prefer_calm)
    used |= {row_key(r) for r in g_hVlA}
    # 低V高A: 中央寄せせずプールから一様サンプル（速い例も含む）
    g_lVhA = sample_n(pool_minus("lowV_highA"), n, rng=rng)
    used |= {row_key(r) for r in g_lVhA}
    g_lVlA = sample_n(pool_minus("lowV_lowA"), n, rng=rng, prefer_key=prefer_calm)
    used |= {row_key(r) for r in g_lVlA}

    def tag_group(rows: list[dict[str, Any]], source: str, curated: str) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            r = dict(row)
            r["source"] = source
            r["source_orig"] = row.get("source")
            r["curated"] = curated
            out.append(r)
        return out

    bal_parts = [
        tag_group(g_high_fast, "bal_high_fast", "v+a>=0.5_bpm>=150"),
        tag_group(g_hVlA, "bal_highV_lowA", "highV_lowA"),
        tag_group(g_lVhA, "bal_lowV_highA", "lowV_highA"),
        tag_group(g_lVlA, "bal_lowV_lowA", "lowV_lowA"),
    ]
    balanced: list[dict[str, Any]] = []
    for part in bal_parts:
        balanced.extend(part)
    rng.shuffle(balanced)

    # 確認用: 全象限プールも残す
    write_jsonl(out_dir / "structure_pairs_core.jsonl", core)
    write_jsonl(out_dir / "structure_pairs_pool_high_fast.jsonl", pool_high_fast)
    for q, rows in by_quad.items():
        write_jsonl(quad_dir / f"{q}.jsonl", rows)
        write_jsonl(out_dir / f"structure_pairs_{q}.jsonl", rows)

    write_jsonl(out_dir / "structure_pairs_balanced_116.jsonl", balanced)
    # グループ別も残す（中身確認用）
    write_jsonl(out_dir / "balanced_part_high_fast.jsonl", bal_parts[0])
    write_jsonl(out_dir / "balanced_part_highV_lowA.jsonl", bal_parts[1])
    write_jsonl(out_dir / "balanced_part_lowV_highA.jsonl", bal_parts[2])
    write_jsonl(out_dir / "balanced_part_lowV_lowA.jsonl", bal_parts[3])

    def band_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(Counter(bpm_band(float(r["structure"]["bpm"])) for r in rows))

    report = f"""# VA×BPM 整理レポート

生成: `organize_va_bpm_pairs.py --per-group {n} --seed {args.seed}`

## 学習本線セット

**`structure_pairs_balanced_116.jsonl`**（合計 **{len(balanced)}** 件）

| グループ | source | 目標 | 実数 | BPM帯 |
|---|---|---:|---:|---|
| 高VA×BPM≥150 | bal_high_fast | {n} | {len(bal_parts[0])} | {band_counts(bal_parts[0])} |
| 高V低A | bal_highV_lowA | {n} | {len(bal_parts[1])} | {band_counts(bal_parts[1])} |
| 低V高A | bal_lowV_highA | {n} | {len(bal_parts[2])} | {band_counts(bal_parts[2])} |
| 低V低A | bal_lowV_lowA | {n} | {len(bal_parts[3])} | {band_counts(bal_parts[3])} |

プール（重複なしコア後）: high_fast={len(pool_high_fast)}, highV_lowA={len(by_quad['highV_lowA'])}, lowV_highA={len(by_quad['lowV_highA'])}, lowV_lowA={len(by_quad['lowV_lowA'])}

高V低A・低V低Aは低め BPM 優先。低V高Aは一様サンプル（≥150 専用バケットは作らない）。高VA×≥150 はプール全件が {n} 未満なら全採用。

## 学習コマンド例

```text
python scripts/train_structure_prior.py \\
  --jsonl data/prior_pairs/manifests/accept.jsonl \\
  --extra-jsonl \\
    data/emopia_pairs/structure_pairs.jsonl \\
    data/va_bpm_curated/structure_pairs_balanced_116.jsonl
```

accept は進行・キー用。VA→BPM の条件つき分布は balanced_116 が主。
EMOPIA はキー／モード用（BPM マスク）。

コア除外: {dict(drop_reason)}（計 {sum(drop_reason.values())}）
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir / 'report.md'}")
    print(
        f"balanced={len(balanced)} "
        f"(high_fast={len(bal_parts[0])} hVlA={len(bal_parts[1])} "
        f"lVhA={len(bal_parts[2])} lVlA={len(bal_parts[3])})"
    )
    print(
        f"pools high_fast={len(pool_high_fast)} "
        f"hVlA={len(by_quad['highV_lowA'])} "
        f"lVhA={len(by_quad['lowV_highA'])} "
        f"lVlA={len(by_quad['lowV_lowA'])}"
    )


if __name__ == "__main__":
    main()
