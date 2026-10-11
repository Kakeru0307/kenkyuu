"""音声感情データセット → structure prior 用 (VA, BPM) ペアの共通処理。

PMEmo / DEAM 向け。音声は学習に直接入れず、推定 BPM と正規化 VA だけを JSONL にする。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from structure_prior import BPM_HI, BPM_LO


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def normalize_va_to_pm1(
    v: float,
    a: float,
    *,
    scale: str = "auto",
) -> tuple[float, float]:
    """観測スケールを [-1, +1] に写す。

    scale:
      - \"01\": [0, 1] → [-1, 1]（PMEmo）
      - \"pm1\": 既に [-1, 1]（DEAM の多く）
      - \"sam19\": [1, 9] → [-1, 1]
      - \"auto\": コーパス統計なしの簡易推定（単行）。可能なら明示 scale を使う
    """
    vv, aa = float(v), float(a)
    mode = scale
    if mode == "auto":
        if 1.0 <= vv <= 9.0 and 1.0 <= aa <= 9.0 and (vv > 1.0 or aa > 1.0):
            mode = "sam19"
        elif -1.5 <= vv <= 1.5 and -1.5 <= aa <= 1.5:
            # 負を含む、または 0–1 に見えても pm1 扱い（誤射を避ける）
            # 呼び出し側で PMEmo は scale=\"01\" を指定すること
            mode = "pm1"
        else:
            mode = "pm1"
    if mode == "01":
        return clamp(vv * 2.0 - 1.0, -1.0, 1.0), clamp(aa * 2.0 - 1.0, -1.0, 1.0)
    if mode == "sam19":
        return clamp((vv - 5.0) / 4.0, -1.0, 1.0), clamp((aa - 5.0) / 4.0, -1.0, 1.0)
    return clamp(vv, -1.0, 1.0), clamp(aa, -1.0, 1.0)


def infer_corpus_scale(values: list[tuple[float, float]]) -> str:
    """コーパス全体から VA スケールを推定。"""
    if not values:
        return "pm1"
    flat_v = [v for v, _ in values]
    flat_a = [a for _, a in values]
    vmin, vmax = min(flat_v), max(flat_v)
    amin, amax = min(flat_a), max(flat_a)
    if vmin >= 0.0 and amin >= 0.0 and vmax <= 1.0 + 1e-6 and amax <= 1.0 + 1e-6:
        return "01"
    if vmin >= 1.0 - 1e-6 and amin >= 1.0 - 1e-6 and vmax <= 9.0 + 1e-6 and amax <= 9.0 + 1e-6:
        return "sam19"
    return "pm1"


def estimate_bpm(audio_path: Path) -> float | None:
    """librosa で tempo 推定。失敗時 None。"""
    try:
        import librosa
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "librosa が必要です: pip install librosa soundfile"
        ) from e

    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    except Exception:
        return None
    if y is None or len(y) < sr // 2:
        return None
    try:
        if hasattr(librosa, "feature") and hasattr(librosa.feature, "rhythm"):
            tempo = librosa.feature.rhythm.tempo(y=y, sr=sr)
        else:
            tempo = librosa.beat.tempo(y=y, sr=sr)  # older librosa
        if tempo is None or len(np.asarray(tempo).reshape(-1)) == 0:
            return None
        bpm = float(np.asarray(tempo).reshape(-1)[0])
    except Exception:
        return None
    if not (bpm == bpm) or bpm <= 0:  # NaN / non-positive
        return None
    return float(bpm)


def clip_bpm(bpm: float) -> float:
    return round(clamp(bpm, BPM_LO, BPM_HI), 1)


def load_bpm_cache(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def save_bpm_cache(path: Path, cache: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_bpm(
    audio_path: Path,
    *,
    cache: dict[str, float],
    cache_key: str | None = None,
) -> float | None:
    key = cache_key or str(audio_path.resolve())
    if key in cache:
        return float(cache[key])
    bpm = estimate_bpm(audio_path)
    if bpm is None:
        return None
    bpm = clip_bpm(bpm)
    cache[key] = bpm
    return bpm


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def log_va_bpm_stats(rows: list[dict[str, Any]], *, label: str) -> None:
    if not rows:
        print(f"[{label}] empty")
        return
    vs = [float(r["va"]["valence"]) for r in rows]
    as_ = [float(r["va"]["arousal"]) for r in rows]
    bpms = [float(r["structure"]["bpm"]) for r in rows]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    def std(xs: list[float]) -> float:
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    print(
        f"[{label}] n={len(rows)}  "
        f"V mean/std={mean(vs):.3f}/{std(vs):.3f}  "
        f"A mean/std={mean(as_):.3f}/{std(as_):.3f}  "
        f"|V|>=0.6={sum(1 for v in vs if abs(v) >= 0.6)}  "
        f"|A|>=0.6={sum(1 for a in as_ if abs(a) >= 0.6)}  "
        f"BPM mean/min/max={mean(bpms):.1f}/{min(bpms):.1f}/{max(bpms):.1f}"
    )
    # A 帯別 BPM
    bands = [(-1.0, -0.2, "A_low"), (-0.2, 0.2, "A_mid"), (0.2, 0.6, "A_midhigh"), (0.6, 1.01, "A_high")]
    for lo, hi, name in bands:
        xs = [b for v, a, b in zip(vs, as_, bpms) if lo <= a < hi]
        if xs:
            print(f"  {name}: n={len(xs)} bpm_mean={mean(xs):.1f}")
        else:
            print(f"  {name}: n=0")


def make_pair_row(
    *,
    source: str,
    clip: str,
    valence: float,
    arousal: float,
    bpm: float,
    va_source: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "clip": clip,
        "va": {"valence": float(valence), "arousal": float(arousal)},
        "va_source": va_source,
        "structure": {"bpm": float(bpm), "bars": 8},
    }


def count_skips(stats: Counter[str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
