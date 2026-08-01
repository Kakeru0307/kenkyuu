"""合成時の BPM↔N 相関（推論の骨格条件とは独立）。"""

from __future__ import annotations

import random

from .constants import ATTACKS_PER_BAR_WEIGHTS


def preferred_attacks_range(bpm: float) -> tuple[int, int]:
    if bpm < 80:
        return (1, 4)
    if bpm < 110:
        return (3, 8)
    if bpm < 130:
        return (5, 10)
    return (6, 16)


def sample_attacks_for_bpm(bpm: float, rng: random.Random) -> int:
    # 重みが1種だけなら BPM 帯に関わらずそれを使う（例: 8分固定実験）
    if len(ATTACKS_PER_BAR_WEIGHTS) == 1:
        return int(next(iter(ATTACKS_PER_BAR_WEIGHTS)))
    lo, hi = preferred_attacks_range(float(bpm))
    candidates = [n for n in ATTACKS_PER_BAR_WEIGHTS if lo <= n <= hi]
    if not candidates:
        candidates = [n for n in ATTACKS_PER_BAR_WEIGHTS if n <= hi] or list(
            ATTACKS_PER_BAR_WEIGHTS.keys()
        )
    weights = [ATTACKS_PER_BAR_WEIGHTS[n] for n in candidates]
    return int(rng.choices(candidates, weights=weights, k=1)[0])
