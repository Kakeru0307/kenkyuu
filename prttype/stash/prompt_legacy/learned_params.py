"""合成データの manifest（台帳）から進行・BPM・キーを選択する【実験用】。

本実装ではない。U-Net 本線は `generate_backing.py`。
このモジュールは `--use-manifest` 時のみ使われるルックアップであり、
ネットワーク推論ではない。
"""

刻み N はここでは選ばない（モデルが創造。条件は BPM チャンネル）。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "data" / "raw" / "synthetic" / "manifest.json"

_EMOTION_PREF: dict[str, dict[str, Any]] = {
    "joy": {
        "modes": ("major",),
        "families": ("diatonic_major",),
        "bpm": (100, 140),
    },
    "sadness": {
        "modes": ("natural_minor",),
        "families": ("diatonic_minor",),
        "bpm": (60, 100),
    },
    "calm": {
        "modes": ("natural_minor", "major"),
        "families": ("diatonic_minor", "diatonic_major"),
        "bpm": (60, 105),
    },
    "tension": {
        "modes": ("natural_minor", "major"),
        "families": ("borrowed", "blues", "diatonic_minor"),
        "bpm": (110, 150),
    },
    "na": {
        "modes": ("major", "natural_minor"),
        "families": None,
        "bpm": (70, 140),
    },
}


@dataclass(frozen=True)
class LearnedSample:
    progression: str
    key: str
    bpm: float
    bars: int
    bars_per_chord: int
    mode: str
    family: str
    source_file: str


def _iter_trainable_entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        if not e.get("progression"):
            continue
        if e.get("bpm") is None or e.get("key") is None:
            continue
        out.append(e)
    return out


@lru_cache(maxsize=4)
def load_manifest_entries(manifest_path: str) -> tuple[dict[str, Any], ...]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest が見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = _iter_trainable_entries(data.get("entries") or [])
    if not entries:
        raise RuntimeError(f"選択可能な進行つきエントリがありません: {path}")
    return tuple(entries)


def clear_manifest_cache() -> None:
    load_manifest_entries.cache_clear()


def _match(
    entry: dict[str, Any],
    *,
    modes: tuple[str, ...] | None,
    families: tuple[str, ...] | None,
    bpm_lo: float | None,
    bpm_hi: float | None,
) -> bool:
    if modes is not None and entry.get("mode") not in modes:
        return False
    if families is not None and entry.get("family") not in families:
        return False
    bpm = float(entry["bpm"])
    if bpm_lo is not None and bpm < bpm_lo:
        return False
    if bpm_hi is not None and bpm > bpm_hi:
        return False
    return True


def _filter_candidates(
    entries: tuple[dict[str, Any], ...],
    *,
    emotion: str,
    bpm_hint: float | None = None,
) -> list[dict[str, Any]]:
    pref = _EMOTION_PREF.get(emotion, _EMOTION_PREF["na"])
    modes = pref["modes"]
    families = pref["families"]
    bpm_lo, bpm_hi = pref["bpm"]

    if bpm_hint is not None:
        bpm_lo = max(60.0, bpm_hint - 20)
        bpm_hi = min(150.0, bpm_hint + 20)

    stages: list[dict[str, Any]] = [
        {"modes": modes, "families": families, "bpm_lo": bpm_lo, "bpm_hi": bpm_hi},
        {"modes": modes, "families": None, "bpm_lo": bpm_lo, "bpm_hi": bpm_hi},
        {"modes": modes, "families": None, "bpm_lo": None, "bpm_hi": None},
        {"modes": None, "families": None, "bpm_lo": None, "bpm_hi": None},
    ]

    for stage in stages:
        hit = [
            e
            for e in entries
            if _match(
                e,
                modes=stage["modes"],
                families=stage["families"],
                bpm_lo=stage["bpm_lo"],
                bpm_hi=stage["bpm_hi"],
            )
        ]
        if hit:
            return hit
    return list(entries)


def sample_from_manifest(
    *,
    emotion: str,
    rng: random.Random,
    manifest_path: Path | None = None,
    bpm_hint: float | None = None,
) -> LearnedSample:
    """学習エントリを1本サンプリング（進行・キー・BPM）。"""
    path = manifest_path or DEFAULT_MANIFEST
    entries = load_manifest_entries(str(path))
    candidates = _filter_candidates(entries, emotion=emotion, bpm_hint=bpm_hint)
    chosen = rng.choice(candidates)
    return LearnedSample(
        progression=str(chosen["progression"]),
        key=str(chosen["key"]),
        bpm=float(chosen["bpm"]),
        bars=max(8, int(chosen.get("bars") or 8)),
        bars_per_chord=max(1, int(chosen.get("bars_per_chord") or 1)),
        mode=str(chosen.get("mode") or ""),
        family=str(chosen.get("family") or ""),
        source_file=str(chosen.get("file") or ""),
    )
