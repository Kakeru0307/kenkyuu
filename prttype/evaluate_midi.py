"""生成 MIDI の客観評価（MusPy）。

評価軸の正本: docs/evaluation-criteria-slides.md
美学ではなく技術的妥当性（破綻・分布の異常）を見る。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import muspy

# makeData/constants.KEYS と揃える（メジャー名）
_PC = {
    "C": 0,
    "Db": 1,
    "C#": 1,
    "D": 2,
    "Eb": 3,
    "D#": 3,
    "E": 4,
    "F": 5,
    "Gb": 6,
    "F#": 6,
    "G": 7,
    "Ab": 8,
    "G#": 8,
    "A": 9,
    "Bb": 10,
    "A#": 10,
    "B": 11,
}

_MUSPY_MODE = {
    "major": "major",
    "minor": "minor",
    "natural_minor": "minor",
    "harmonic_minor": "minor",
    "melodic_minor": "minor",
}


def _finite(x: float | int | None) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def parse_key_root(key: str | None) -> int | None:
    if not key:
        return None
    k = key.strip()
    # "Am" / "F#m" → root only（mode は別引数）
    if len(k) >= 2 and k[-1] in ("m", "M") and k[:-1] in _PC:
        return _PC[k[:-1]]
    if k in _PC:
        return _PC[k]
    if k.rstrip("m") in _PC:
        return _PC[k.rstrip("m")]
    return None


def parse_mode(mode: str | None, key: str | None = None) -> str | None:
    if mode:
        m = mode.strip().lower()
        if m in _MUSPY_MODE:
            return _MUSPY_MODE[m]
    if key:
        k = key.strip()
        if len(k) >= 2 and k.endswith("m") and not k.endswith("M"):
            return "minor"
    return None


def evaluate_midi(
    midi_path: Path | str,
    *,
    key: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """単曲の MusPy 指標を返す。

    Returns keys aligned with slides M1–M4 (+ helpers).
    M5 (Yang–Lerch OA/KLD) needs a reference corpus → omitted here.
    """
    path = Path(midi_path)
    music = muspy.read_midi(str(path))

    root = parse_key_root(key)
    muspy_mode = parse_mode(mode, key)

    pitch_in_scale: float | None = None
    if root is not None and muspy_mode is not None:
        pitch_in_scale = _finite(muspy.pitch_in_scale_rate(music, root, muspy_mode))

    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "key": key,
        "mode": muspy_mode or mode,
        "M1_pitch_in_scale_rate": pitch_in_scale,
        "M2_empty_beat_rate": _finite(muspy.empty_beat_rate(music)),
        "M3_polyphony": _finite(muspy.polyphony(music)),
        "M3b_polyphony_rate": _finite(muspy.polyphony_rate(music)),
        "M4_scale_consistency": _finite(muspy.scale_consistency(music)),
        "pitch_class_entropy": _finite(muspy.pitch_class_entropy(music)),
        "n_pitches_used": muspy.n_pitches_used(music),
        "n_pitch_classes_used": muspy.n_pitch_classes_used(music),
        "pitch_range": muspy.pitch_range(music),
        "note": (
            "Objective sanity only (not aesthetic quality). "
            "M5 Yang-Lerch OA/KLD requires a reference set (not computed)."
        ),
    }
    return result


def format_report(metrics: dict[str, Any]) -> str:
    lines = [
        "=== MusPy objective eval (validity) ===",
        f"file: {metrics.get('path')}",
    ]
    if metrics.get("key") or metrics.get("mode"):
        lines.append(f"key/mode: {metrics.get('key')} / {metrics.get('mode')}")

    rows = [
        ("M1 pitch_in_scale_rate", metrics.get("M1_pitch_in_scale_rate")),
        ("M2 empty_beat_rate", metrics.get("M2_empty_beat_rate")),
        ("M3 polyphony", metrics.get("M3_polyphony")),
        ("M3b polyphony_rate", metrics.get("M3b_polyphony_rate")),
        ("M4 scale_consistency", metrics.get("M4_scale_consistency")),
        ("pitch_class_entropy", metrics.get("pitch_class_entropy")),
        ("n_pitches_used", metrics.get("n_pitches_used")),
        ("n_pitch_classes_used", metrics.get("n_pitch_classes_used")),
        ("pitch_range", metrics.get("pitch_range")),
    ]
    for name, val in rows:
        if val is None:
            lines.append(f"  {name}: (n/a)")
        elif isinstance(val, float):
            lines.append(f"  {name}: {val:.4f}")
        else:
            lines.append(f"  {name}: {val}")
    lines.append("  M5 OA/KLD: skipped (needs reference corpus)")
    lines.append(str(metrics.get("note", "")))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MIDI with MusPy metrics")
    parser.add_argument("midi", type=Path, help="Path to .mid")
    parser.add_argument("--key", type=str, default=None, help="e.g. C, Db, Am")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="major | minor | natural_minor (for M1)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    if not args.midi.is_file():
        raise SystemExit(f"MIDI not found: {args.midi}")

    metrics = evaluate_midi(args.midi, key=args.key, mode=args.mode)
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print(format_report(metrics))


if __name__ == "__main__":
    main()
