"""Guitar-TECHS P1_techniques MIDI から各テクニックの統計を抽出する。

出力: guitar_techs_stats.json
  {
    "bend":      {"duration": [...], "pitch": [...], "velocity": [...], "n_notes": N},
    "vibrato":   {...},
    "palm_mute": {...},
    "normal":    {...},
    ...
  }

使い方:
  python scripts/extract_technique_stats.py \
      --midi-dir C:/Users/kake0/GitHub/研究/guitar_techs/P1_techniques/P1_techniques/midi \
      --out kenkyuu/prttype/data/guitar_techs_stats.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import mido

# Guitar-TECHS のファイル名 → テクニックラベルのマッピング
FILE_TO_TECHNIQUE: dict[str, str] = {
    "midi_Bendings.mid": "bend",
    "midi_Vibrato.mid": "vibrato",
    "midi_PalmMute.mid": "palm_mute",
    "midi_Harmonics.mid": "harmonic",
    "midi_PinchHarmonics.mid": "pinch_harmonic",
}

# articulation_layer.py の TECHNIQUE_ID に合わせたマッピング
TECHNIQUE_TO_ID: dict[str, int] = {
    "normal": 0,
    "bend": 1,
    "vibrato": 2,
    "palm_mute": 3,
    "harmonic": 4,
    "pinch_harmonic": 5,
}


def _extract_notes(mid_path: Path) -> list[dict]:
    """MIDI から note イベントを抽出し (pitch, velocity, duration_ticks) のリストを返す。"""
    mid = mido.MidiFile(str(mid_path))
    ticks_per_beat = mid.ticks_per_beat
    notes: list[dict] = []

    for track in mid.tracks:
        abs_tick = 0
        pending: dict[int, int] = {}  # pitch → note_on absolute tick

        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending[msg.note] = (abs_tick, msg.velocity)
            elif msg.type in ("note_off",) or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                if msg.note in pending:
                    on_tick, vel = pending.pop(msg.note)
                    dur_ticks = abs_tick - on_tick
                    if dur_ticks > 0:
                        notes.append(
                            {
                                "pitch": msg.note,
                                "velocity": vel,
                                "duration_ticks": dur_ticks,
                                "duration_beats": dur_ticks / ticks_per_beat,
                            }
                        )

    return notes


def _compute_stats(notes: list[dict]) -> dict:
    if not notes:
        return {"n_notes": 0}

    pitches = [n["pitch"] for n in notes]
    velocities = [n["velocity"] for n in notes]
    durations = [n["duration_beats"] for n in notes]

    return {
        "n_notes": len(notes),
        "pitch": {
            "min": min(pitches),
            "max": max(pitches),
            "mean": round(statistics.mean(pitches), 2),
            "stdev": round(statistics.stdev(pitches) if len(pitches) > 1 else 0.0, 2),
        },
        "velocity": {
            "min": min(velocities),
            "max": max(velocities),
            "mean": round(statistics.mean(velocities), 2),
            "stdev": round(
                statistics.stdev(velocities) if len(velocities) > 1 else 0.0, 2
            ),
        },
        "duration_beats": {
            "min": round(min(durations), 4),
            "max": round(max(durations), 4),
            "mean": round(statistics.mean(durations), 4),
            "stdev": round(
                statistics.stdev(durations) if len(durations) > 1 else 0.0, 4
            ),
            "p25": round(sorted(durations)[len(durations) // 4], 4),
            "p75": round(sorted(durations)[3 * len(durations) // 4], 4),
        },
        "raw": {
            "pitches": pitches,
            "velocities": velocities,
            "duration_beats": [round(d, 4) for d in durations],
        },
    }


def extract_all(midi_dir: Path) -> dict:
    results: dict[str, dict] = {}

    for filename, technique in FILE_TO_TECHNIQUE.items():
        midi_path = midi_dir / filename
        if not midi_path.exists():
            print(f"  [skip] {filename} not found")
            continue

        notes = _extract_notes(midi_path)
        stats = _compute_stats(notes)
        stats["technique_id"] = TECHNIQUE_TO_ID.get(technique, -1)
        results[technique] = stats

        print(
            f"  {technique:15s}: {stats['n_notes']:4d} notes "
            f"| vel mean={stats['velocity']['mean']:.1f} "
            f"| dur mean={stats['duration_beats']['mean']:.3f} beats"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Guitar-TECHS 統計抽出")
    parser.add_argument(
        "--midi-dir",
        type=Path,
        default=Path(
            r"C:\Users\kake0\GitHub\研究\guitar_techs"
            r"\P1_techniques\P1_techniques\midi"
        ),
        help="Guitar-TECHS P1_techniques/midi フォルダ",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "guitar_techs_stats.json",
        help="出力 JSON パス",
    )
    args = parser.parse_args()

    print(f"Guitar-TECHS MIDI フォルダ: {args.midi_dir}")
    stats = extract_all(args.midi_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n統計を保存しました: {args.out}")
    print(f"テクニック数: {len(stats)}")


if __name__ == "__main__":
    main()
