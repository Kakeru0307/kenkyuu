"""ベース学習用の input/target パッチペアを生成する。

input  : ギターコード骨格 (downbeat_chord tonal 11ch) + BPM cond
target : ベースライン (tonal 11ch, category 4 のみ非ゼロ)

ドラムと同一 seed / progression / key / bpm / beat_type で生成し、
kick_times を渡して協調させる。beat_type は manifest 記録のみ（bass U-Net 非依存）。
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Literal

import muspy
import numpy as np

from density_cond import bpm_to_unit
from makeData.bass import generate_bass_line
from makeData.constants import BEAT_BPM_RANGE, BEAT_TYPES, DEFAULT_BARS, KEYS
from makeData.drum import extract_kick_times, generate_drum_pattern
from makeData.patterns import choose_progression
from midi_to_patch import MidiPatch, midi_to_patches
from progression_input import build_backing_skeleton_music

SCRIPT_DIR = Path(__file__).resolve().parent
EnergyLevel = Literal["low", "mid", "high"]


def _music_to_patches(music: muspy.Music, tmp_midi: Path) -> list[MidiPatch]:
    muspy.write_midi(tmp_midi, music)
    return midi_to_patches(tmp_midi)


def _has_onsets(patch: MidiPatch, min_onsets: int) -> bool:
    return int((patch.tonal_chw == 1).sum()) >= min_onsets


def _sample_energy(rng: random.Random, bpm: float) -> EnergyLevel:
    if bpm < 85:
        return rng.choices(("low", "mid", "high"), weights=(0.45, 0.40, 0.15), k=1)[0]
    if bpm < 120:
        return rng.choices(("low", "mid", "high"), weights=(0.20, 0.55, 0.25), k=1)[0]
    return rng.choices(("low", "mid", "high"), weights=(0.10, 0.35, 0.55), k=1)[0]


def generate_bass_pairs(
    pairs_dir: Path,
    *,
    count: int,
    seed: int = 42,
    bars: int = DEFAULT_BARS,
    min_onsets: int = 1,
) -> dict:
    input_root = pairs_dir / "input"
    target_root = pairs_dir / "target"
    rng = random.Random(seed)

    stats: dict = {
        "mode": "bass",
        "pairs_dir": str(pairs_dir),
        "songs": [],
        "total_patches": 0,
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = Path(tmp) / "in.mid"
        tmp_tg = Path(tmp) / "tg.mid"
        for i in range(count):
            # drum と同一乱数列にするため、曲ごとの RNG を派生する
            song_rng = random.Random(rng.randint(0, 2**31 - 1))
            beat_type = BEAT_TYPES[i % len(BEAT_TYPES)]
            lo, hi = BEAT_BPM_RANGE[beat_type]
            bpm = float(song_rng.randint(lo, hi))

            spec = choose_progression(song_rng)
            key = song_rng.choice(KEYS)
            bars_per_chord = song_rng.choice((1, 1, 1, 2))
            energy = _sample_energy(song_rng, bpm)

            drum_music = generate_drum_pattern(
                bpm=bpm,
                bars=bars,
                beat_type=beat_type,
                rng=song_rng,
            )
            kick_times = extract_kick_times(drum_music)

            target_music = generate_bass_line(
                spec=spec,
                key=key,
                bpm=bpm,
                bars=bars,
                bars_per_chord=bars_per_chord,
                energy=energy,
                kick_times=kick_times,
                rng=song_rng,
            )
            input_music = build_backing_skeleton_music(
                progression=spec,
                key=key,
                bars=bars,
                bpm=bpm,
                bars_per_chord=bars_per_chord,
            )

            tgt_patches = _music_to_patches(target_music, tmp_tg)
            inp_patches = _music_to_patches(input_music, tmp_in)
            if not tgt_patches or not inp_patches:
                continue

            song_id = f"bass{i:05d}"
            in_dir = input_root / song_id
            tg_dir = target_root / song_id
            in_dir.mkdir(parents=True, exist_ok=True)
            tg_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for inp, tgt in zip(inp_patches, tgt_patches):
                if not _has_onsets(tgt, min_onsets):
                    continue
                stem = f"bar{tgt.bar_index:04d}"
                np.save(
                    in_dir / f"{stem}_tonal.npy",
                    inp.tonal_chw.astype(np.uint8, copy=False),
                )
                np.save(
                    tg_dir / f"{stem}_tonal.npy",
                    tgt.tonal_chw.astype(np.uint8, copy=False),
                )
                np.save(in_dir / f"{stem}_cond.npy", np.float32(bpm_to_unit(bpm)))
                saved += 1

            stats["songs"].append(
                {
                    "song_id": song_id,
                    "progression": spec.name,
                    "key": key,
                    "bpm": bpm,
                    "energy": energy,
                    "beat_type": beat_type,
                    "kick_times": len(kick_times),
                    "patches": saved,
                }
            )
            stats["total_patches"] += saved
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{count} 本 ... 累計 {stats['total_patches']} パッチ")

    pairs_dir.mkdir(parents=True, exist_ok=True)
    with open(pairs_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"合計: {stats['total_patches']} パッチ / {len(stats['songs'])} 本")
    print(f"保存先: {pairs_dir}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="ベース学習用 input/target ペアを生成")
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "pairs" / "bass",
    )
    parser.add_argument("--count", type=int, default=2000, help="生成する曲数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bars", type=int, default=DEFAULT_BARS)
    parser.add_argument("--min-onsets", type=int, default=1)
    args = parser.parse_args()

    generate_bass_pairs(
        args.pairs_dir,
        count=args.count,
        seed=args.seed,
        bars=args.bars,
        min_onsets=args.min_onsets,
    )


if __name__ == "__main__":
    main()
