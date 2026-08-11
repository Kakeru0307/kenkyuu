"""ドラム学習用の input/target パッチペアを生成する。

input  : ギターコード骨格 (downbeat_chord tonal 11ch) + BPM cond + beat one-hot id
target : ドラムパターン (*_drum.npy, shape (1,128,128), 0/1 二値)

beat_type を均等サンプリングし、BPM は BEAT_BPM_RANGE[beat_type] から取る。
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import muspy
import numpy as np

from density_cond import bpm_to_unit
from makeData.constants import BEAT_BPM_RANGE, BEAT_TYPES, DEFAULT_BARS, KEYS
from makeData.drum import beat_type_to_id, generate_drum_pattern
from makeData.patterns import choose_progression
from midi_to_patch import MidiPatch, midi_to_patches
from progression_input import build_backing_skeleton_music

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COUNT = 7200


def _music_to_patches(music: muspy.Music, tmp_midi: Path) -> list[MidiPatch]:
    muspy.write_midi(tmp_midi, music)
    return midi_to_patches(tmp_midi)


def _binarize_drum(drum_chw: np.ndarray) -> np.ndarray:
    """onset=1 / sustain=2 → 0/1 二値 (1,128,128)。"""
    return (drum_chw > 0).astype(np.uint8)


def _has_drum_hits(patch: MidiPatch, min_hits: int) -> bool:
    return int((patch.drum > 0).sum()) >= min_hits


def _sample_bpm(rng: random.Random, beat_type: str) -> float:
    lo, hi = BEAT_BPM_RANGE[beat_type]
    return float(rng.randint(lo, hi))


def generate_drum_pairs(
    pairs_dir: Path,
    *,
    count: int,
    seed: int = 42,
    bars: int = DEFAULT_BARS,
    min_hits: int = 4,
) -> dict:
    input_root = pairs_dir / "input"
    target_root = pairs_dir / "target"
    rng = random.Random(seed)

    stats: dict = {
        "mode": "drum",
        "pairs_dir": str(pairs_dir),
        "beat_types": list(BEAT_TYPES),
        "songs": [],
        "total_patches": 0,
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = Path(tmp) / "in.mid"
        tmp_tg = Path(tmp) / "tg.mid"
        for i in range(count):
            song_rng = random.Random(rng.randint(0, 2**31 - 1))
            # 型を均等サンプリング（型あたり count/12 程度）
            beat_type = BEAT_TYPES[i % len(BEAT_TYPES)]

            spec = choose_progression(song_rng)
            key = song_rng.choice(KEYS)
            bpm = _sample_bpm(song_rng, beat_type)
            bars_per_chord = song_rng.choice((1, 1, 1, 2))
            beat_id = beat_type_to_id(beat_type)

            target_music = generate_drum_pattern(
                bpm=bpm,
                bars=bars,
                beat_type=beat_type,
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

            song_id = f"drum{i:05d}"
            in_dir = input_root / song_id
            tg_dir = target_root / song_id
            in_dir.mkdir(parents=True, exist_ok=True)
            tg_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for inp, tgt in zip(inp_patches, tgt_patches):
                if not _has_drum_hits(tgt, min_hits):
                    continue
                stem = f"bar{tgt.bar_index:04d}"
                np.save(
                    in_dir / f"{stem}_tonal.npy",
                    inp.tonal_chw.astype(np.uint8, copy=False),
                )
                np.save(
                    tg_dir / f"{stem}_drum.npy",
                    _binarize_drum(tgt.drum_chw),
                )
                np.save(in_dir / f"{stem}_cond.npy", np.float32(bpm_to_unit(bpm)))
                np.save(in_dir / f"{stem}_beat.npy", np.int64(beat_id))
                saved += 1

            stats["songs"].append(
                {
                    "song_id": song_id,
                    "progression": spec.name,
                    "key": key,
                    "bpm": bpm,
                    "beat_type": beat_type,
                    "beat_id": beat_id,
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
    parser = argparse.ArgumentParser(description="ドラム学習用 input/target ペアを生成")
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "pairs" / "drum",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="生成する曲数（既定 7200 = 12型×600）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bars", type=int, default=DEFAULT_BARS)
    parser.add_argument("--min-hits", type=int, default=4)
    args = parser.parse_args()

    generate_drum_pairs(
        args.pairs_dir,
        count=args.count,
        seed=args.seed,
        bars=args.bars,
        min_hits=args.min_hits,
    )


if __name__ == "__main__":
    main()
