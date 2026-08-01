"""リード学習用の input/target パッチペアを生成する。

backingと別checkpointだが、backing power chord attack maskを条件共有する。
  input  : コード骨格11ch + BPM 1ch + power attack mask 1ch
  target : 単音主体＋一部power chordのフレーズ型リード

出力形式は prepare_dataset.py と同じ:
  <pairs_dir>/input/<song_id>/bar####_tonal.npy
  <pairs_dir>/input/<song_id>/bar####_power.npy
  <pairs_dir>/target/<song_id>/bar####_tonal.npy
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import muspy
import numpy as np

from makeData.constants import BPM_RANGE, DEFAULT_BARS, KEYS
from makeData.patterns import (
    choose_progression,
    generate_progression_lead,
    sample_backing_power_chord_onsets,
)
from midi_to_patch import PATCH_TICKS, TICKS_PER_BAR, MidiPatch, midi_to_patches
from progression_input import build_backing_skeleton_music
from density_cond import bpm_to_unit

SCRIPT_DIR = Path(__file__).resolve().parent


def _music_to_patches(music: muspy.Music, tmp_midi: Path) -> list[MidiPatch]:
    muspy.write_midi(tmp_midi, music)
    return midi_to_patches(tmp_midi)


def _has_onsets(patch: MidiPatch, min_onsets: int) -> bool:
    return int((patch.tonal_chw == 1).sum()) >= min_onsets


def _power_mask_for_patch(
    blocked_onsets: set[int],
    *,
    bar_index: int,
) -> np.ndarray:
    """(time,pitch) mask。1の時刻ではlead power chord attackを避ける。"""
    mask = np.zeros((PATCH_TICKS, 128), dtype=np.uint8)
    start = bar_index * TICKS_PER_BAR
    for absolute_time in blocked_onsets:
        relative = absolute_time - start
        if 0 <= relative < PATCH_TICKS:
            mask[relative, :] = 1
    return mask


def generate_lead_pairs(
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

    stats = {"mode": "lead", "pairs_dir": str(pairs_dir), "songs": [], "total_patches": 0}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = Path(tmp) / "in.mid"
        tmp_tg = Path(tmp) / "tg.mid"
        for i in range(count):
            spec = choose_progression(rng)
            key = rng.choice(KEYS)
            bpm = rng.randint(*BPM_RANGE)
            bars_per_chord = rng.choice((1, 1, 1, 2))
            blocked_power_onsets = sample_backing_power_chord_onsets(
                bars=bars,
                bpm=bpm,
                rng=rng,
            )

            target_music = generate_progression_lead(
                spec=spec, key=key, bpm=bpm, bars=bars,
                bars_per_chord=bars_per_chord, rng=rng,
                blocked_power_onsets=blocked_power_onsets,
            )
            input_music = build_backing_skeleton_music(
                progression=spec, key=key, bars=bars, bpm=bpm,
                bars_per_chord=bars_per_chord,
            )
            tgt_patches = _music_to_patches(target_music, tmp_tg)
            inp_patches = _music_to_patches(input_music, tmp_in)
            if not tgt_patches or not inp_patches:
                continue

            song_id = f"lead{i:05d}"
            in_dir = input_root / song_id
            tg_dir = target_root / song_id
            in_dir.mkdir(parents=True, exist_ok=True)
            tg_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for inp, tgt in zip(inp_patches, tgt_patches):
                if not _has_onsets(tgt, min_onsets):
                    continue
                stem = f"bar{tgt.bar_index:04d}"
                np.save(in_dir / f"{stem}_tonal.npy", inp.tonal_chw.astype(np.uint8, copy=False))
                np.save(tg_dir / f"{stem}_tonal.npy", tgt.tonal_chw.astype(np.uint8, copy=False))
                np.save(in_dir / f"{stem}_cond.npy", np.float32(bpm_to_unit(bpm)))
                np.save(
                    in_dir / f"{stem}_power.npy",
                    _power_mask_for_patch(
                        blocked_power_onsets,
                        bar_index=tgt.bar_index,
                    ),
                )
                saved += 1

            stats["songs"].append({
                "song_id": song_id,
                "progression": spec.name,
                "key": key,
                "patches": saved,
                "blocked_power_onsets": len(blocked_power_onsets),
            })
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
    parser = argparse.ArgumentParser(description="リード学習用 input/target ペアを生成")
    parser.add_argument("--pairs-dir", type=Path, default=SCRIPT_DIR / "data" / "pairs" / "lead")
    parser.add_argument("--count", type=int, default=6000, help="生成する曲数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bars", type=int, default=DEFAULT_BARS)
    parser.add_argument("--min-onsets", type=int, default=1)
    args = parser.parse_args()

    generate_lead_pairs(
        args.pairs_dir,
        count=args.count,
        seed=args.seed,
        bars=args.bars,
        min_onsets=args.min_onsets,
    )


if __name__ == "__main__":
    main()
