"""raw MIDI から学習用 input/target パッチペアを一括生成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import muspy
import numpy as np

from density_cond import bpm_to_unit, midi_tempo_bpm
from midi_to_patch import MidiPatch, midi_to_patches
from skeleton import PAIR_MODES, make_input_tonal


def song_id_from_path(midi_path: Path, raw_root: Path) -> str:
    rel = midi_path.relative_to(raw_root)
    return str(rel.with_suffix("")).replace("\\", "/")


def patch_has_notes(patch: MidiPatch, *, min_onsets: int = 1) -> bool:
    onsets = int((patch.tonal_chw == 1).sum())
    return onsets >= min_onsets


SCRIPT_DIR = Path(__file__).resolve().parent


def save_pair_patches(
    patches: list[MidiPatch],
    input_dir: Path,
    target_dir: Path,
    *,
    mode: str,
    min_onsets: int,
    bpm: float,
    first_patch_only: bool = False,
) -> int:
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    cond_unit = np.float32(bpm_to_unit(bpm))

    saved = 0
    for patch in patches:
        if first_patch_only and patch.bar_index > 0:
            continue
        if not patch_has_notes(patch, min_onsets=min_onsets):
            continue

        target_tonal = patch.tonal_chw
        input_tonal = make_input_tonal(target_tonal, mode)
        stem = f"bar{patch.bar_index:04d}"

        np.save(input_dir / f"{stem}_tonal.npy", input_tonal.astype(np.uint8, copy=False))
        np.save(target_dir / f"{stem}_tonal.npy", target_tonal.astype(np.uint8, copy=False))
        np.save(input_dir / f"{stem}_cond.npy", cond_unit)
        saved += 1

    return saved


def prepare_pairs(
    raw_dir: Path,
    pairs_dir: Path,
    *,
    mode: str = "onset_to_full",
    min_onsets: int = 1,
    categories: list[str] | None = None,
    midi_files: list[Path] | None = None,
    first_patch_only: bool = False,
) -> dict:
    input_root = pairs_dir / "input"
    target_root = pairs_dir / "target"
    if midi_files is not None:
        files = sorted(midi_files)
    else:
        files = sorted(raw_dir.rglob("*.mid"))
        if categories:
            files = [
                path
                for path in files
                if path.relative_to(raw_dir).parts[0] in categories
            ]

    if not files:
        raise FileNotFoundError(f"MIDI が見つかりません: {raw_dir}")

    stats = {
        "mode": mode,
        "raw_dir": str(raw_dir),
        "pairs_dir": str(pairs_dir),
        "midi_files": len(files),
        "songs": [],
        "total_patches": 0,
        "cond": "bpm_unit",
    }

    for midi_path in files:
        song_id = song_id_from_path(midi_path, raw_dir)
        music = muspy.read_midi(midi_path)
        bpm = midi_tempo_bpm(music)
        patches = midi_to_patches(midi_path)
        saved = save_pair_patches(
            patches,
            input_root / song_id,
            target_root / song_id,
            mode=mode,
            min_onsets=min_onsets,
            bpm=bpm,
            first_patch_only=first_patch_only,
        )
        stats["songs"].append(
            {
                "song_id": song_id,
                "midi": str(midi_path),
                "patches": saved,
                "bpm": bpm,
            }
        )
        stats["total_patches"] += saved
        print(f"{song_id}: {saved} patches (bpm={bpm:.1f})")

    manifest_path = pairs_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n合計: {stats['total_patches']} パッチ ({stats['midi_files']} MIDI)")
    print(f"保存先: {pairs_dir}")
    print(f"manifest: {manifest_path}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="raw MIDI から U-Net 学習用 input/target パッチペアを生成",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "raw" / "guitar-techs",
        help="生 MIDI のルート（サブフォルダごとに曲 ID 化）",
    )
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "pairs" / "guitar-techs",
        help="output/input と output/target を作る先",
    )
    parser.add_argument(
        "--mode",
        choices=PAIR_MODES,
        default="onset_to_full",
        help="identity / onset_to_full / downbeat_chord / root_per_bar / melody_line",
    )
    parser.add_argument(
        "--min-onsets",
        type=int,
        default=1,
        help="パッチを残す最小オンセット数",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="例: P3_music P1_techniques（指定時は該当フォルダのみ）",
    )
    parser.add_argument(
        "--first-patch-only",
        action="store_true",
        help="各 MIDI から bar_index=0 の最初のパッチのみ保存（patches:2 重複回避）",
    )
    args = parser.parse_args()

    prepare_pairs(
        args.raw_dir,
        args.pairs_dir,
        mode=args.mode,
        min_onsets=args.min_onsets,
        categories=args.categories,
        first_patch_only=args.first_patch_only,
    )


if __name__ == "__main__":
    main()
