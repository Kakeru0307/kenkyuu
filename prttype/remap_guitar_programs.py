"""raw ギター MIDI の program を category 3（ギター層）に揃える。"""

from __future__ import annotations

import argparse
from pathlib import Path

import muspy

from program_utils import GUITAR_PROGRAM, remap_tonal_program

SCRIPT_DIR = Path(__file__).resolve().parent


def remap_directory(raw_dir: Path, *, program: int = GUITAR_PROGRAM) -> int:
    midi_files = sorted(raw_dir.rglob("*.mid"))
    if not midi_files:
        raise FileNotFoundError(f"MIDI が見つかりません: {raw_dir}")

    updated = 0
    for midi_path in midi_files:
        music = muspy.read_midi(midi_path)
        before = [t.program for t in music.tracks if not t.is_drum]
        remap_tonal_program(music, program=program)
        after = [t.program for t in music.tracks if not t.is_drum]
        if before != after:
            updated += 1
        muspy.write_midi(midi_path, music)

    print(f"処理: {len(midi_files)} ファイル（program 変更: {updated}）")
    print(f"program={program} (category {program // 8})")
    return len(midi_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guitar-TECHS 等の tonal トラック program をギター番号に統一",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "raw" / "guitar-techs",
    )
    parser.add_argument(
        "--program",
        type=int,
        default=GUITAR_PROGRAM,
        help=f"デフォルト {GUITAR_PROGRAM} (Electric Guitar clean)",
    )
    args = parser.parse_args()
    remap_directory(args.raw_dir, program=args.program)


if __name__ == "__main__":
    main()
