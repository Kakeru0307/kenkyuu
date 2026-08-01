"""フル MIDI から骨格 MIDI を書き出す（確認用）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from midi_to_patch import MidiPatch, midi_to_patches
from patch_to_midi import patches_to_music, save_music
from program_utils import GUITAR_OVERDRIVE_PROGRAM, remap_tonal_program
from skeleton import PAIR_MODES, make_input_tonal

SCRIPT_DIR = Path(__file__).resolve().parent


def export_skeleton_midi(
    midi_path: Path,
    output_path: Path,
    *,
    mode: str,
) -> None:
    patches = midi_to_patches(midi_path)
    skeleton_patches = []
    for patch in patches:
        skeletal_chw = make_input_tonal(patch.tonal_chw, mode)
        skeleton_patches.append(
            MidiPatch(
                tonal=skeletal_chw.transpose(1, 2, 0),
                drum=patch.drum,
                bar_index=patch.bar_index,
            )
        )

    music = patches_to_music(skeleton_patches)
    music = remap_tonal_program(music, GUITAR_OVERDRIVE_PROGRAM)
    save_music(music, output_path)

    note_count = sum(len(track.notes) for track in music.tracks)
    print(f"骨格 MIDI を保存しました: {output_path}")
    print(f"モード: {mode}, ノート数: {note_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="フル MIDI から骨格 MIDI を書き出す")
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=PAIR_MODES, default="downbeat_chord")
    args = parser.parse_args()

    export_skeleton_midi(args.midi, args.output, mode=args.mode)


if __name__ == "__main__":
    main()
