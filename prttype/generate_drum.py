"""ドラム MIDI を生成する。

流れ: 進行骨格 → drum U-Net（1ch）→ Drums トラック。
extract_kick_times は generate_form から bass 協調用に公開する。
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import muspy

from checkpoint_paths import resolve_part_checkpoint
from inference import MIDI_DIR, load_model, predict_drum_patches
from makeData.constants import DRUM_KICK
from makeData.drum import extract_kick_times  # re-export
from midi_to_patch import midi_to_patches
from patch_to_midi import patches_to_music, save_music
from progression_input import build_backing_skeleton_music, get_progression
from sample_structure_params import DEFAULT_PRIOR_CKPT, resolve_structure_params

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = resolve_part_checkpoint("drum")
DEFAULT_PRIOR_CHECKPOINT = DEFAULT_PRIOR_CKPT

__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_PRIOR_CHECKPOINT",
    "extract_kick_times",
    "generate_drum_music",
    "main",
]


def _apply_tempo(music: muspy.Music, bpm: float) -> muspy.Music:
    music.tempos = [muspy.Tempo(time=0, qpm=float(bpm))]
    return music


def _name_drum_track(music: muspy.Music) -> muspy.Music:
    for track in music.tracks:
        if track.is_drum:
            track.name = "Drums"
            track.program = 0
    return music


def generate_drum_music(
    *,
    progression: str,
    key: str,
    bars: int = 8,
    bpm: float = 120.0,
    bars_per_chord: int = 1,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    identity: bool = False,
    onset_th: float = 0.35,
    beat_type: str = "eight_basic",
    model: Any | None = None,
    device: Any | None = None,
) -> muspy.Music:
    """進行骨格からドラムを生成し、muspy.Music を返す。"""
    ckpt = Path(checkpoint)
    if not identity and not ckpt.is_file():
        raise FileNotFoundError(f"drum checkpoint がありません: {ckpt}")

    spec = get_progression(progression)
    bpc = max(1, int(bars_per_chord))
    skeleton = build_backing_skeleton_music(
        progression=spec,
        key=key,
        bars=bars,
        bpm=bpm,
        bars_per_chord=bpc,
        chords_per_bar=1,
        raise_odd_loop_last=False,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_midi = Path(tmp) / "skeleton.mid"
        muspy.write_midi(tmp_midi, skeleton)
        patches = midi_to_patches(tmp_midi)
        if not patches:
            raise RuntimeError(f"パッチが 0 件です。bars={bars} を 8 以上にしてください。")

        if identity:
            # 骨格にはドラムが無いので空の drum を返す
            output_patches = patches
        else:
            import torch

            if model is None or device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = load_model(ckpt, device)
            output_patches = predict_drum_patches(
                model,
                patches,
                device,
                input_mode="downbeat_chord",
                onset_th=onset_th,
                bpm=float(bpm),
                beat_type=beat_type,
            )

    music = patches_to_music(output_patches)
    # tonal は空なのでドラムのみ残す
    drum_tracks = [t for t in music.tracks if t.is_drum]
    music.tracks = drum_tracks
    music = _name_drum_track(music)
    music = _apply_tempo(music, bpm)
    return music


def main() -> None:
    parser = argparse.ArgumentParser(description="ドラム MIDI を生成する")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--prior-checkpoint", type=Path, default=DEFAULT_PRIOR_CHECKPOINT)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--prior-argmax", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--onset-th", type=float, default=0.35)
    parser.add_argument(
        "--beat-type",
        type=str,
        default="eight_basic",
        help="ドラム型（BEAT_TYPES のいずれか）",
    )
    args = parser.parse_args()

    params = resolve_structure_params(
        text=args.text,
        prior_checkpoint=args.prior_checkpoint,
        seed=args.seed,
        sample=not args.prior_argmax,
        temperature=args.prior_temperature,
    )
    print(
        f"[structure:{params.source}] progression={params.progression} "
        f"key={params.key} bpm={params.bpm} energy={params.energy} "
        f"beat_type={args.beat_type}"
    )

    music = generate_drum_music(
        progression=params.progression,
        key=params.key,
        bars=params.bars,
        bpm=params.bpm,
        bars_per_chord=params.bars_per_chord,
        checkpoint=args.checkpoint,
        onset_th=args.onset_th,
        beat_type=args.beat_type,
    )
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or (
        MIDI_DIR / f"drum_{params.progression}_{params.key}_bpm{int(params.bpm):03d}.mid"
    )
    save_music(music, out)
    kicks = sum(
        1
        for t in music.tracks
        if t.is_drum
        for n in t.notes
        if int(n.pitch) == DRUM_KICK
    )
    print(
        f"ドラム: {out}  notes={sum(len(t.notes) for t in music.tracks)} "
        f"kicks={kicks}"
    )


if __name__ == "__main__":
    main()
