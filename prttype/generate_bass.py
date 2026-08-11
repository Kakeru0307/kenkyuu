"""ベース MIDI を生成する。

流れ: 進行骨格 → bass U-Net（cat4）→ Bass トラック。
kick_times がある場合、キック時刻付近にルート onset を補強する。
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import muspy

from checkpoint_paths import resolve_part_checkpoint
from inference import MIDI_DIR, load_model, predict_patches
from makeData.bass import root_in_bass_range
from makeData.constants import BASS_PROGRAM, TICKS_PER_BAR
from makeData.progressions import resolve_progression_chords
from midi_to_patch import midi_to_patches
from patch_to_midi import patches_to_music, save_music
from progression_input import build_backing_skeleton_music, get_progression
from program_utils import remap_tonal_program
from sample_structure_params import DEFAULT_PRIOR_CKPT, resolve_structure_params

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = resolve_part_checkpoint("bass")
DEFAULT_PRIOR_CHECKPOINT = DEFAULT_PRIOR_CKPT


def _apply_tempo(music: muspy.Music, bpm: float) -> muspy.Music:
    music.tempos = [muspy.Tempo(time=0, qpm=float(bpm))]
    return music


def _name_bass_track(music: muspy.Music, program: int) -> muspy.Music:
    for track in music.tracks:
        if not track.is_drum:
            track.name = "Bass"
            track.program = program
    return music


def _reinforce_kick_roots(
    music: muspy.Music,
    *,
    progression: str,
    key: str,
    bars: int,
    bars_per_chord: int,
    kick_times: set[int] | frozenset[int],
) -> muspy.Music:
    """キック時刻にベースルートが無い場合、短いルートを追加する。"""
    if not kick_times:
        return music
    spec = get_progression(progression)
    chords = resolve_progression_chords(spec, key)
    bpc = max(1, bars_per_chord)
    bass_track = None
    for track in music.tracks:
        if not track.is_drum:
            bass_track = track
            break
    if bass_track is None:
        bass_track = muspy.Track(program=BASS_PROGRAM, is_drum=False, name="Bass")
        music.tracks.append(bass_track)

    existing = {int(n.time) for n in bass_track.notes}
    for t in sorted(kick_times):
        if t in existing:
            continue
        bar = t // TICKS_PER_BAR
        if bar < 0 or bar >= bars:
            continue
        slot = (bar // bpc) % len(chords)
        root = root_in_bass_range(chords[slot][0])
        bass_track.append(
            muspy.Note(time=int(t), pitch=root, duration=2, velocity=90)
        )
    return music


def generate_bass_music(
    *,
    progression: str,
    key: str,
    bars: int = 8,
    bpm: float = 120.0,
    bars_per_chord: int = 1,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    bass_program: int = BASS_PROGRAM,
    identity: bool = False,
    onset_th: float = 0.3,
    kick_times: set[int] | frozenset[int] | None = None,
    model: Any | None = None,
    device: Any | None = None,
) -> muspy.Music:
    """進行からベースを生成し、muspy.Music を返す。"""
    ckpt = Path(checkpoint)
    if not identity and not ckpt.is_file():
        raise FileNotFoundError(f"bass checkpoint がありません: {ckpt}")

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
            output_patches = patches
        else:
            import torch

            if model is None or device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = load_model(ckpt, device)
            output_patches = predict_patches(
                model,
                patches,
                device,
                input_mode="downbeat_chord",
                guitar_only=False,
                bass_only=True,
                onset_th=onset_th,
                chord_peak_decode=False,
                bpm=float(bpm),
            )

    music = patches_to_music(output_patches)
    music = remap_tonal_program(music, bass_program)
    music = _name_bass_track(music, bass_program)
    if kick_times:
        music = _reinforce_kick_roots(
            music,
            progression=progression,
            key=key,
            bars=bars,
            bars_per_chord=bpc,
            kick_times=kick_times,
        )
    music = _apply_tempo(music, bpm)
    return music


def main() -> None:
    parser = argparse.ArgumentParser(description="ベース MIDI を生成する")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--prior-checkpoint", type=Path, default=DEFAULT_PRIOR_CHECKPOINT)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--prior-argmax", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bass-program", type=int, default=BASS_PROGRAM)
    parser.add_argument("--onset-th", type=float, default=0.3)
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
        f"mode={params.mode} bars_per_chord={params.bars_per_chord}"
    )

    music = generate_bass_music(
        progression=params.progression,
        key=params.key,
        bars=params.bars,
        bpm=params.bpm,
        bars_per_chord=params.bars_per_chord,
        checkpoint=args.checkpoint,
        bass_program=args.bass_program,
        onset_th=args.onset_th,
    )
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or (
        MIDI_DIR / f"bass_{params.progression}_{params.key}_bpm{int(params.bpm):03d}.mid"
    )
    save_music(music, out)
    print(f"ベース: {out}  notes={sum(len(t.notes) for t in music.tracks)}")


if __name__ == "__main__":
    main()
