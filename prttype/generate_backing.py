"""バッキングギター MIDI を生成する。【現行マイルストーン入口】

流れ:
  構造パラメータを学習データ分布からサンプル（進行・キー・BPMはユーザー指定しない）
    → 骨格 MIDI（小節頭コードトーン）
    → U-Net（骨格 + BPM 条件 ch）
    → バッキング MIDI（1 トラック）

最終形（文のみ→端到端）ではない。関数引数の progression/key/bpm は内部用。
CLI では明示指定しない。
推論にルール後処理は置かない（モデル出力のまま保存）。
スケルトン MIDI のディスク保存はしない（モデル入力用の一時ファイルのみ）。
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import muspy

from checkpoint_paths import resolve_part_checkpoint, resolve_structure_prior_checkpoint
from inference import (
    MIDI_DIR,
    export_guitar_music,
    load_model,
    predict_patches,
)
from midi_to_patch import midi_to_patches
from patch_to_midi import patches_to_music, save_music
from progression_input import (
    build_backing_skeleton_music,
    get_progression,
)
from program_utils import (
    GUITAR_OVERDRIVE_PROGRAM,
    remap_tonal_program,
)
from makeData.progressions import resolve_progression_chords
from sample_structure_params import resolve_structure_params

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = resolve_part_checkpoint("backing")
DEFAULT_PRIOR_CHECKPOINT = resolve_structure_prior_checkpoint()


def _apply_tempo(music: muspy.Music, bpm: float) -> muspy.Music:
    music.tempos = [muspy.Tempo(time=0, qpm=float(bpm))]
    return music


def _name_backing_track(music: muspy.Music, program: int) -> muspy.Music:
    for track in music.tracks:
        if not track.is_drum:
            track.name = "Backing"
            track.program = program
    return music


def generate_backing_music(
    *,
    progression: str,
    key: str,
    bars: int = 8,
    bpm: float = 120.0,
    bars_per_chord: int = 1,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    guitar_program: int = GUITAR_OVERDRIVE_PROGRAM,
    identity: bool = False,
    seed: int | None = None,
    temperature: float = 1.0,
    chord_peak_decode: bool = True,
    onset_th: float = 0.3,
    peak_min_distance: int = 2,
    release_gap_ticks: int = 1,
    model: Any | None = None,
    device: Any | None = None,
) -> muspy.Music:
    """進行からバッキングを生成し、muspy.Music を返す（ファイル保存なし）。"""
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
            raise RuntimeError(
                f"パッチが 0 件です。bars={bars} を 8 以上にしてください。"
            )

        if identity:
            output_patches = patches
        else:
            import torch

            if model is None or device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = load_model(checkpoint, device)
            output_patches = predict_patches(
                model,
                patches,
                device,
                input_mode="downbeat_chord",
                guitar_only=True,
                seed=seed,
                temperature=temperature,
                bpm=float(bpm),
                chord_peak_decode=chord_peak_decode,
                onset_th=onset_th,
                peak_min_distance=peak_min_distance,
                release_gap_ticks=release_gap_ticks,
            )

    music = patches_to_music(output_patches)
    music = export_guitar_music(
        remap_tonal_program(music, guitar_program),
        program=guitar_program,
        include_drums=False,
    )
    music = _name_backing_track(music, guitar_program)
    music = _apply_tempo(music, bpm)
    return music


def generate_backing(
    *,
    progression: str,
    key: str,
    bars: int = 8,
    bpm: float = 120.0,
    bars_per_chord: int = 1,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output: Path | None = None,
    guitar_program: int = GUITAR_OVERDRIVE_PROGRAM,
    identity: bool = False,
    seed: int | None = None,
    temperature: float = 1.0,
    chord_peak_decode: bool = True,
    onset_th: float = 0.3,
    peak_min_distance: int = 2,
    release_gap_ticks: int = 1,
    model: Any | None = None,
    device: Any | None = None,
) -> Path:
    """進行からバッキング MIDI を生成して保存する。"""
    spec = get_progression(progression)
    chords = resolve_progression_chords(spec, key)
    chord_label = "-".join(
        f"{root}{'' if quality == 'maj' else quality}" for root, quality in chords
    )

    stem = f"backing_{progression}_{key}_bpm{int(bpm):03d}_{bars}bars"
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output or (MIDI_DIR / f"{stem}.mid")

    music = generate_backing_music(
        progression=progression,
        key=key,
        bars=bars,
        bpm=bpm,
        bars_per_chord=bars_per_chord,
        checkpoint=checkpoint,
        guitar_program=guitar_program,
        identity=identity,
        seed=seed,
        temperature=temperature,
        chord_peak_decode=chord_peak_decode,
        onset_th=onset_th,
        peak_min_distance=peak_min_distance,
        release_gap_ticks=release_gap_ticks,
        model=model,
        device=device,
    )
    save_music(music, output_path)

    note_count = sum(len(t.notes) for t in music.tracks)
    print(f"進行: {progression} ({spec.family}, mode={spec.mode})")
    print(f"キー: {key} / 例: {chord_label}")
    print(f"BPM: {bpm}, bars: {bars}, bars_per_chord: {max(1, int(bars_per_chord))}")
    print(f"バッキング: {output_path}")
    print(f"ノート数: {note_count}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "バッキングギター MIDI を生成する。"
            "進行・キー・BPM はユーザー指定せず、文→WRIME→prior（無ければカタログ乱択）。"
        ),
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="雰囲気の文（WRIME→structure prior）。省略時はカタログ乱択",
    )
    parser.add_argument(
        "--prior-checkpoint",
        type=Path,
        default=DEFAULT_PRIOR_CHECKPOINT,
    )
    parser.add_argument(
        "--prior-temperature",
        type=float,
        default=1.0,
        help="prior サンプリング温度（高いほど多様）",
    )
    parser.add_argument(
        "--prior-argmax",
        action="store_true",
        help="prior をサンプルせず argmax（決定的）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="prior / カタログ /（CVAE時）の乱数シード。省略時は毎回違う prior サンプル",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--guitar-program",
        type=int,
        default=GUITAR_OVERDRIVE_PROGRAM,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="CVAE のサンプリング温度（CVAE checkpoint のみ）",
    )
    parser.add_argument(
        "--no-chord-peak-decode",
        action="store_true",
        help="コード共通ピークデコードをOFF（従来の閾値+2tick分割）",
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
        f"mode={params.mode} bars_per_chord={params.bars_per_chord}"
    )
    generate_backing(
        progression=params.progression,
        key=params.key,
        bars=params.bars,
        bpm=params.bpm,
        bars_per_chord=params.bars_per_chord,
        checkpoint=args.checkpoint,
        output=args.output,
        guitar_program=args.guitar_program,
        seed=args.seed,
        temperature=args.temperature,
        chord_peak_decode=not args.no_chord_peak_decode,
    )


if __name__ == "__main__":
    main()
