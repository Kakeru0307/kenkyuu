"""リードギター MIDI（単音主体＋一部パワーコード）を生成する。

構造パラメータ（進行・キー・BPM）はユーザー指定せず、学習データ分布からサンプルする。
関数引数の progression/key/bpm は内部用。
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

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
DEFAULT_CHECKPOINT = resolve_part_checkpoint("lead")
DEFAULT_PRIOR_CHECKPOINT = resolve_structure_prior_checkpoint()


def _apply_tempo(music: muspy.Music, bpm: float) -> muspy.Music:
    music.tempos = [muspy.Tempo(time=0, qpm=float(bpm))]
    return music


def _shape_lead_voicings(
    music: muspy.Music,
    program: int,
    *,
    allow_power_chords: bool,
    blocked_power_onsets: set[int] | frozenset[int] | None = None,
) -> muspy.Music:
    """単音を基本に、完全5度の2音だけをpower chordとして残す。"""
    blocked = blocked_power_onsets or frozenset()
    for track in music.tracks:
        if track.is_drum:
            continue
        by_time: dict[int, list[muspy.Note]] = {}
        for note in track.notes:
            by_time.setdefault(int(note.time), []).append(note)
        selected: list[muspy.Note] = []
        for time, notes in by_time.items():
            notes = sorted(notes, key=lambda n: n.pitch)
            pair: tuple[muspy.Note, muspy.Note] | None = None
            if allow_power_chords and time not in blocked:
                by_pitch = {int(n.pitch): n for n in notes}
                for root in sorted(by_pitch):
                    if root + 7 in by_pitch:
                        pair = (by_pitch[root], by_pitch[root + 7])
                        break
            if pair is not None:
                selected.extend(pair)
            else:
                selected.append(notes[-1])
        track.notes = sorted(selected, key=lambda n: (n.time, n.pitch))
        track.name = "Lead"
        track.program = program
    return music


def generate_lead_music(
    *,
    progression: str,
    key: str,
    bars: int = 8,
    bpm: float = 120.0,
    bars_per_chord: int = 1,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    guitar_program: int = GUITAR_OVERDRIVE_PROGRAM,
    identity: bool = False,
    onset_th: float = 0.3,
    blocked_power_onsets: set[int] | frozenset[int] | None = None,
    model=None,
    device=None,
) -> muspy.Music:
    """進行からリードを生成し、muspy.Music を返す（ファイル保存なし）。"""
    spec = get_progression(progression)

    skeleton = build_backing_skeleton_music(
        progression=spec,
        key=key,
        bars=bars,
        bpm=bpm,
        bars_per_chord=bars_per_chord,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_midi = Path(tmp) / "skeleton.mid"
        muspy.write_midi(tmp_midi, skeleton)
        patches = midi_to_patches(tmp_midi)
        if not patches:
            raise RuntimeError(f"パッチが 0 件です。bars={bars} を 8 以上にしてください。")

        allow_power_chords = False
        if identity:
            output_patches = patches
        else:
            import torch

            if model is None or device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = load_model(checkpoint, device)
            try:
                in_channels = int(model.encoder.conv1.in_channels)  # type: ignore[attr-defined]
            except Exception:
                in_channels = 11
            allow_power_chords = in_channels >= 13
            output_patches = predict_patches(
                model,
                patches,
                device,
                input_mode="downbeat_chord",
                guitar_only=True,
                onset_th=onset_th,
                bpm=float(bpm),
                chord_peak_decode=not allow_power_chords,
                blocked_power_onsets=blocked_power_onsets,
            )

    music = patches_to_music(output_patches)
    music = export_guitar_music(
        remap_tonal_program(music, guitar_program),
        program=guitar_program,
        include_drums=False,
    )
    music = _shape_lead_voicings(
        music,
        guitar_program,
        allow_power_chords=allow_power_chords,
        blocked_power_onsets=blocked_power_onsets,
    )
    music = _apply_tempo(music, bpm)
    return music


def generate_lead(
    *,
    progression: str,
    key: str,
    bars: int = 8,
    bpm: float = 120.0,
    bars_per_chord: int = 1,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output: Path | None = None,
    save_skeleton: bool = False,
    guitar_program: int = GUITAR_OVERDRIVE_PROGRAM,
    identity: bool = False,
    onset_th: float = 0.3,
    blocked_power_onsets: set[int] | frozenset[int] | None = None,
) -> Path:
    """進行からリード MIDI を生成して保存する。"""
    spec = get_progression(progression)
    chords = resolve_progression_chords(spec, key)
    chord_label = "-".join(
        f"{root}{'' if quality == 'maj' else quality}" for root, quality in chords
    )

    stem = f"lead_{progression}_{key}_bpm{int(bpm):03d}_{bars}bars"
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output or (MIDI_DIR / f"{stem}.mid")
    skeleton_path = MIDI_DIR / f"{stem}_skeleton.mid"

    if save_skeleton:
        skeleton = build_backing_skeleton_music(
            progression=spec,
            key=key,
            bars=bars,
            bpm=bpm,
            bars_per_chord=bars_per_chord,
        )
        save_music(skeleton, skeleton_path)

    music = generate_lead_music(
        progression=progression,
        key=key,
        bars=bars,
        bpm=bpm,
        bars_per_chord=bars_per_chord,
        checkpoint=checkpoint,
        guitar_program=guitar_program,
        identity=identity,
        onset_th=onset_th,
        blocked_power_onsets=blocked_power_onsets,
    )
    save_music(music, output_path)

    note_count = sum(len(t.notes) for t in music.tracks)
    print(f"進行: {progression} ({spec.family}, mode={spec.mode})")
    print(f"キー: {key} / 例: {chord_label}")
    print(f"BPM: {bpm}, bars: {bars}, bars_per_chord: {bars_per_chord}")
    if save_skeleton:
        print(f"骨格: {skeleton_path}")
    print(f"リード: {output_path}")
    print(f"ノート数(単音主体・power chord許容): {note_count}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "リードギター MIDI（単音主体＋一部power chord）を生成する。"
            "進行・キー・BPM はユーザー指定せず、文→WRIME→prior（無ければカタログ乱択）。"
        ),
    )
    parser.add_argument("--text", type=str, default=None, help="雰囲気の文（WRIME→prior）")
    parser.add_argument("--prior-checkpoint", type=Path, default=DEFAULT_PRIOR_CHECKPOINT)
    parser.add_argument(
        "--prior-temperature",
        type=float,
        default=1.0,
        help="prior サンプリング温度",
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
        help="prior / カタログの乱数シード。省略時は毎回違う prior サンプル",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    # 旧CLI互換。スケルトンは常に保存しない。
    parser.add_argument("--no-skeleton", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--guitar-program", type=int, default=GUITAR_OVERDRIVE_PROGRAM)
    parser.add_argument("--onset-th", type=float, default=0.3, help="発音とみなすしきい値")
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
    generate_lead(
        progression=params.progression,
        key=params.key,
        bars=params.bars,
        bpm=params.bpm,
        bars_per_chord=params.bars_per_chord,
        checkpoint=args.checkpoint,
        output=args.output,
        save_skeleton=False,
        guitar_program=args.guitar_program,
        onset_th=args.onset_th,
    )


if __name__ == "__main__":
    main()
