"""backing + lead を生成し、1 つの MIDI に 2 トラックで統合する。

構造パラメータ（進行・キー・BPM）はユーザー指定せず、文→WRIME→prior
（無ければカタログ乱択）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import muspy

from generate_backing import generate_backing
from generate_lead import generate_lead
from inference import MIDI_DIR
from midi_music_utils import power_chord_attack_times
from patch_to_midi import save_music
from program_utils import (
    GUITAR_OVERDRIVE_PROGRAM,
)
from sample_structure_params import DEFAULT_PRIOR_CKPT, resolve_structure_params

SCRIPT_DIR = Path(__file__).resolve().parent
BACKING_CKPT = SCRIPT_DIR / "checkpoints" / "backing" / "unet_last.pt"
LEAD_CKPT = SCRIPT_DIR / "checkpoints" / "lead" / "unet_last.pt"
DEFAULT_PRIOR_CHECKPOINT = DEFAULT_PRIOR_CKPT


def _load_tracks(midi_path: Path, name: str) -> list[muspy.Track]:
    music = muspy.read_midi(midi_path)
    tracks = [t for t in music.tracks if not t.is_drum and t.notes]
    for t in tracks:
        t.name = name
    return tracks


def generate_song(
    *,
    progression: str,
    key: str,
    bars: int = 8,
    bpm: float = 120.0,
    bars_per_chord: int = 1,
    backing_ckpt: Path = BACKING_CKPT,
    lead_ckpt: Path = LEAD_CKPT,
    lead_onset_th: float = 0.3,
    output: Path | None = None,
    with_lead: bool = True,
    with_backing: bool = True,
    eval_muspy: bool = False,
    mode: str | None = None,
) -> Path:
    """backing + lead を生成して 2 トラック MIDI に統合する。"""
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"song_{progression}_{key}_bpm{int(bpm):03d}_{bars}bars"
    output_path = output or (MIDI_DIR / f"{stem}.mid")

    tracks: list[muspy.Track] = []
    blocked_power_onsets: set[int] = set()

    if with_backing:
        backing_path = generate_backing(
            progression=progression,
            key=key,
            bars=bars,
            bpm=bpm,
            bars_per_chord=bars_per_chord,
            checkpoint=backing_ckpt,
            guitar_program=GUITAR_OVERDRIVE_PROGRAM,
        )
        tracks += _load_tracks(backing_path, "Backing")
        blocked_power_onsets = power_chord_attack_times(muspy.read_midi(backing_path))

    if with_lead:
        lead_path = generate_lead(
            progression=progression,
            key=key,
            bars=bars,
            bpm=bpm,
            bars_per_chord=bars_per_chord,
            checkpoint=lead_ckpt,
            save_skeleton=False,
            guitar_program=GUITAR_OVERDRIVE_PROGRAM,
            onset_th=lead_onset_th,
            blocked_power_onsets=blocked_power_onsets,
        )
        tracks += _load_tracks(lead_path, "Lead")

    if not tracks:
        raise RuntimeError("トラックが空です（with_backing / with_lead を確認）")

    song = muspy.Music(
        resolution=4,
        tempos=[muspy.Tempo(time=0, qpm=float(bpm))],
        tracks=tracks,
    )
    save_music(song, output_path)

    total = sum(len(t.notes) for t in tracks)
    print("\n=== 統合 ===")
    print(f"進行: {progression} / キー: {key} / BPM: {bpm} / {bars}小節")
    print(f"トラック: {[t.name for t in tracks]}")
    print(f"統合 MIDI: {output_path}（総ノート数 {total}）")

    if eval_muspy:
        try:
            from evaluate_midi import evaluate_midi, format_report

            print(format_report(evaluate_midi(output_path, key=key, mode=mode)))
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluate_midi] skipped: {exc}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "backing + lead を生成し 1 つの MIDI に統合する。"
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
        help="prior / カタログの乱数シード。省略時は毎回違う prior サンプル",
    )
    parser.add_argument("--backing-ckpt", type=Path, default=BACKING_CKPT)
    parser.add_argument("--lead-ckpt", type=Path, default=LEAD_CKPT)
    parser.add_argument("--lead-onset-th", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-lead", action="store_true", help="バッキングのみ")
    parser.add_argument("--no-backing", action="store_true", help="リードのみ")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="生成後に MusPy 客観評価を表示",
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
        f"mode={params.mode} emotion={params.emotion_target or '-'} "
        f"bars_per_chord={params.bars_per_chord}"
    )
    generate_song(
        progression=params.progression,
        key=params.key,
        bars=params.bars,
        bpm=params.bpm,
        bars_per_chord=params.bars_per_chord,
        backing_ckpt=args.backing_ckpt,
        lead_ckpt=args.lead_ckpt,
        lead_onset_th=args.lead_onset_th,
        output=args.output,
        with_lead=not args.no_lead,
        with_backing=not args.no_backing,
        eval_muspy=args.eval,
        mode=params.mode or None,
    )


if __name__ == "__main__":
    main()
