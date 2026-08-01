"""構成レイヤ入口: 雰囲気文 → 曲の色 → インスト構成 → バッキング+リード連結 → 1本の MIDI。

区間ごとに8小節バッキング／リードをメモリ上で生成し、小節オフセットを加算して連結する。
進行・キー・BPM はユーザー指定しない（prior / カタログ）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import muspy

from generate_backing import DEFAULT_CHECKPOINT, DEFAULT_PRIOR_CHECKPOINT, generate_backing_music
from generate_lead import DEFAULT_CHECKPOINT as DEFAULT_LEAD_CHECKPOINT
from generate_lead import generate_lead_music
from inference import MIDI_DIR, load_model
from makeData.constants import TICKS_PER_BAR
from midi_music_utils import power_chord_attack_times
from patch_to_midi import save_music
from program_utils import GUITAR_OVERDRIVE_PROGRAM
from sample_structure_params import resolve_structure_params
from song_form import SongForm, sample_song_form

SCRIPT_DIR = Path(__file__).resolve().parent


def _shift_and_merge(
    base: muspy.Music | None,
    section: muspy.Music,
    *,
    tick_offset: int,
) -> muspy.Music:
    """section のノート時刻をずらして base に追記する。"""
    shifted_tracks: list[muspy.Track] = []
    for track in section.tracks:
        new_track = muspy.Track(
            program=track.program,
            is_drum=track.is_drum,
            name=track.name or "Backing",
        )
        for note in track.notes:
            new_track.append(
                muspy.Note(
                    time=int(note.time) + tick_offset,
                    duration=int(note.duration),
                    pitch=int(note.pitch),
                    velocity=int(note.velocity),
                )
            )
        shifted_tracks.append(new_track)

    if base is None:
        return muspy.Music(
            resolution=section.resolution,
            tempos=list(section.tempos) if section.tempos else [],
            tracks=shifted_tracks,
        )

    # program+name+is_drum でマージ（同 program の Backing/Lead を分離）
    by_key: dict[tuple[int, str, bool], muspy.Track] = {}
    for track in base.tracks:
        key = (int(track.program), track.name or "", bool(track.is_drum))
        by_key[key] = track
    for track in shifted_tracks:
        key = (int(track.program), track.name or "", bool(track.is_drum))
        if key not in by_key:
            by_key[key] = muspy.Track(
                program=track.program,
                is_drum=track.is_drum,
                name=track.name,
            )
            base.tracks.append(by_key[key])
        by_key[key].extend(track.notes)
    return base


def generate_form(
    *,
    progression: str,
    key: str,
    bpm: float,
    bars_per_chord: int = 1,
    form: SongForm | None = None,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    lead_checkpoint: Path = DEFAULT_LEAD_CHECKPOINT,
    output: Path | None = None,
    guitar_program: int = GUITAR_OVERDRIVE_PROGRAM,
    seed: int | None = None,
    temperature: float = 1.0,
    chord_peak_decode: bool = True,
    with_lead: bool = True,
) -> Path:
    """構成テンプレに沿ってバッキング（+リード）を連結し、1本の MIDI を保存する。"""
    form = form or sample_song_form(seed=seed)
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"form_{form.template_id}_{progression}_{key}_bpm{int(bpm):03d}_{form.total_bars}bars"
    output_path = output or (MIDI_DIR / f"{stem}.mid")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backing_model = load_model(checkpoint, device)
    lead_model = load_model(lead_checkpoint, device) if with_lead else None

    music: muspy.Music | None = None
    bar_offset = 0
    section_notes: list[tuple[str, int, int]] = []

    for i, section in enumerate(form.sections):
        section_seed = None if seed is None else int(seed) + i
        decode = section.decode_params
        part = generate_backing_music(
            progression=progression,
            key=key,
            bars=section.bars,
            bpm=bpm,
            bars_per_chord=bars_per_chord,
            checkpoint=checkpoint,
            guitar_program=guitar_program,
            seed=section_seed,
            temperature=temperature,
            chord_peak_decode=chord_peak_decode,
            onset_th=float(decode["onset_th"]),
            peak_min_distance=int(decode["peak_min_distance"]),
            release_gap_ticks=int(decode["release_gap_ticks"]),
            model=backing_model,
            device=device,
        )
        n_backing = sum(len(t.notes) for t in part.tracks)
        tick_offset = bar_offset * TICKS_PER_BAR
        music = _shift_and_merge(music, part, tick_offset=tick_offset)

        n_lead = 0
        if with_lead and lead_model is not None:
            blocked = power_chord_attack_times(part)
            lead_part = generate_lead_music(
                progression=progression,
                key=key,
                bars=section.bars,
                bpm=bpm,
                bars_per_chord=bars_per_chord,
                checkpoint=lead_checkpoint,
                guitar_program=guitar_program,
                onset_th=float(decode["lead_onset_th"]),
                blocked_power_onsets=blocked,
                model=lead_model,
                device=device,
            )
            n_lead = sum(len(t.notes) for t in lead_part.tracks)
            music = _shift_and_merge(music, lead_part, tick_offset=tick_offset)

        section_notes.append((section.label, n_backing, n_lead))
        bar_offset += section.bars

    if music is None:
        raise RuntimeError("セクションが空です")

    music.tempos = [muspy.Tempo(time=0, qpm=float(bpm))]
    save_music(music, output_path)

    total = sum(len(t.notes) for t in music.tracks)
    print(f"[form] {form.describe()}")
    for label, n_b, n_l in section_notes:
        if with_lead:
            print(f"  {label}: backing={n_b} lead={n_l} notes")
        else:
            print(f"  {label}: {n_b} notes")
    print(f"進行: {progression} / キー: {key} / BPM: {bpm}")
    print(f"トラック: {[t.name for t in music.tracks]}")
    print(f"総小節: {form.total_bars} / 総ノート: {total}")
    print(f"出力: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "構成レイヤ付きバッキング(+リード)を1本の MIDI に生成する。"
            "進行・キー・BPM は文→prior（無ければカタログ乱択）。"
        ),
    )
    parser.add_argument("--text", type=str, default=None, help="雰囲気の文")
    parser.add_argument("--prior-checkpoint", type=Path, default=DEFAULT_PRIOR_CHECKPOINT)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--prior-argmax", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="prior / テンプレ / セクション派生の基点")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lead-ckpt", type=Path, default=DEFAULT_LEAD_CHECKPOINT)
    parser.add_argument("--no-lead", action="store_true", help="バッキングのみ")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--guitar-program", type=int, default=GUITAR_OVERDRIVE_PROGRAM)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--no-chord-peak-decode",
        action="store_true",
        help="コード共通ピークデコードをOFF",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="テンプレ一覧を表示して終了",
    )
    args = parser.parse_args()

    if args.list_templates:
        from song_form import list_templates

        for tid, bars, path in list_templates():
            print(f"{tid:10s} {bars:3d}bars  {path}")
        return

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

    # テンプレ抽選は seed、演奏は seed+index（同一役割の複製を避ける）
    form = sample_song_form(seed=args.seed)
    generate_form(
        progression=params.progression,
        key=params.key,
        bpm=params.bpm,
        bars_per_chord=params.bars_per_chord,
        form=form,
        checkpoint=args.checkpoint,
        lead_checkpoint=args.lead_ckpt,
        output=args.output,
        guitar_program=args.guitar_program,
        seed=args.seed,
        temperature=args.temperature,
        chord_peak_decode=not args.no_chord_peak_decode,
        with_lead=not args.no_lead,
    )


if __name__ == "__main__":
    main()
