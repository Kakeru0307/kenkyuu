"""構成レイヤ入口: 雰囲気文 → 曲の色 → インスト構成 → 4パート連結 → 1本の MIDI。

区間ごとに進行・beat_type・bars を変え、バッキング／リード／ドラム／ベースを連結する。
キー・BPM は曲通し固定。bass/drum は常時生成。
生成後に form_manifest.json を書き、form_gate_server で評価できる。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import muspy

from generate_backing import DEFAULT_CHECKPOINT, DEFAULT_PRIOR_CHECKPOINT, generate_backing_music
from generate_bass import DEFAULT_CHECKPOINT as DEFAULT_BASS_CHECKPOINT
from generate_bass import generate_bass_music
from generate_drum import DEFAULT_CHECKPOINT as DEFAULT_DRUM_CHECKPOINT
from generate_drum import extract_kick_times, generate_drum_music
from generate_lead import DEFAULT_CHECKPOINT as DEFAULT_LEAD_CHECKPOINT
from generate_lead import generate_lead_music
from inference import MIDI_DIR, load_model
from makeData.constants import TICKS_PER_BAR
from makeData.progressions import PROGRESSION_BY_NAME
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


def _resolve_home_mode(progression: str, mode: str | None) -> str:
    if mode:
        return mode
    spec = PROGRESSION_BY_NAME.get(progression)
    if spec is None:
        raise ValueError(f"unknown progression: {progression!r}")
    return spec.mode


def _write_form_manifest(
    path: Path,
    *,
    midi_name: str,
    text: str | None,
    home_progression: str,
    home_mode: str,
    key: str,
    bpm: float,
    va: tuple[float, float],
    form: SongForm,
) -> None:
    payload: dict[str, Any] = {
        "id": path.stem,
        "midi": midi_name,
        "text": text,
        "home_progression": home_progression,
        "home_mode": home_mode,
        "key": key,
        "bpm": float(bpm),
        "va": [float(va[0]), float(va[1])],
        "template_id": form.template_id,
        "sections": [
            {
                "role": s.role,
                "label": s.label,
                "progression": s.progression,
                "bars": s.bars,
                "beat_type": s.beat_type,
                "energy": s.energy,
            }
            for s in form.sections
        ],
        "gate": {"status": "pending", "score_fit": None, "note": ""},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_form(
    *,
    progression: str,
    key: str,
    bpm: float,
    bars_per_chord: int = 1,
    form: SongForm | None = None,
    home_mode: str | None = None,
    text: str | None = None,
    va: tuple[float, float] = (0.0, 0.0),
    checkpoint: Path = DEFAULT_CHECKPOINT,
    lead_checkpoint: Path = DEFAULT_LEAD_CHECKPOINT,
    bass_checkpoint: Path = DEFAULT_BASS_CHECKPOINT,
    drum_checkpoint: Path = DEFAULT_DRUM_CHECKPOINT,
    output: Path | None = None,
    guitar_program: int = GUITAR_OVERDRIVE_PROGRAM,
    seed: int | None = None,
    temperature: float = 1.0,
    chord_peak_decode: bool = True,
    with_lead: bool = True,
) -> Path:
    """構成テンプレに沿って 4 パートを連結し、1本の MIDI と form_manifest を保存する。"""
    bass_ckpt = Path(bass_checkpoint)
    drum_ckpt = Path(drum_checkpoint)
    if not bass_ckpt.is_file():
        raise FileNotFoundError(f"bass checkpoint がありません: {bass_ckpt}")
    if not drum_ckpt.is_file():
        raise FileNotFoundError(f"drum checkpoint がありません: {drum_ckpt}")

    resolved_mode = _resolve_home_mode(progression, home_mode)
    form = form or sample_song_form(
        seed=seed,
        home_progression=progression,
        home_mode=resolved_mode,
    )
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    stem = (
        f"form_{form.template_id}_{progression}_{key}_"
        f"bpm{int(bpm):03d}_{form.total_bars}bars"
    )
    output_path = output or (MIDI_DIR / f"{stem}.mid")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backing_model = load_model(checkpoint, device)
    lead_model = load_model(lead_checkpoint, device) if with_lead else None
    drum_model = load_model(drum_ckpt, device)
    bass_model = load_model(bass_ckpt, device)

    music: muspy.Music | None = None
    bar_offset = 0
    section_notes: list[tuple[str, str, str, int, int, int, int, int]] = []

    for i, section in enumerate(form.sections):
        section_seed = None if seed is None else int(seed) + i
        decode = section.decode_params
        tick_offset = bar_offset * TICKS_PER_BAR
        prog = section.progression

        part = generate_backing_music(
            progression=prog,
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
        music = _shift_and_merge(music, part, tick_offset=tick_offset)

        n_lead = 0
        if with_lead and lead_model is not None:
            blocked = power_chord_attack_times(part)
            lead_part = generate_lead_music(
                progression=prog,
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

        drum_part = generate_drum_music(
            progression=prog,
            key=key,
            bars=section.bars,
            bpm=bpm,
            bars_per_chord=bars_per_chord,
            checkpoint=drum_ckpt,
            onset_th=float(decode["drum_onset_th"]),
            beat_type=section.beat_type,
            model=drum_model,
            device=device,
        )
        n_drum = sum(len(t.notes) for t in drum_part.tracks)
        music = _shift_and_merge(music, drum_part, tick_offset=tick_offset)

        kick_times = extract_kick_times(drum_part)
        bass_part = generate_bass_music(
            progression=prog,
            key=key,
            bars=section.bars,
            bpm=bpm,
            bars_per_chord=bars_per_chord,
            checkpoint=bass_ckpt,
            onset_th=float(decode["bass_onset_th"]),
            kick_times=kick_times,
            model=bass_model,
            device=device,
        )
        n_bass = sum(len(t.notes) for t in bass_part.tracks)
        music = _shift_and_merge(music, bass_part, tick_offset=tick_offset)

        section_notes.append(
            (
                section.label,
                section.progression,
                section.beat_type,
                section.bars,
                n_backing,
                n_lead,
                n_drum,
                n_bass,
            )
        )
        bar_offset += section.bars

    if music is None:
        raise RuntimeError("セクションが空です")

    music.tempos = [muspy.Tempo(time=0, qpm=float(bpm))]
    save_music(music, output_path)

    manifest_path = output_path.with_suffix(".json")
    _write_form_manifest(
        manifest_path,
        midi_name=output_path.name,
        text=text,
        home_progression=progression,
        home_mode=resolved_mode,
        key=key,
        bpm=bpm,
        va=va,
        form=form,
    )

    total = sum(len(t.notes) for t in music.tracks)
    print(f"[form] {form.describe()}")
    for label, prog, beat, bars, n_b, n_l, n_d, n_ba in section_notes:
        print(
            f"  {label}[{prog}|{beat}|{bars}]: "
            f"backing={n_b} lead={n_l} drum={n_d} bass={n_ba} notes"
        )
    print(f"home進行: {progression} / キー: {key} / BPM: {bpm}")
    print(f"トラック: {[t.name for t in music.tracks]}")
    print(f"総小節: {form.total_bars} / 総ノート: {total}")
    print(f"出力: {output_path}")
    print(f"manifest: {manifest_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "構成レイヤ付き 4 パート（backing/lead/drum/bass）を1本の MIDI に生成する。"
            "進行・キー・BPM は文→prior（無ければカタログ乱択）。"
            "区間ごとに進行対比（同 mode）を適用する。"
        ),
    )
    parser.add_argument("--text", type=str, default=None, help="雰囲気の文")
    parser.add_argument("--prior-checkpoint", type=Path, default=DEFAULT_PRIOR_CHECKPOINT)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--prior-argmax", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lead-ckpt", type=Path, default=DEFAULT_LEAD_CHECKPOINT)
    parser.add_argument("--bass-ckpt", type=Path, default=DEFAULT_BASS_CHECKPOINT)
    parser.add_argument("--drum-ckpt", type=Path, default=DEFAULT_DRUM_CHECKPOINT)
    parser.add_argument("--no-lead", action="store_true", help="リードなし（他3パートは常時）")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--guitar-program", type=int, default=GUITAR_OVERDRIVE_PROGRAM)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-chord-peak-decode", action="store_true")
    parser.add_argument("--list-templates", action="store_true")
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
    home_mode = _resolve_home_mode(params.progression, params.mode or None)
    print(
        f"[structure:{params.source}] progression={params.progression} "
        f"key={params.key} bpm={params.bpm} energy={params.energy} "
        f"mode={home_mode} va=({params.va[0]:+.2f},{params.va[1]:+.2f}) "
        f"bars_per_chord={params.bars_per_chord}"
    )

    form = sample_song_form(
        seed=args.seed,
        home_progression=params.progression,
        home_mode=home_mode,
        va=params.va,
    )
    generate_form(
        progression=params.progression,
        key=params.key,
        bpm=params.bpm,
        bars_per_chord=params.bars_per_chord,
        form=form,
        home_mode=home_mode,
        text=args.text,
        va=params.va,
        checkpoint=args.checkpoint,
        lead_checkpoint=args.lead_ckpt,
        bass_checkpoint=args.bass_ckpt,
        drum_checkpoint=args.drum_ckpt,
        output=args.output,
        guitar_program=args.guitar_program,
        seed=args.seed,
        temperature=args.temperature,
        chord_peak_decode=not args.no_chord_peak_decode,
        with_lead=not args.no_lead,
    )


if __name__ == "__main__":
    main()
