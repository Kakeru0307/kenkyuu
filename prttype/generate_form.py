"""構成レイヤ入口: 雰囲気文 → 曲の色 → インスト構成 → 4パート連結 → 1本の MIDI。

区間ごとに進行・beat_type・奏法・bars・編成レシピを変え、ON のパートだけ生成して連結する。
リードの奏法は区間が決め、保存後にそのトラックへ pitch bend として書く。
キー・BPM は曲通し固定。編成 OFF のパートは生成スキップ。
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
from util.ensemble_lock import (
    build_drum_grid,
    collect_onsets,
    shift_lead_to_gaps,
    snap_music_to_grid,
)
from util.midi_music_utils import crop_music_to_bars, power_chord_attack_times

# パッチ化下限。編成 span が短いときはこれで生成してから crop する。
_GEN_MIN_BARS = 8
from util.patch_to_midi import save_music
from util.program_utils import GUITAR_OVERDRIVE_PROGRAM
from sample_structure_params import resolve_structure_params
from song_form import ArrSpan, SongForm, sample_song_form

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
                "stroke_groove": s.stroke_groove,
                "technique_type": s.technique_type,
                "energy": s.energy,
                "arrangement_id": s.arrangement_id,
            }
            for s in form.sections
        ],
        "gate": {"status": "pending", "score_fit": None, "note": ""},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_lead_note_techs(
    midi_path: Path,
    *,
    seed: int | None,
) -> None:
    """Lead にノート単位奏法付与し、曲全体のトラック velocity バランスを適用。"""
    from guitar_tech.apply_midi import (
        apply_techniques_to_midi,
        balance_track_velocities,
    )

    ckpt = Path("checkpoints/guitar_tech/guitar_tech_apply_last.pt")
    if not ckpt.is_file():
        return
    apply_techniques_to_midi(
        midi_path,
        checkpoint=ckpt,
        out_path=midi_path,
        seed=seed,
        none_bias=2.0,  # 学習の class-balance 補正。normal 寄りにしてうるささを抑える
        top_voice=True,
        role="lead",
        instrument_name="Lead",
    )
    balance_track_velocities(midi_path, out_path=midi_path)


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
    chord_peak_decode: bool = False,
    onset_segment_ticks: int = 0,
    with_lead: bool = True,
    harmony_thin_level: int = 0,
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
        bpm=float(bpm),
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
    section_notes: list[tuple[str, str, str, str, int, int, int, int, int]] = []
    has_lead_notes = False

    for i, section in enumerate(form.sections):
        section_seed = None if seed is None else int(seed) + i
        decode = section.decode_params
        prog = section.progression
        n_backing = n_lead = n_drum = n_bass = 0
        spans: tuple[ArrSpan, ...] = section.spans or (
            ArrSpan(bars=section.bars, parts=frozenset({"backing", "lead", "drum", "bass"})),
        )

        for j, span in enumerate(spans):
            span_seed = None if section_seed is None else int(section_seed) + j * 17
            tick_offset = bar_offset * TICKS_PER_BAR
            # パッチ化は 8 小節以上必須。短 span（lead_build の半区間など）は生成後に crop。
            gen_bars = max(_GEN_MIN_BARS, int(span.bars))
            # 噛み合わせ: drum → backing(グリッド) → bass(キック) → lead(隙間)
            drum_part: muspy.Music | None = None
            backing_part: muspy.Music | None = None

            if span.has("drum"):
                drum_part = generate_drum_music(
                    progression=prog,
                    key=key,
                    bars=gen_bars,
                    bpm=bpm,
                    bars_per_chord=bars_per_chord,
                    checkpoint=drum_ckpt,
                    onset_th=float(decode["drum_onset_th"]),
                    beat_type=section.beat_type,
                    model=drum_model,
                    device=device,
                )
                if gen_bars != span.bars:
                    drum_part = crop_music_to_bars(drum_part, span.bars)
                n_drum += sum(len(t.notes) for t in drum_part.tracks)
                music = _shift_and_merge(music, drum_part, tick_offset=tick_offset)

            grid = build_drum_grid(drum_part, bars=span.bars, subdiv_ticks=2)

            if span.has("backing"):
                backing_part = generate_backing_music(
                    progression=prog,
                    key=key,
                    bars=gen_bars,
                    bpm=bpm,
                    bars_per_chord=bars_per_chord,
                    checkpoint=checkpoint,
                    guitar_program=guitar_program,
                    seed=span_seed,
                    temperature=temperature,
                    chord_peak_decode=chord_peak_decode,
                    onset_th=float(decode["onset_th"]),
                    peak_min_distance=int(decode["peak_min_distance"]),
                    release_gap_ticks=int(decode["release_gap_ticks"]),
                    onset_segment_ticks=onset_segment_ticks,
                    stroke_groove=section.stroke_groove,
                    model=backing_model,
                    device=device,
                )
                if gen_bars != span.bars:
                    backing_part = crop_music_to_bars(backing_part, span.bars)
                if grid:
                    backing_part = snap_music_to_grid(
                        backing_part, grid, max_shift=2, skip_drums=True
                    )
                n_backing += sum(len(t.notes) for t in backing_part.tracks)
                music = _shift_and_merge(music, backing_part, tick_offset=tick_offset)

            if span.has("bass"):
                kick_times = extract_kick_times(drum_part) if drum_part is not None else None
                bass_part = generate_bass_music(
                    progression=prog,
                    key=key,
                    bars=gen_bars,
                    bpm=bpm,
                    bars_per_chord=bars_per_chord,
                    checkpoint=bass_ckpt,
                    onset_th=float(decode["bass_onset_th"]),
                    energy=section.energy,
                    kick_times=kick_times,
                    model=bass_model,
                    device=device,
                )
                if gen_bars != span.bars:
                    bass_part = crop_music_to_bars(bass_part, span.bars)
                if grid:
                    bass_part = snap_music_to_grid(
                        bass_part, grid, max_shift=2, skip_drums=True
                    )
                n_bass += sum(len(t.notes) for t in bass_part.tracks)
                music = _shift_and_merge(music, bass_part, tick_offset=tick_offset)

            if with_lead and lead_model is not None and span.has("lead"):
                blocked_power = (
                    power_chord_attack_times(backing_part)
                    if backing_part is not None
                    else frozenset()
                )
                lead_part = generate_lead_music(
                    progression=prog,
                    key=key,
                    bars=gen_bars,
                    bpm=bpm,
                    bars_per_chord=bars_per_chord,
                    checkpoint=lead_checkpoint,
                    guitar_program=guitar_program,
                    onset_th=float(decode["lead_onset_th"]),
                    sustain_th=float(decode.get("lead_sustain_th", 0.72)),
                    seed=seed,
                    harmony_thin_level=harmony_thin_level,
                    blocked_power_onsets=blocked_power,
                    section_role=section.role,
                    model=lead_model,
                    device=device,
                )
                if gen_bars != span.bars:
                    lead_part = crop_music_to_bars(lead_part, span.bars)
                blocked_gaps = collect_onsets(backing_part, tonal_only=True)
                blocked_gaps |= collect_onsets(drum_part, drums_only=True)
                lead_part = shift_lead_to_gaps(
                    lead_part,
                    blocked_gaps,
                    collide_window=0,
                    search_radius=2,
                    bars=span.bars,
                )
                n_here = sum(len(t.notes) for t in lead_part.tracks)
                n_lead += n_here
                if n_here > 0:
                    has_lead_notes = True
                music = _shift_and_merge(music, lead_part, tick_offset=tick_offset)

            bar_offset += span.bars

        section_notes.append(
            (
                section.label,
                section.progression,
                section.beat_type,
                section.arrangement_id,
                section.bars,
                n_backing,
                n_lead,
                n_drum,
                n_bass,
            )
        )

    if music is None:
        raise RuntimeError("セクションが空です")

    music.tempos = [muspy.Tempo(time=0, qpm=float(bpm))]
    save_music(music, output_path)
    if has_lead_notes:
        _apply_lead_note_techs(output_path, seed=seed)
    else:
        from guitar_tech.apply_midi import balance_track_velocities

        balance_track_velocities(output_path, out_path=output_path)

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
    for label, prog, beat, arr, bars, n_b, n_l, n_d, n_ba in section_notes:
        print(
            f"  {label}[{prog}|{beat}|{arr}|{bars}]: "
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
    parser.add_argument(
        "--chord-peak-decode",
        action="store_true",
        help="コード共通ピークデコードをON（既定OFF）",
    )
    parser.add_argument(
        "--onset-segment-ticks",
        type=int,
        default=0,
        help="backing all-onset blob 分割幅。既定0=分割なし",
    )
    parser.add_argument(
        "--harmony-thin-level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help=(
            "リード進行外音: 0=現状 / 1=ハード落としなし / "
            "2=減点弱め / 3=和声ほぼ無視 / 4=和声スコアなし"
        ),
    )
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
        chord_peak_decode=args.chord_peak_decode,
        onset_segment_ticks=args.onset_segment_ticks,
        with_lead=not args.no_lead,
        harmony_thin_level=args.harmony_thin_level,
    )


if __name__ == "__main__":
    main()
