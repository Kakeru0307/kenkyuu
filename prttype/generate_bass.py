"""ベース MIDI を生成する。

流れ: 進行骨格 → bass CVAE（cat4）→ ルートなぞり後処理 → Bass トラック。
kick_times がある場合、小節上限内でキック時刻にルートを補強する。
"""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import muspy

from util.checkpoint_paths import resolve_part_checkpoint
from inference import MIDI_DIR, load_model, predict_patches
from makeData.bass import BASS_DENSITY, max_onsets_for_energy, root_in_bass_range
from makeData.constants import BASS_PROGRAM, BEAT_TICKS, TICKS_PER_BAR
from makeData.progressions import resolve_progression_chords
from util.midi_to_patch import midi_to_patches
from util.patch_to_midi import patches_to_music, save_music
from util.progression_input import build_backing_skeleton_music, get_progression
from util.program_utils import remap_tonal_program
from sample_structure_params import DEFAULT_PRIOR_CKPT, resolve_structure_params

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = resolve_part_checkpoint("bass")
DEFAULT_PRIOR_CHECKPOINT = DEFAULT_PRIOR_CKPT

EnergyLevel = Literal["low", "mid", "high"]

# 間引き優先: 強拍ほど残す（同点は velocity）
_BEAT_PRIORITY = {0: 0, 8: 1, 4: 2, 12: 3}


def _apply_tempo(music: muspy.Music, bpm: float) -> muspy.Music:
    music.tempos = [muspy.Tempo(time=0, qpm=float(bpm))]
    return music


def _name_bass_track(music: muspy.Music, program: int) -> muspy.Music:
    for track in music.tracks:
        if not track.is_drum:
            track.name = "Bass"
            track.program = program
    return music


def _bass_track(music: muspy.Music) -> muspy.Track | None:
    for track in music.tracks:
        if not track.is_drum:
            return track
    return None


def _root_for_bar(
    chords: list[tuple[str, str]],
    bar: int,
    bars_per_chord: int,
) -> int:
    bpc = max(1, int(bars_per_chord))
    slot = (bar // bpc) % len(chords)
    return root_in_bass_range(chords[slot][0])


def snap_bass_to_roots(
    music: muspy.Music,
    *,
    progression: str,
    key: str,
    bars: int,
    bars_per_chord: int = 1,
    energy: EnergyLevel = "mid",
) -> muspy.Music:
    """ピッチをコードルートへ、四分グリッド・小節上限に揃える。"""
    track = _bass_track(music)
    if track is None or not track.notes:
        return music

    spec = get_progression(progression)
    chords = resolve_progression_chords(spec, key)
    max_onsets = max_onsets_for_energy(energy)

    # 1) 四分スナップ + 同時刻は velocity 優先で1音
    by_time: dict[int, muspy.Note] = {}
    for note in track.notes:
        t4 = (int(note.time) // BEAT_TICKS) * BEAT_TICKS
        bar = t4 // TICKS_PER_BAR
        if bar < 0 or bar >= bars:
            continue
        root = _root_for_bar(chords, bar, bars_per_chord)
        cand = muspy.Note(
            time=t4,
            pitch=root,
            duration=BEAT_TICKS,
            velocity=int(note.velocity),
        )
        prev = by_time.get(t4)
        if prev is None or int(cand.velocity) >= int(prev.velocity):
            by_time[t4] = cand

    # 2) 小節上限で間引き
    by_bar: dict[int, list[muspy.Note]] = defaultdict(list)
    for t, note in by_time.items():
        by_bar[t // TICKS_PER_BAR].append(note)

    kept: list[muspy.Note] = []
    for bar in range(bars):
        notes = by_bar.get(bar, [])
        if not notes:
            continue
        if len(notes) > max_onsets:
            notes = sorted(
                notes,
                key=lambda n: (
                    _BEAT_PRIORITY.get(int(n.time) % TICKS_PER_BAR, 9),
                    -int(n.velocity),
                    int(n.time),
                ),
            )[:max_onsets]
        kept.extend(sorted(notes, key=lambda n: int(n.time)))

    # 3) 次ノート直前で duration を切る（常に ≤4）
    kept.sort(key=lambda n: int(n.time))
    for i, note in enumerate(kept):
        end = int(note.time) + BEAT_TICKS
        if i + 1 < len(kept):
            end = min(end, int(kept[i + 1].time))
        note.duration = max(1, end - int(note.time))
        # 仕様: 音価は4。次が近いときだけ短くする
        if note.duration > BEAT_TICKS:
            note.duration = BEAT_TICKS

    track.notes = kept
    return music


def _reinforce_kick_roots(
    music: muspy.Music,
    *,
    progression: str,
    key: str,
    bars: int,
    bars_per_chord: int,
    kick_times: set[int] | frozenset[int],
    energy: EnergyLevel = "mid",
) -> muspy.Music:
    """キック時刻にベースルートが無い場合、小節上限内で追加する。"""
    if not kick_times:
        return music
    spec = get_progression(progression)
    chords = resolve_progression_chords(spec, key)
    bpc = max(1, bars_per_chord)
    max_onsets = max_onsets_for_energy(energy)

    bass_track = _bass_track(music)
    if bass_track is None:
        bass_track = muspy.Track(program=BASS_PROGRAM, is_drum=False, name="Bass")
        music.tracks.append(bass_track)

    existing = {int(n.time) for n in bass_track.notes}
    bar_counts: dict[int, int] = defaultdict(int)
    for n in bass_track.notes:
        bar_counts[int(n.time) // TICKS_PER_BAR] += 1

    for t in sorted(kick_times):
        t4 = (int(t) // BEAT_TICKS) * BEAT_TICKS
        if t4 in existing:
            continue
        bar = t4 // TICKS_PER_BAR
        if bar < 0 or bar >= bars:
            continue
        if bar_counts[bar] >= max_onsets:
            continue
        slot = (bar // bpc) % len(chords)
        root = root_in_bass_range(chords[slot][0])
        existing.add(t4)
        bar_counts[bar] += 1
        bass_track.append(
            muspy.Note(time=t4, pitch=root, duration=BEAT_TICKS, velocity=90)
        )
    bass_track.notes = sorted(bass_track.notes, key=lambda n: (n.time, n.pitch))
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
    onset_th: float = 0.38,
    energy: EnergyLevel = "mid",
    kick_times: set[int] | frozenset[int] | None = None,
    model: Any | None = None,
    device: Any | None = None,
) -> muspy.Music:
    """進行からベースを生成し、muspy.Music を返す。"""
    ckpt = Path(checkpoint)
    if not identity and not ckpt.is_file():
        raise FileNotFoundError(f"bass checkpoint がありません: {ckpt}")

    if energy not in BASS_DENSITY:
        energy = "mid"

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

    music = patches_to_music(output_patches, onset_segment_ticks=BEAT_TICKS)
    music = remap_tonal_program(music, bass_program)
    music = _name_bass_track(music, bass_program)
    music = snap_bass_to_roots(
        music,
        progression=progression,
        key=key,
        bars=bars,
        bars_per_chord=bpc,
        energy=energy,
    )
    if kick_times:
        music = _reinforce_kick_roots(
            music,
            progression=progression,
            key=key,
            bars=bars,
            bars_per_chord=bpc,
            kick_times=kick_times,
            energy=energy,
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
    parser.add_argument("--onset-th", type=float, default=0.38)
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

    energy: EnergyLevel = params.energy if params.energy in BASS_DENSITY else "mid"
    music = generate_bass_music(
        progression=params.progression,
        key=params.key,
        bars=params.bars,
        bpm=params.bpm,
        bars_per_chord=params.bars_per_chord,
        checkpoint=args.checkpoint,
        bass_program=args.bass_program,
        onset_th=args.onset_th,
        energy=energy,
    )
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or (
        MIDI_DIR / f"bass_{params.progression}_{params.key}_bpm{int(params.bpm):03d}.mid"
    )
    save_music(music, out)
    print(f"ベース: {out}  notes={sum(len(t.notes) for t in music.tracks)}")


if __name__ == "__main__":
    main()
