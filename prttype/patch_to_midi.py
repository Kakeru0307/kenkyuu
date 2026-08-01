"""ViTex 互換パッチから MIDI を再構成する。"""

from __future__ import annotations

from pathlib import Path

import muspy
import numpy as np

from midi_to_patch import (
    NUM_CATEGORIES,
    MidiPatch,
    denormalize_pianoroll,
)
from program_utils import program_for_category

DEFAULT_VELOCITY = 80
DEFAULT_RESOLUTION = 4


def _extract_notes_from_channel(
    channel: np.ndarray,
    bar_index: int,
    onset_segment_ticks: int = 2,
) -> list[muspy.Note]:
    """channel: (128, 128) time x pitch with values 0/1/2.

    Active blobs with no sustain value (2) — i.e. all-onset blobs produced
    by a model that hasn't learned the onset/sustain distinction — are split
    into onset_segment_ticks-tick segments so continuous activation becomes
    discrete strokes (e.g. 2 ticks = 8th note at resolution 4).

    Blobs that contain any sustain (2) use the classic 2→1 re-attack rule.
    """
    notes: list[muspy.Note] = []
    start_tick = bar_index * 16

    for pitch in range(128):
        row = channel[:, pitch]
        w = 0
        while w < 128:
            if row[w] == 0:
                w += 1
                continue

            # Collect the entire contiguous active run.
            blob_start = w
            while w < 128 and row[w] != 0:
                w += 1
            blob_end = w

            blob = row[blob_start:blob_end]
            has_sustain = bool((blob == 2).any())

            if has_sustain:
                # Classic 2 → 1 re-attack logic for proper onset/sustain blobs.
                pos = blob_start
                while pos < blob_end:
                    note_start = pos
                    prev = int(row[pos])
                    pos += 1
                    while pos < blob_end:
                        val = int(row[pos])
                        if val == 1 and prev == 2:
                            break
                        prev = val
                        pos += 1
                    duration = pos - note_start
                    notes.append(
                        muspy.Note(
                            time=start_tick + note_start,
                            pitch=pitch,
                            velocity=DEFAULT_VELOCITY,
                            duration=max(1, duration),
                        )
                    )
            else:
                # All-onset blob: split into onset_segment_ticks-tick strokes.
                pos = blob_start
                while pos < blob_end:
                    chunk_end = min(pos + onset_segment_ticks, blob_end)
                    notes.append(
                        muspy.Note(
                            time=start_tick + pos,
                            pitch=pitch,
                            velocity=DEFAULT_VELOCITY,
                            duration=max(1, chunk_end - pos),
                        )
                    )
                    pos = chunk_end

    return notes


def tonal_chw_to_tracks(tonal_chw: np.ndarray, bar_index: int) -> list[muspy.Track]:
    """(11, 128, 128) -> muspy tracks grouped by category."""
    if tonal_chw.ndim == 3 and tonal_chw.shape[0] == NUM_CATEGORIES:
        pianoroll = tonal_chw.transpose(1, 2, 0)
    else:
        pianoroll = tonal_chw

    tracks: list[muspy.Track] = []
    for category in range(NUM_CATEGORIES):
        channel_notes = _extract_notes_from_channel(
            pianoroll[:, :, category], bar_index
        )
        if not channel_notes:
            continue
        track = muspy.Track(
            program=program_for_category(category),
            is_drum=False,
            name=f"category_{category}",
        )
        track.extend(channel_notes)
        tracks.append(track)

    return tracks


def drum_chw_to_track(drum_chw: np.ndarray, bar_index: int) -> muspy.Track | None:
    if drum_chw.ndim == 3:
        channel = drum_chw[0]
    else:
        channel = drum_chw

    notes = _extract_notes_from_channel(channel, bar_index)
    if not notes:
        return None

    track = muspy.Track(program=0, is_drum=True, name="drums")
    track.extend(notes)
    return track


def patches_to_music(
    patches: list[MidiPatch],
    *,
    from_normalized: bool = False,
) -> muspy.Music:
    track_map: dict[int, muspy.Track] = {}
    drum_track: muspy.Track | None = None

    for patch in patches:
        tonal = patch.tonal_chw
        drum = patch.drum_chw
        if from_normalized:
            tonal = denormalize_pianoroll(tonal)
            drum = denormalize_pianoroll(drum)

        for track in tonal_chw_to_tracks(tonal, patch.bar_index):
            key = track.program
            if key not in track_map:
                track_map[key] = muspy.Track(
                    program=track.program, is_drum=False, name=track.name
                )
            track_map[key].extend(track)

        drum_part = drum_chw_to_track(drum, patch.bar_index)
        if drum_part is not None:
            if drum_track is None:
                drum_track = muspy.Track(program=0, is_drum=True, name="drums")
            drum_track.extend(drum_part)

    tracks = list(track_map.values())
    if drum_track is not None:
        tracks.append(drum_track)

    return muspy.Music(resolution=DEFAULT_RESOLUTION, tracks=tracks)


def save_music(music: muspy.Music, output_path: str | Path) -> None:
    muspy.write_midi(output_path, music)


if __name__ == "__main__":
    from midi_to_patch import midi_to_patches

    script_dir = Path(__file__).resolve().parent
    midi_path = script_dir / "midi" / "backing_marusa_C_bpm100_8bars_skeleton.mid"
    out_path = script_dir / "data" / "backing_skeleton_roundtrip.mid"

    patches = midi_to_patches(midi_path)
    music = patches_to_music(patches)
    save_music(music, out_path)
    print(f"往復変換 MIDI を保存しました: {out_path}")
    print(f"トラック数: {len(music.tracks)}")
