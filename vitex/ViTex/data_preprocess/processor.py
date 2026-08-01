import muspy
import numpy as np
from pathlib import Path

from utils.chord_utils import midi_fpath_to_chords
from utils.hyperscore_utils import get_hyperscore_per_track, get_hyperscore_drum
import pickle


class MIDIProcessor:

    def __init__(self, midi_fpath: Path):
        self.midi_fpath = midi_fpath
        self.music = muspy.read_midi(midi_fpath).adjust_resolution(4)
        self.bar_num = (self.music.get_end_time() // 16) + 1
        # W, C, H
        self.chord = self._extract_chord()
        # W, H, Instrument, 3
        self.drum_hyperscore, self.tonal_hyperscore = self._extract_hyperscore()
        self.tonal_notes, self.tonal_prefix_sum = self._extract_tonal_notes()
        self.drum_notes, self.drum_prefix_sum = self._extract_drum_notes()

    def _extract_chord(self):
        chord_fpath = self.midi_fpath.with_suffix(".chord")
        return midi_fpath_to_chords(self.midi_fpath, chord_fpath)

    def _extract_hyperscore(self):
        hyperscores = []
        for track in self.music.tracks:
            notes = [(note.time, note.duration, note.pitch)
                     for note in track.notes]
            if track.is_drum:
                track_hyperscore = get_hyperscore_drum(
                    notes, self.bar_num)
                program_num = -1
            else:
                track_hyperscore = get_hyperscore_per_track(
                    notes, self.bar_num)
                program_num = track.program // 8
                if program_num > 10:
                    program_num = 10

            hyperscores.append((program_num, track_hyperscore))
        drum_hyperscore = np.zeros((self.bar_num, 3, 2), dtype=bool)
        instrument_hyperscore = np.zeros((self.bar_num, 8, 11, 3), dtype=bool)
        for (program_num, track_hyperscore) in hyperscores:
            if program_num < 0:
                drum_hyperscore += np.array(
                    track_hyperscore).transpose(1, 2, 0)
            else:
                instrument_hyperscore[:, :, program_num,
                                      :] += np.array(track_hyperscore).transpose(1, 2, 0)
        return drum_hyperscore, instrument_hyperscore

    def _extract_tonal_notes(self):
        notes_list = []
        for track in self.music.tracks:
            if not track.is_drum:
                category = track.program // 8
                if category > 10:
                    category = 10
                notes_list.extend(
                    [
                        [
                            note.start,
                            note.pitch,
                            category,
                            note.duration + int(note.duration == 0),
                        ]
                        for note in track.notes
                    ]
                )

        sorted_notes_array = np.array(notes_list, dtype=np.int32)
        sorted_notes_array = sorted_notes_array[sorted_notes_array[:, 0].argsort(
        )]

        measure_length = 16
        max_measure = int(np.ceil(sorted_notes_array[-1, 0] / measure_length))
        measures = (sorted_notes_array[:, 0] // measure_length).astype(int)
        notes_per_measure = np.bincount(measures, minlength=max_measure)
        prefix_sum = np.cumsum(notes_per_measure)

        return sorted_notes_array, prefix_sum

    def _extract_drum_notes(self):
        notes_list = []
        for track in self.music.tracks:
            if track.is_drum:
                notes_list.extend(
                    [
                        [
                            note.start,
                            note.pitch,
                            note.duration + int(note.duration == 0),
                        ]
                        for note in track.notes
                    ]
                )
        sorted_notes_array = np.array(notes_list, dtype=np.int32)
        sorted_notes_array = sorted_notes_array[sorted_notes_array[:, 0].argsort(
        )]
        measure_length = 16
        max_measure = int(np.ceil(sorted_notes_array[-1, 0] / measure_length))
        measures = (sorted_notes_array[:, 0] // measure_length).astype(int)
        notes_per_measure = np.bincount(measures, minlength=max_measure)
        prefix_sum = np.cumsum(notes_per_measure)
        return sorted_notes_array, prefix_sum

    def to_dict(self):
        return {
            "chord": self.chord,
            "drum_hyperscore": self.drum_hyperscore,
            "drum_notes": self.drum_notes,
            "drum_prefix_sum": self.drum_prefix_sum,
            "tonal_hyperscore": self.tonal_hyperscore,
            "tonal_notes": self.tonal_notes,
            "tonal_prefix_sum": self.tonal_prefix_sum,
        }

    def save(self, save_fpath: Path):
        if len(self.drum_prefix_sum) < 10:
            raise ValueError("Drum track too short!")
        if len(self.tonal_prefix_sum) < 10:
            raise ValueError("Instrument track (excluding drum) too short!")

        with open(save_fpath, "wb") as f:
            pickle.dump(self.to_dict(), f)


if __name__ == "__main__":
    processor = MIDIProcessor(
        Path(r"C:\Projects\hyperscore\normalized_midi\00833.mid"))

    from utils.output_utils import notes_to_midi, plot_tonal_hyperscore, plot_pianoroll, plot_drum_hyperscore
    notes_to_midi(processor.tonal_notes, processor.drum_notes,
                  "recon.mid")
    plot_tonal_hyperscore(processor.tonal_hyperscore, "hs.png")
    plot_drum_hyperscore(processor.drum_hyperscore, "dr.png")

    plot_pianoroll(processor.tonal_notes, processor.drum_notes, "pr.png")
