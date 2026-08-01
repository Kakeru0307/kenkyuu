import pickle
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset


@dataclass
class Segment:
    chord: np.ndarray
    drum_hyperscore: np.ndarray
    tonal_hyperscore: np.ndarray
    drum_pianoroll: np.ndarray
    tonal_pianoroll: np.ndarray


class Song:
    def __init__(self, song: dict):

        self.chord = song["chord"]
        self.drum_hyperscore = song["drum_hyperscore"]
        self.drum_notes = song["drum_notes"]
        self.drum_prefix_sum = song["drum_prefix_sum"]
        self.tonal_hyperscore = song["tonal_hyperscore"]
        self.tonal_notes = song["tonal_notes"]
        self.tonal_prefix_sum = song["tonal_prefix_sum"]

        self.beat_num = self.chord.shape[0]
        bar_num = self.beat_num // 4
        self.bar_num = min(bar_num, len(self.drum_hyperscore), len(
            self.drum_prefix_sum), len(self.tonal_hyperscore), len(self.tonal_prefix_sum))

    def __len__(self):
        return self.bar_num - 8

    def __getitem__(self, idx: int) -> Segment:

        chord = self.chord[idx * 4: (idx + 8) * 4]
        drum_hyperscore = self.drum_hyperscore[idx: idx+8]
        tonal_hyperscore = self.tonal_hyperscore[idx: idx+8]

        drum_notes_start_idx = self.drum_prefix_sum[idx -
                                                    1] if idx > 0 else 0
        drum_notes_end_idx = self.drum_prefix_sum[idx + 7]
        drum_notes = self.drum_notes[drum_notes_start_idx:drum_notes_end_idx]

        drum_pianoroll = np.zeros(shape=(128, 128), dtype=np.int32)
        for drum_note in drum_notes:
            onset, pitch, duration = drum_note
            w = onset - idx*16
            dw = duration - 1
            drum_pianoroll[w, pitch] = 1
            drum_pianoroll[w+1:w+dw, pitch] = 2

        tonal_notes_start_idx = self.tonal_prefix_sum[idx -
                                                      1] if idx > 0 else 0
        tonal_notes_end_idx = self.tonal_prefix_sum[idx + 7]
        tonal_notes = self.tonal_notes[tonal_notes_start_idx:tonal_notes_end_idx]

        tonal_pianoroll = np.zeros(shape=(128, 128, 11), dtype=np.int32)
        for tonal_note in tonal_notes:
            onset, pitch, category, duration = tonal_note
            w = onset - idx*16
            dw = duration - 1
            tonal_pianoroll[w, pitch, category] = 1
            tonal_pianoroll[w+1:w+dw, pitch, category] = 2

        return Segment(chord, drum_hyperscore, tonal_hyperscore, drum_pianoroll, tonal_pianoroll)


class HyperscoreDataset(Dataset):
    def __init__(self, pkl_dir):
        self.pkl_dir = pkl_dir
        self.song_list = list(Path(pkl_dir).glob("*.pkl"))
        from tqdm import tqdm
        print(f"Loading {len(self.song_list)} songs from {pkl_dir}")
        self.songs = [Song(pickle.load(open(song, "rb")))
                      for song in tqdm(self.song_list)]
        self.lengths = [len(song) for song in self.songs]
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx: int) -> Segment:
        song_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")

        song = self.songs[song_idx]
        if song_idx > 0:
            segment_idx = idx - self.cumulative_lengths[song_idx - 1]
        else:
            segment_idx = idx

        segment = song[segment_idx]
        return segment


