from dataset import Segment, HyperscoreDataset
from torch.utils.data import DataLoader
from config import TrainingConfig
from dataclasses import dataclass
import torch
import random
import numpy as np


def collate_fn(batch: list[Segment]):
    batch_chord = []
    batch_drum_hyperscore = []
    batch_drum_pianoroll = []
    batch_tonal_hyperscore = []
    batch_tonal_pianoroll = []
    for segment in batch:
        shift_key = random.randint(-6, 5)
        # W, C, H = 32, 3, 12 to C, W, H = 3, 32, 12
        chord = np.roll(segment.chord, shift_key, axis=-1).transpose(1, 0, 2)
        # W, H, C = 8, 3, 2 to C, W, H = 2, 8, 3
        drum_hyperscore = segment.drum_hyperscore.transpose(2, 0, 1)
        # W, H = 128, 128 to C, W, H = 1, 128, 128
        drum_pianoroll = segment.drum_pianoroll[np.newaxis, ...]
        # W, H, C1, C2 = 8, 8, 11, 3 to C1*C2, W, H = 33, 8, 8
        tonal_hyperscore = segment.tonal_hyperscore.transpose(
            2, 3, 0, 1).reshape(11 * 3, 8, 8)
        # W, H, C = 128, 128, 11 to C, W, H = 11, 128, 128
        tonal_pianoroll = np.roll(
            segment.tonal_pianoroll, shift_key, axis=1).transpose(2, 0, 1)

        batch_chord.append(chord)
        batch_drum_hyperscore.append(drum_hyperscore)
        batch_drum_pianoroll.append(drum_pianoroll)
        batch_tonal_hyperscore.append(tonal_hyperscore)
        batch_tonal_pianoroll.append(tonal_pianoroll)

    return (torch.from_numpy(np.array(batch_chord)).float(),
            torch.from_numpy(np.array(batch_drum_hyperscore)).float(),
            torch.from_numpy(np.array(batch_drum_pianoroll)).float(),
            torch.from_numpy(np.array(batch_tonal_hyperscore)).float(),
            torch.from_numpy(np.array(batch_tonal_pianoroll)).float(),
            )

def collate_fn_no_shift(batch: list[Segment]):
    batch_chord = []
    batch_drum_hyperscore = []
    batch_drum_pianoroll = []
    batch_tonal_hyperscore = []
    batch_tonal_pianoroll = []
    for segment in batch:
        shift_key = 0
        # W, C, H = 32, 3, 12 to C, W, H = 3, 32, 12
        chord = np.roll(segment.chord, shift_key, axis=-1).transpose(1, 0, 2)
        # W, H, C = 8, 3, 2 to C, W, H = 2, 8, 3
        drum_hyperscore = segment.drum_hyperscore.transpose(2, 0, 1)
        # W, H = 128, 128 to C, W, H = 1, 128, 128
        drum_pianoroll = segment.drum_pianoroll[np.newaxis, ...]
        # W, H, C1, C2 = 8, 8, 11, 3 to C1*C2, W, H = 33, 8, 8
        tonal_hyperscore = segment.tonal_hyperscore.transpose(
            2, 3, 0, 1).reshape(11 * 3, 8, 8)
        # W, H, C = 128, 128, 11 to C, W, H = 11, 128, 128
        tonal_pianoroll = np.roll(
            segment.tonal_pianoroll, shift_key, axis=1).transpose(2, 0, 1)

        batch_chord.append(chord)
        batch_drum_hyperscore.append(drum_hyperscore)
        batch_drum_pianoroll.append(drum_pianoroll)
        batch_tonal_hyperscore.append(tonal_hyperscore)
        batch_tonal_pianoroll.append(tonal_pianoroll)

    return (torch.from_numpy(np.array(batch_chord)).float(),
            torch.from_numpy(np.array(batch_drum_hyperscore)).float(),
            torch.from_numpy(np.array(batch_drum_pianoroll)).float(),
            torch.from_numpy(np.array(batch_tonal_hyperscore)).float(),
            torch.from_numpy(np.array(batch_tonal_pianoroll)).float(),
            )



def get_train_dataloader(cfg: TrainingConfig) -> DataLoader:
    dataset = HyperscoreDataset(cfg.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return dataloader
