import torch
from config import UNetConfig, TrainingConfig
import numpy as np

import torch.nn.functional as F
import torch.nn as nn

from d3pm import D3PM

import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch import optim
from model.dual_unet import DualUNet
from tqdm import tqdm
from config import UNetConfig, TrainingConfig
from dataloader import get_train_dataloader
from utils.chord_utils import chords_to_sequence
from utils.output_utils import plot_drum_hyperscore, plot_tonal_hyperscore, chord_to_midi, plot_pianoroll_from_multitrack_pianoroll, plot_pianoroll_from_drum_pianoroll, multitrack_pianoroll_to_midi
from dataset import HyperscoreDataset

import argparse


def main():
    dataset = HyperscoreDataset("/workspace/midi_pkl_test_new")

    cond_hyperscore_idx = [840]
    cond_hyperscores = []
    cond_inps = []
    for idx, i in enumerate(cond_hyperscore_idx):
        sample = dataset[i]
        drum_hyperscore = sample.drum_hyperscore
        plot_drum_hyperscore(drum_hyperscore, f"fig/drum_hyperscore_{idx}.svg")
        tonal_hyperscore = sample.tonal_hyperscore
        plot_tonal_hyperscore(tonal_hyperscore, f"fig/tonal_hyperscore_{idx}.svg")
        chd = sample.chord
        chord_to_midi(chd, "fig/chord.mid")
        cond_hyperscores.append((torch.from_numpy(tonal_hyperscore.transpose(2,3,0,1).reshape(33,8,8)).float(),
                                 torch.from_numpy(drum_hyperscore.transpose(2,0,1)).float()))
        tonal_pianoroll = sample.tonal_pianoroll
        plot_pianoroll_from_multitrack_pianoroll(tonal_pianoroll, f"fig/tonal_pianoroll_{idx}.svg")
        # plot_pianoroll_from_drum_pianoroll(drum_pianoroll, f"{cond_hyperscore_path}/drum_pianoroll_{idx}.png")
        # multitrack_pianoroll_to_midi(tonal_pianoroll, drum_pianoroll, f"{cond_hyperscore_path}/orig_{idx}.mid")

        
    
    

    # print("=============condition hyperscore ready==================")
    # print("Generating...")
    # num_gen = 20
    # use_inp = False
    # for i1, cond_chord in enumerate(cond_chords):
    #     cond_chord = cond_chord[None, :, :, :].to(device)
    #     for i2, cond_hyperscore in enumerate(cond_hyperscores):
    #         tonal_hyperscore, drum_hyperscore = cond_hyperscore
    #         tonal_hyperscore = tonal_hyperscore[None, :, :, :].to(device)
    #         drum_hyperscore = drum_hyperscore[None, :, :, :].to(device)
    #         x_inp = None
    #         mask = None
    #         if use_inp:
    #             x_inp = cond_inps[i2]
    #             x_inp = x_inp[None, :,:,:].to(device).to(torch.long)
    #             mask = torch.zeros_like(x_inp).float()
    #             mask[:,5,:,:] = 1


    #         for n_gen in range(num_gen):
    #             sub_path = f"{save_path}/d3pm/chord_{i1}_hyperscore_{i2}"
    #             os.makedirs(sub_path, exist_ok=True)

    #             init_noise_tonal_pianoroll = 3*torch.ones((1, 11, 128, 128)
    #                                       ).to(device).to(torch.long)
    #             init_noise_drum_pianoroll = 3*torch.ones((1, 1, 128, 128)
    #                                       ).to(device).to(torch.long)
    #             x1, x2, x1s, x2s = d3pm.sample(
    #                 init_noise_tonal_pianoroll, init_noise_drum_pianoroll, tonal_hyperscore, drum_hyperscore, cond_chord,
    #                 chord_scale=args.chord_scale, hyperscore_scale=args.hyperscore_scale, x_inp=x_inp, mask=mask
    #             )

    #             x1_np = x1.cpu().numpy()[0]
    #             x1_np = x1_np.transpose(1,2,0)
    #             x2_np = x2.cpu().numpy()[0]
    #             x2_np = x2_np.reshape(128, 128)
    #             multitrack_pianoroll_to_midi(x1_np, x2_np, f"{sub_path}/{n_gen}.mid")
    #             plot_pianoroll_from_multitrack_pianoroll(x1_np, f"{sub_path}/{n_gen}_tonal_pianoroll.png")
    #             plot_pianoroll_from_drum_pianoroll(x2_np, f"{sub_path}/{n_gen}_drum_pianoroll.png")

if __name__ == "__main__":
    main()
