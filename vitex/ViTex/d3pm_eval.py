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
    parser = argparse.ArgumentParser()
    parser.add_argument("--chord_scale", type=float, required=True,
                        help="chord scale (e.g., 1.0)")
    parser.add_argument("--hyperscore_scale", type=float, required=True,
                        help="hyperscore scale (e.g., 1.0)")
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    

    import os
    os.makedirs(args.save_path, exist_ok=True)
    save_path = f"{args.save_path}/h_{args.hyperscore_scale}_c_{args.chord_scale}"
    cond_chord_path = f"{save_path}/cond_chord"
    cond_hyperscore_path = f"{save_path}/cond_hyperscore"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(cond_chord_path, exist_ok=True)
    os.makedirs(cond_hyperscore_path, exist_ok=True)

    unet_config = UNetConfig(
        x1_shape=(11, 128, 128),
        x2_shape=(1, 128, 128),
        d_temb=256,
        hyperscore1_shape=(33, 8, 8),
        chord1_shape=(3, 32, 12),
        hyperscore2_shape=(2, 8, 3),
        base_channel=32,
        channel_mult=[1, 2, 4, 4],
        num_blocks=2,
        hyperscore1_level=[2, 3],
        hyperscore2_level=[2, 3],
        chord_level=[1, 2],
        fuse_rhythm_level=[],
        self_attn_level=[2, 3],
        num_heads=4,
        N=4,
        use_d3pm=True
    )


    training_config = TrainingConfig(
        batch_size=24,
        lr=5e-4,
        num_epochs=20,
        num_workers=10,
        pin_memory=True,
        accumulation_step=5,
        output_path="output",
        data_path="/workspace/midi_pkl",
        save_interval=20000,
        log_interval=100,
        null_condition_prob=0.5
    )

    # 2. 初始化 dummy optimizer 和 scheduler，用于恢复
    dataset = HyperscoreDataset("/workspace/midi_pkl_test_new")
    # 2. Load model
    model = DualUNet(**vars(unet_config))

    optimizer = optim.AdamW(model.parameters(), lr=training_config.lr)

    d3pm = D3PM(model, 1000, num_classes=4, hybrid_loss_coeff=0.0)
    accelerator = Accelerator()
    # Prepare everything with accelerator
    model, d3pm, optimizer, _ = accelerator.prepare(
        model, d3pm, optimizer, []
    )
    accelerator.load_state(
        "/workspace/hyperscore/output/20250909-070831/ckpt")  # 改成你实际路径
    model = accelerator.unwrap_model(model)
    model.eval()
    d3pm: D3PM = accelerator.unwrap_model(d3pm)
    d3pm.eval()
    print("================Model Ready==================")
    # 4. 准备条件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 示例：生成8小节32拍的和弦序列（每拍一个和弦）
    example_chords1 = ['A', 'A', 'A', 'A',
                       'B7', 'B7', 'B7', 'B7',
                       'D', 'D', 'D', 'D',
                       'A', 'A', 'A', 'A',
                       'A', 'A', 'A', 'A',
                       'B7', 'B7', 'B7', 'B7',
                       'D', 'D', 'D', 'D',
                       'A', 'A', 'A', 'A',]
    sequence1 = chords_to_sequence(example_chords1, beats_per_bar=4, bars=8)
    example_chords2 = ['A#m7', 'A#m7', 'A#m7', 'A#m7',
                       'D#7', 'D#7', 'D#7', 'D',
                       'Cm7', 'Cm7', 'Bb', 'Bb',
                       'Am7', 'Am7', 'Ab', 'Ab',
                       'Bm7', 'Bm7', 'A', 'A',
                       'F#7', 'F#7', 'F#7', 'F#7',
                       'F#7', 'F#7', 'Gm', 'Gm',
                       'Ab', 'Ab', 'Bb', 'Bb',]
    sequence2 = chords_to_sequence(example_chords2, beats_per_bar=4, bars=8)
    example_chords3 = ['F', 'F', 'F', 'F',
                       'G', 'G', 'G', 'G',
                       'Em', 'Em', 'Em', 'Em',
                       'Am', 'Am', 'Am', 'Am',
                       'Dm', 'Dm', 'Dm', 'Dm',
                       'G', 'G', 'G', 'G',
                       'C', 'C', 'C', 'C',
                       'C', 'C', 'C', 'C',]
    sequence3 = chords_to_sequence(example_chords3, beats_per_bar=4, bars=8)
    example_chords4 = ['FM7', 'FM7', 'Em7', 'Em7',
                       'Am', 'Am', 'Am', 'Am',
                       'Dm7', 'Dm7', 'Em7', 'Em7',
                       'Am', 'E', 'Gm7', 'C7',
                       'FM7', 'FM7', 'Em7', 'Em7',
                       'Am', 'E', 'Gm7', 'C7',
                       'Dm7', 'Dm7', 'Em7', 'Em7',
                       'A','A','A','A'
                       ]
    sequence4 = chords_to_sequence(example_chords4, beats_per_bar=4, bars=8)
    cond_chords = [sequence4]

    for idx, cond1 in enumerate(cond_chords):
        chord_to_midi(cond1.permute(1,0,2), f"{cond_chord_path}/orig_chord_{idx}.midi")
    
    print("=============condition chord ready====================")
    test_hyperscore = np.zeros((8,8,11,3), dtype=bool)
    test_hyperscore[:,1,4,0] = True
    test_hyperscore[:,1,4,1] = False
    test_hyperscore[:,1,4,2] = True

    test_hyperscore[:,2,0,0] = False
    test_hyperscore[:,2,0,1] = True
    test_hyperscore[:,2,0,2] = True

    test_hyperscore[:,3,3,0] = False
    test_hyperscore[:,3,3,1] = True
    test_hyperscore[:,3,3,2] = False

    test_hyperscore[:,4,5,0] = True
    test_hyperscore[:,4,5,1] = False
    test_hyperscore[:,4,5,2] = False



    cond_hyperscore_idx = [49]
    cond_hyperscores = []
    cond_inps = []
    for idx, i in enumerate(cond_hyperscore_idx):
        sample = dataset[i]
        drum_hyperscore = sample.drum_hyperscore
        plot_drum_hyperscore(drum_hyperscore, f"{cond_hyperscore_path}/drum_hyperscore_{idx}.png")
        tonal_hyperscore = sample.tonal_hyperscore
        plot_tonal_hyperscore(tonal_hyperscore, f"{cond_hyperscore_path}/tonal_hyperscore_{idx}.png")
        cond_hyperscores.append((torch.from_numpy(tonal_hyperscore.transpose(2,3,0,1).reshape(33,8,8)).float(),
                                 torch.from_numpy(drum_hyperscore.transpose(2,0,1)).float()))
        drum_pianoroll = sample.drum_pianoroll
        tonal_pianoroll = sample.tonal_pianoroll
        cond_inps.append(torch.from_numpy(tonal_pianoroll.transpose(2,0,1)).float())
        plot_pianoroll_from_multitrack_pianoroll(tonal_pianoroll, f"{cond_hyperscore_path}/tonal_pianoroll_{idx}.png")
        plot_pianoroll_from_drum_pianoroll(drum_pianoroll, f"{cond_hyperscore_path}/drum_pianoroll_{idx}.png")
        multitrack_pianoroll_to_midi(tonal_pianoroll, drum_pianoroll, f"{cond_hyperscore_path}/orig_{idx}.mid")

        
    
    

    print("=============condition hyperscore ready==================")
    print("Generating...")
    num_gen = 20
    use_inp = False
    for i1, cond_chord in enumerate(cond_chords):
        cond_chord = cond_chord[None, :, :, :].to(device)
        for i2, cond_hyperscore in enumerate(cond_hyperscores):
            tonal_hyperscore, drum_hyperscore = cond_hyperscore
            tonal_hyperscore = tonal_hyperscore[None, :, :, :].to(device)
            drum_hyperscore = drum_hyperscore[None, :, :, :].to(device)
            x_inp = None
            mask = None
            if use_inp:
                x_inp = cond_inps[i2]
                x_inp = x_inp[None, :,:,:].to(device).to(torch.long)
                mask = torch.zeros_like(x_inp).float()
                mask[:,5,:,:] = 1


            for n_gen in range(num_gen):
                sub_path = f"{save_path}/d3pm/chord_{i1}_hyperscore_{i2}"
                os.makedirs(sub_path, exist_ok=True)

                init_noise_tonal_pianoroll = 3*torch.ones((1, 11, 128, 128)
                                          ).to(device).to(torch.long)
                init_noise_drum_pianoroll = 3*torch.ones((1, 1, 128, 128)
                                          ).to(device).to(torch.long)
                x1, x2, x1s, x2s = d3pm.sample(
                    init_noise_tonal_pianoroll, init_noise_drum_pianoroll, tonal_hyperscore, drum_hyperscore, cond_chord,
                    chord_scale=args.chord_scale, hyperscore_scale=args.hyperscore_scale, x_inp=x_inp, mask=mask
                )

                x1_np = x1.cpu().numpy()[0]
                x1_np = x1_np.transpose(1,2,0)
                x2_np = x2.cpu().numpy()[0]
                x2_np = x2_np.reshape(128, 128)
                multitrack_pianoroll_to_midi(x1_np, x2_np, f"{sub_path}/{n_gen}.mid")
                plot_pianoroll_from_multitrack_pianoroll(x1_np, f"{sub_path}/{n_gen}_tonal_pianoroll.png")
                plot_pianoroll_from_drum_pianoroll(x2_np, f"{sub_path}/{n_gen}_drum_pianoroll.png")

if __name__ == "__main__":
    main()
