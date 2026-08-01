from model.dual_unet import DualUNet as UNet
from d3pm import D3PM
from config import UNetConfig
import torch
import os
from .output_utils import multitrack_pianoroll_to_midi, chord_to_midi, plot_tonal_hyperscore, plot_pianoroll_from_multitrack_pianoroll, plot_pianoroll_from_drum_pianoroll
from .chord_utils import chords_to_sequence
from dataset import HyperscoreDataset
import numpy as np
import random

def get_model() -> D3PM:
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
    unet = UNet(**vars(unet_config))
    model = D3PM(unet, 1000, num_classes=4, hybrid_loss_coeff=0.0)
    ckpt_path = "checkpoint/d3pm.ckpt"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model

def get_dataset() -> HyperscoreDataset:
    dataset = HyperscoreDataset("data/midi_pkl_test")
    return dataset

DRUM_HYPERSCORE_FIXED = torch.zeros((1,2,8,3))
DRUM_HYPERSCORE_FIXED[:,0,:,:] = 1.0

def uncond_gen(model:D3PM, bar_gen, device):
    if bar_gen < 8 or bar_gen % 4 != 0:
        raise ValueError("Bars must be greater than 8 and divisible by 4!")
    model = model.to(device)
    num_call = 1 + (bar_gen - 8) // 4

    canvas_1 = torch.zeros((11, bar_gen * 16, 128),dtype=torch.long).to(device)
    canvas_2 = torch.zeros((1, bar_gen * 16, 128),dtype=torch.long).to(device)


    for i in range(num_call):
        x_inp = None if i == 0 else canvas_1[:,64*i:64*i+128,:][None,:,:,:]
        mask = None if i == 0 else torch.zeros_like(x_inp)
        if i != 0:
            mask[:,:,0:64,:] = 1

        print(f"Generating bar [{4*i+4*(i!=0)},{4*i+8}]")
        x1, x2 = model.sample(
            3*torch.ones((1, 11, 128, 128)).to(device).to(torch.long),
            3*torch.ones((1, 1, 128, 128)).to(device).to(torch.long),
            -torch.ones((1,33,8,8)).to(device),
            DRUM_HYPERSCORE_FIXED.to(device),
            -torch.ones((1,3,32,12)).to(device),
            chord_scale=0.0, hyperscore_scale=0.0, x_inp=x_inp, mask=mask, return_all=False
        )
        canvas_1[:,64*i:64*i+128,:] = x1[0]
        canvas_2[:,64*i:64*i+128,:] = x2[0]
    
    canvas_1 = canvas_1.cpu().numpy()
    canvas_1 = canvas_1.transpose(1,2,0)
    canvas_2 = canvas_2.cpu().numpy()
    canvas_2 = canvas_2.reshape(-1, 128)
    return canvas_1, canvas_2

def uncond_gen_api(model:D3PM, num_gen, bar_gen, save_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_folder, exist_ok=True)
    for i in range(num_gen):
        x1, x2 = uncond_gen(model, bar_gen, device)
        multitrack_pianoroll_to_midi(x1, x2, os.path.join(save_folder,f"{i}.mid"))
        plot_pianoroll_from_multitrack_pianoroll(x1, os.path.join(save_folder, f"{i}.png"))
        plot_pianoroll_from_drum_pianoroll(x2, os.path.join(save_folder, f"{i}_d.png"))

def get_chord(sequence,dataset:HyperscoreDataset,idx,save_folder):
    if sequence is None and dataset is None:
        raise ValueError("Chord not specified!")
    if sequence:
        chord = chords_to_sequence(sequence)
    else:
        if idx is None:
            idx = random.randrange(0, len(dataset))
        sample = dataset[idx]
        chord = torch.from_numpy(sample.chord.transpose(1,0,2))
    chord_to_midi(chord.permute(1,0,2).to(torch.float32), os.path.join(save_folder, "chd.mid"))
    return chord

def get_hyperscore(hyperscore,dataset:HyperscoreDataset,idx,save_folder):
    if hyperscore is None and dataset is None:
        raise ValueError("Hyperscore not specified!")
    if hyperscore is not None:
        hyperscore = hyperscore
    else:
        if idx is None:
            idx = random.randrange(0, len(dataset))
        sample = dataset[idx]
        hyperscore = sample.tonal_hyperscore
    plot_tonal_hyperscore(hyperscore, os.path.join(save_folder,"ins.png"))
    hyperscore = torch.from_numpy(hyperscore.transpose(
            2, 3, 0, 1).reshape(11 * 3, 8, 8))
    return hyperscore

def get_x_inp(dataset:HyperscoreDataset,idx,save_folder):
    sample=dataset[idx]
    x_inp = sample.tonal_pianoroll
    multitrack_pianoroll_to_midi(x_inp, None, os.path.join(save_folder, "gt.mid"))

    x_inp = torch.from_numpy(x_inp.transpose(2, 0, 1))
    return x_inp

def cond_gen(model:D3PM, bar_gen, chord, chord_scale, hyperscore, hyperscore_scale, device):

    if bar_gen < 8 or bar_gen % 4 != 0:
        raise ValueError("Bars must be greater than 8 and divisible by 4!")
    model = model.to(device)
    num_call = 1 + (bar_gen - 8) // 4

    canvas_1 = torch.zeros((11, bar_gen * 16, 128),dtype=torch.long).to(device)
    canvas_2 = torch.zeros((1, bar_gen * 16, 128),dtype=torch.long).to(device)

    chord_all = -torch.ones((3, 4*bar_gen, 12)).to(device)
    chord_all[:,0:chord.shape[1],:] = chord
   
    hyperscore_all = -torch.ones((33, bar_gen, 8)).to(device)
    hyperscore_all[:,0:hyperscore.shape[1],:] = hyperscore


    for i in range(num_call):
        x_inp = None if i == 0 else canvas_1[:,64*i:64*i+128,:][None,:,:,:]
        mask = None if i == 0 else torch.zeros_like(x_inp)
        
        if i != 0:
            mask[:,:,0:64,:] = 1

        print(f"Generating bar [{4*i+4*(i!=0)},{4*i+8}]")
        x1, x2 = model.sample(
            3*torch.ones((1, 11, 128, 128)).to(device).to(torch.long),
            3*torch.ones((1, 1, 128, 128)).to(device).to(torch.long),
            hyperscore_all[None, :, 4*i:4*i+8],
            DRUM_HYPERSCORE_FIXED.to(device),
            chord_all[None,:,16*i:16*i+32,:],
            chord_scale=chord_scale, hyperscore_scale=hyperscore_scale, x_inp=x_inp, mask=mask, return_all=False
        )
        canvas_1[:,64*i:64*i+128,:] = x1[0]
        canvas_2[:,64*i:64*i+128,:] = x2[0]
    
    canvas_1 = canvas_1.cpu().numpy()
    canvas_1 = canvas_1.transpose(1,2,0)
    canvas_2 = canvas_2.cpu().numpy()
    canvas_2 = canvas_2.reshape(-1, 128)
    return canvas_1, canvas_2
    
def cond_gen_api(model:D3PM, num_gen, bar_gen, chord, chord_scale, hyperscore, hyperscore_scale, save_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_folder, exist_ok=True)
    for i in range(num_gen):
        x1, x2 = cond_gen(model, bar_gen,chord, chord_scale, hyperscore, hyperscore_scale, device)
        multitrack_pianoroll_to_midi(x1, x2, os.path.join(save_folder,f"{i}.mid"))
        plot_pianoroll_from_multitrack_pianoroll(x1, os.path.join(save_folder, f"{i}.png"))
        plot_pianoroll_from_drum_pianoroll(x2, os.path.join(save_folder, f"{i}_d.png"))


def continuation_gen(model:D3PM, bar_gen, chord, chord_scale, hyperscore, hyperscore_scale, x_inp, device):

    if bar_gen != 8:
        raise ValueError("Only 8 bars is supported!")
    model = model.to(device)

    
    chord_all = -torch.ones((3, 4*bar_gen, 12)).to(device)
    chord_all[:,0:chord.shape[1],:] = chord
   
    hyperscore_all = -torch.ones((33, bar_gen, 8)).to(device)
    hyperscore_all[:,0:hyperscore.shape[1],:] = hyperscore

    x_inp = x_inp[None,:,:,:].to(device)
    mask = torch.zeros_like(x_inp)
    mask[:,:,0:32,:] = 1.0
    
   
    x1, x2 = model.sample(
        3*torch.ones((1, 11, 128, 128)).to(device).to(torch.long),
        3*torch.ones((1, 1, 128, 128)).to(device).to(torch.long),
        hyperscore_all[None, :, :, :],
        DRUM_HYPERSCORE_FIXED.to(device),
        chord_all[None,:,:,:],
        chord_scale=chord_scale, hyperscore_scale=hyperscore_scale, x_inp=x_inp, mask=mask, return_all=False
    )
    canvas_1=x1[0]
    canvas_2=x2[0]

    canvas_1 = canvas_1.cpu().numpy()
    canvas_1 = canvas_1.transpose(1,2,0)
    canvas_2 = canvas_2.cpu().numpy()
    canvas_2 = canvas_2.reshape(-1, 128)
    return canvas_1, canvas_2

def continuation_gen_api(model:D3PM, num_gen, bar_gen, chord, chord_scale, hyperscore, hyperscore_scale,x_inp, save_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_folder, exist_ok=True)
    for i in range(num_gen):
        x1, x2 = continuation_gen(model, bar_gen,chord, chord_scale, hyperscore, hyperscore_scale,x_inp, device)
        multitrack_pianoroll_to_midi(x1, x2, os.path.join(save_folder,f"{i}.mid"))