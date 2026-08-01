import random
import os
from utils.inference_utils import get_model, get_dataset
import numpy as np
from torch.utils.data import DataLoader
from dataloader import collate_fn
from tqdm import tqdm
import torch
from utils.output_utils import multitrack_pianoroll_to_midi

def generate(folder, batch_size, device):
    os.makedirs(folder, exist_ok=True)
    model = get_model().to(device)
    model.eval()
    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=True)
    idx = 0
    with torch.no_grad():
        
        for batch_segment in tqdm(dataloader):
            if idx >= 800:
                return
            bs = batch_segment[0].shape[0]
            cond1 = batch_segment[3].to(device)
            cond2 = batch_segment[1].to(device)
            cond_chord = batch_segment[0].to(device)

            x1, x2 = model.sample(
                3*torch.ones((bs,11,128,128)).to(torch.long).to(device),
                3*torch.ones((bs,1,128,128)).to(torch.long).to(device),
                -torch.ones_like(cond1), cond2, -torch.ones_like(cond_chord), 0.0, 0.0, None, None, 10, False
            )
            for i in range(bs):
                save_folder = os.path.join(folder, f"{idx+200}")
                os.makedirs(save_folder, exist_ok=True)
                pr_np = x1[i].cpu().numpy().transpose(1,2,0)
                drum_np = x2[i].cpu().numpy().reshape(128, 128)
                multitrack_pianoroll_to_midi(pr_np,None,os.path.join(save_folder, "ours.mid"))


                idx += 1


        
if __name__=="__main__":
    folder = "uncond_1000/"
    batch_size=50
    device = "cuda"
    generate(folder, batch_size, device)

        
