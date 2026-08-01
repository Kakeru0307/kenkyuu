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
            if idx >= 1000:
                return
            bs = batch_segment[0].shape[0]
            cond1 = batch_segment[3].to(device)
            cond2 = batch_segment[1].to(device)
            cond_chord = batch_segment[0].to(device)

            gt_drum_pr = batch_segment[2].to(device)
            gt_tonal_pr = batch_segment[4].to(device)

            x1, x2 = model.sample(
                3*torch.ones((bs,11,128,128)).to(torch.long).to(device),
                3*torch.ones((bs,1,128,128)).to(torch.long).to(device),
                cond1, cond2, cond_chord, 1.0, 1.25, None, None, 10, False
            )
            for i in range(bs):
                save_folder = os.path.join(folder, f"{idx}")
                os.makedirs(save_folder, exist_ok=True)
                cond1_np = cond1[i].cpu().numpy()
                chord_np = cond_chord[i].cpu().numpy()
                np.save(os.path.join(save_folder,"chd.npy"),chord_np)
                np.save(os.path.join(save_folder,"hyperscore.npy"),cond1_np)
                pr_np = x1[i].cpu().numpy().transpose(1,2,0)
                drum_np = x2[i].cpu().numpy().reshape(128, 128)
                multitrack_pianoroll_to_midi(pr_np,drum_np,os.path.join(save_folder, "output_w_drum.mid"))
                multitrack_pianoroll_to_midi(pr_np,None,os.path.join(save_folder, "output_wo_drum.mid"))

                gt_pr_np = gt_tonal_pr[i].cpu().numpy().transpose(1,2,0)
                gt_drum_np = gt_drum_pr[i].cpu().numpy().reshape(128, 128)
                multitrack_pianoroll_to_midi(gt_pr_np,gt_drum_np,os.path.join(save_folder, "gt_w_drum.mid"))
                multitrack_pianoroll_to_midi(gt_pr_np,None,os.path.join(save_folder, "gt_wo_drum.mid"))

                

                idx += 1


        
if __name__=="__main__":
    folder = "cond_1000/"
    batch_size=200
    device = "cuda"
    generate(folder, batch_size, device)

        
