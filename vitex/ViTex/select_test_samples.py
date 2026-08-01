from utils.inference_utils import get_dataset
from dataloader import collate_fn_no_shift
from torch.utils.data import DataLoader
import os
from utils.output_utils import chord_to_midi, multitrack_pianoroll_to_midi, plot_tonal_hyperscore
import numpy as np
from tqdm import tqdm
if __name__=="__main__":
    dataset = get_dataset()
    dataloader = DataLoader(dataset,1,False, collate_fn=collate_fn_no_shift)

    
    for idx, batch_segment in tqdm(enumerate(dataloader)):
        if idx <= 830: continue
        save_folder = os.path.join("test_set_vis",f"{idx}")
        os.makedirs(save_folder, exist_ok=True)

        chord = batch_segment[0][0]
        np.save(os.path.join(save_folder, "gt_chd.npy"), chord.cpu().numpy())
        chord_to_midi(chord.permute(1,0,2), os.path.join(save_folder,"gt_chd.mid"))
        
        instrumentation = batch_segment[3][0].cpu().numpy()
        plot_tonal_hyperscore(instrumentation.transpose(1,2,0).reshape(8,8,11,3), os.path.join(save_folder, "ins.png"))
        np.save(os.path.join(save_folder, "gt_ins.npy"), instrumentation)

        pr_np = batch_segment[4][0].cpu().numpy().transpose(1,2,0)
        drum_np = batch_segment[2][0].cpu().numpy().reshape(128, 128)
        multitrack_pianoroll_to_midi(pr_np,drum_np,os.path.join(save_folder, "gt_w_drum.mid"))
        multitrack_pianoroll_to_midi(pr_np,None,os.path.join(save_folder, "gt_wo_drum.mid"))

        

        

