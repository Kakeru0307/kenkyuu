import muspy
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm

def batch_kl_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    """
    P, Q : np.ndarray
        Arrays of shape (N, C), each row is a discrete distribution or weights.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: {P.shape} vs {Q.shape}")
    P = P / (np.sum(P, axis=1, keepdims=True) + eps)
    Q = Q / (np.sum(Q, axis=1, keepdims=True) + eps)
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)
    kl_per_row = np.sum(P * np.log(P / Q), axis=1)   # shape (N,)
    return float(np.mean(kl_per_row))



def compute_doa(music:muspy.Music):
    reduced_feature = np.zeros((8, 12))
    track_wise_feature = defaultdict(lambda: np.zeros((8,12)))
    for track in music.tracks:
        if track.is_drum:
            continue
        program = track.program
        for note in track.notes:
            bar = note.start // 16
            pitch_class = note.pitch % 12
            dur = note.duration

            reduced_feature[bar, pitch_class] += dur
            track_wise_feature[program][bar, pitch_class] += dur 
    KLs = []
    for feature in track_wise_feature.values():
        kl = batch_kl_divergence(feature, reduced_feature)
        KLs.append(kl)
    doa = np.std(KLs)
    return doa

def midi_fpath_to_doa(fpath):
    music = muspy.read_midi(fpath)
    music.adjust_resolution(4)
    return compute_doa(music)

def compute_doa_folder(folder, file_name):
    doas = []
    for f in tqdm(os.listdir(folder)):
        midi_fpath = os.path.join(folder, f, file_name)
        doa = midi_fpath_to_doa(midi_fpath)
        doas.append(doa)
    return np.mean(doa)

if __name__ == "__main__":
    print(compute_doa_folder("control_exp_random","output_wo_drum.mid"))



