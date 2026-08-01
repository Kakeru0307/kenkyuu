import numpy as np
import os
import muspy
from utils.chord_utils import midi_fpath_to_chords
from utils.hyperscore_utils import get_hyperscore_per_track
from tqdm import tqdm
def safe_cosine_similarity(X, Y, eps=1e-12):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    X_norm = np.linalg.norm(X, keepdims=True)
    Y_norm = np.linalg.norm(Y, keepdims=True)

    X_safe = X / np.maximum(X_norm, eps)
    Y_safe = Y / np.maximum(Y_norm, eps)
    return np.dot(X_safe, Y_safe.T)


def eval_one_sample(folder):
    gt_chord = np.load(os.path.join(folder, 'chd.npy'))
    gt_hyperscore = np.load(os.path.join(folder, 'hyperscore.npy'))

    try:
        sample_chord = midi_fpath_to_chords(os.path.join(folder, 'output_wo_drum.mid'), os.path.join(folder, 'output_wo_drum.out'))
        sample_chord = sample_chord.transpose(1,0,2)
        
        sample_chord_pad = np.zeros((3, 32, 12))
        sample_chord_pad[:,0:sample_chord.shape[1],:] = sample_chord
        gt_chord_flatten = np.array(gt_chord).flatten()
        sample_chord_flatten = np.array(sample_chord_pad).flatten()
        chord_cosine_similarity = safe_cosine_similarity(gt_chord_flatten, 
                                                        sample_chord_flatten)
    except:
        chord_cosine_similarity = 0.0
        
    try:
        music = muspy.read_midi(os.path.join(folder, 'output_wo_drum.mid'))
        music.adjust_resolution(4)
        hyperscores = []
        for track in music.tracks:
            notes = [(note.time, note.duration, note.pitch)
                        for note in track.notes]
            if track.is_drum:
                continue
            track_hyperscore = get_hyperscore_per_track(
                notes, 8)
            program_num = track.program // 8
            if program_num > 10:
                program_num = 10

            hyperscores.append((program_num, track_hyperscore))
        sample_hyperscore = np.zeros((8, 8, 11, 3), dtype=bool)
        for (program_num, track_hyperscore) in hyperscores:
            sample_hyperscore[:, :, program_num,
                                        :] += np.array(track_hyperscore).transpose(1, 2, 0)
        sample_hyperscore = sample_hyperscore.reshape(8,8,33).transpose(2,0,1)
        
        gt_hyperscore_flatten = gt_hyperscore.flatten()
        sample_hyperscore_flatten = sample_hyperscore.flatten()
        hyperscore_cosine_similarity = safe_cosine_similarity(gt_hyperscore_flatten, 
                                                        sample_hyperscore_flatten)
    except:
        hyperscore_cosine_similarity = 0.0
    return chord_cosine_similarity, hyperscore_cosine_similarity

def eval_folder(folder):
    sim_cs = []
    sim_hs = []
    for f in tqdm(os.listdir(folder)):
        sim_c,sim_h = eval_one_sample(os.path.join(folder, f))
        sim_cs.append(sim_c)
        sim_hs.append(sim_h)
    return np.mean(sim_cs), np.mean(sim_hs)
        
    
if __name__ == "__main__":
    print(eval_folder("cond_1000/"))