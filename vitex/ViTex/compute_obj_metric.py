import muspy
import pickle
from tqdm import tqdm
import os
import argparse
import numpy as np
from collections import defaultdict
from utils.chord_utils import midi_fpath_to_chords
from utils.hyperscore_utils import get_hyperscore_per_track

def get_target_fpaths(folder, file_name):
    fpaths = []
    for f in os.listdir(folder):
        if os.path.exists(os.path.join(folder, f, file_name)):
            fpaths.append(os.path.join(folder, f, file_name))
        else:
            raise ValueError(f"{file_name} not found at {os.path.join(folder, f)}")
    return fpaths

def compute_test_set(folder):
    all_pickles = []
    for filename in os.listdir(folder):
        if filename.endswith((".pkl", ".pickle")):
            filepath = os.path.join(folder, filename)
            with open(filepath, "rb") as f:
                obj = pickle.load(f)
            all_pickles.append(obj)
    print(all_pickles[10])


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

def batch_entropy(P: np.ndarray, eps:float = 1e-12) -> float:
    P = np.asarray(P, dtype=float)
    P = P / (np.sum(P, axis=1, keepdims=True) + eps)
    entropy_per_row = -np.sum(P*np.log2(P + eps), axis=1)
    return float(np.mean(entropy_per_row))



def compute_doa(music:muspy.Music):
    reduced_feature = np.zeros((8, 12))
    track_wise_feature = defaultdict(lambda: np.zeros((8,12)))
    for track in music.tracks:
        if track.is_drum:
            continue
        program = track.program
        for note in track.notes:
            bar = note.start // 16
            if bar >= 8:
                continue
            pitch_class = note.pitch % 12
            dur = note.duration
            reduced_feature[bar, pitch_class] += dur
            track_wise_feature[program][bar, pitch_class] += dur 
    KLs = []
    for feature in track_wise_feature.values():
        kl = batch_kl_divergence(feature, reduced_feature)
        KLs.append(kl)
    if len(KLs) == 0 or len(KLs) == 1:
        return None
    doa = np.std(KLs, ddof=1)
    return doa

def midi_fpath_to_doa(fpath):
    music = muspy.read_midi(fpath)
    music.adjust_resolution(4)
    return compute_doa(music)

def compute_doa_fpaths(fpaths):
    doas = []
    for fpath in fpaths:
        doa = midi_fpath_to_doa(fpath)
        if doa is None:
            continue
        doas.append(doa)
    return np.mean(doas)

def compute_pce(music:muspy.Music):
    reduced_feature = np.zeros((8, 12))
    for track in music.tracks:
        if track.is_drum:
            continue
        program = track.program
        for note in track.notes:
            bar = note.start // 16
            if bar >= 8:
                continue
            pitch_class = note.pitch % 12
            dur = note.duration
            reduced_feature[bar, pitch_class] += dur
    pce = batch_entropy(reduced_feature)
    return pce

def midi_fpath_to_pce(fpath):
    music = muspy.read_midi(fpath)
    music.adjust_resolution(4)
    return compute_pce(music)



def compute_pce_fpaths(fpaths):
    pces = []
    for fpath in fpaths:
        pce = midi_fpath_to_pce(fpath)
        if pce is None:
            continue
        pces.append(pce)
    return np.mean(pces)

def compute_gps(music:muspy.Music):
    grooving_pattern = np.zeros((128,), dtype=bool)
    for track in music.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            onset = note.time
            if onset >= 128:
                continue
            grooving_pattern[onset] = True

    gps = []
    for i in range(7):
        for j in range(i+1, 8):
            grooving_pattern_i = grooving_pattern[i*16: i*16+16]
            grooving_pattern_j = grooving_pattern[j*16: j*16+16]
            xor_ij = np.logical_xor(grooving_pattern_i, grooving_pattern_j).astype(np.float32)
            sim_ij = 1.0 - np.mean(xor_ij)
            gps.append(sim_ij)
            
    
    return np.mean(gps)


def midi_fpath_to_gps(fpath):
    music = muspy.read_midi(fpath)
    music.adjust_resolution(4)
    return compute_gps(music)


def compute_gps_fpaths(fpaths):
    gpss = []
    for fpath in fpaths:
        gps = midi_fpath_to_gps(fpath)
        if gps is None:
            continue
        gpss.append(gps)
    return np.mean(gpss)


def safe_cosine_similarity(X, Y, eps=1e-12):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    X_norm = np.linalg.norm(X, keepdims=True)
    Y_norm = np.linalg.norm(Y, keepdims=True)

    X_safe = X / np.maximum(X_norm, eps)
    Y_safe = Y / np.maximum(Y_norm, eps)
    return np.dot(X_safe, Y_safe.T)


def eval_one_sample(folder, file_name):
    gt_chord = np.load(os.path.join(folder, 'chd.npy'))
    gt_hyperscore = np.load(os.path.join(folder, 'hyperscore.npy'))

    try:
        sample_chord = midi_fpath_to_chords(os.path.join(folder, file_name), os.path.join(folder, file_name[:-4]+'out'))
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
        music = muspy.read_midi(os.path.join(folder, file_name))
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

def compute_chord_ins_acc_folder(folder, file_name):
    sim_cs = []
    sim_hs = []
    for f in (os.listdir(folder)):
        sim_c,sim_h = eval_one_sample(os.path.join(folder, f), file_name)
        sim_cs.append(sim_c)
        sim_hs.append(sim_h)
    return np.mean(sim_cs), np.mean(sim_hs)
 

def compute_feature_dist(notes):
    pitch_dist = np.zeros((8, 12))
    dur_dist = np.zeros((8, 16))
    occupied = np.zeros((8, 16))
    ioi_dist = np.zeros((8, 16))
    for note in notes:
        pitch_class = note.pitch // 12
        duration = note.duration
        if duration >= 16:
            duration = 16
        onset = note.start
        if onset >= 128:
            continue
        bar = onset // 16
        pitch_dist[bar, pitch_class] += duration
        dur_dist[bar, duration-1] += 1
        occupied[bar, onset%16] = 1.0

    rows, cols = np.where(occupied == 1.0)
    for b in range(occupied.shape[0]):
        # 取出该 batch 的所有 1 的列索引
        idx = cols[rows == b]
        # 相邻位置差
        d = np.diff(idx)
        ioi_dist[b][d] += 1.0
    
    pitch_dist /= (np.linalg.norm(pitch_dist, axis=1, keepdims=True) + 1e-12)
    dur_dist /= (np.linalg.norm(dur_dist, axis=1, keepdims=True) + 1e-12)
    ioi_dist /= (np.linalg.norm(ioi_dist, axis=1, keepdims=True) + 1e-12)

    return pitch_dist, dur_dist, ioi_dist

def compute_oa(notes_gt:list[muspy.Note], notes_ours:list[muspy.Note]):
    pitch_dist_gt, dur_dist_gt, ioi_dist_gt = compute_feature_dist(notes_gt)
    pitch_dist_ours, dur_dist_ours, ioi_dist_ours = compute_feature_dist(notes_ours)   

    oa_pitch = np.mean(np.sum(np.minimum(pitch_dist_gt, pitch_dist_ours), axis=1))
    oa_dur = np.mean(np.sum(np.minimum(dur_dist_gt, dur_dist_ours), axis=1))
    oa_ioi = np.mean(np.sum(np.minimum(ioi_dist_gt, ioi_dist_ours), axis=1))
    return oa_pitch, oa_dur, oa_ioi


def compute_overlapping_metric_one_sample(folder, file_name):
    gt_music = muspy.read_midi(os.path.join(folder, "gt_wo_drum.mid"))
    gt_music.adjust_resolution(4)
    music = muspy.read_midi(os.path.join(folder, file_name))
    music.adjust_resolution(4)

    gt_tracks = defaultdict(list)
    for track in gt_music:
        if track.is_drum:
            continue
        program_category = track.program // 8
        if program_category >= 10:
            program_category = 10
        gt_tracks[program_category].extend(track.notes) 

    our_tracks = defaultdict(list)
    for track in music:
        if track.is_drum:
            continue
        program_category = track.program // 8
        if program_category >= 10:
            program_category = 10
        if program_category in gt_tracks.keys():
            our_tracks[program_category].extend(track.notes) 

    oa_p_tracks = []
    oa_d_tracks = []
    oa_ioi_tracks = []  

    for key in gt_tracks.keys():
        gt_track = gt_tracks[key]
        our_track = our_tracks[key]
        oa_p, oa_d, oa_ioi = compute_oa(gt_track, our_track)
        oa_p_tracks.append(oa_p)
        oa_d_tracks.append(oa_d)
        oa_ioi_tracks.append(oa_ioi)

    return np.mean(oa_p_tracks), np.mean(oa_d_tracks), np.mean(oa_ioi_tracks)

def compute_overlapping_metric(folder, file_name):
    OA_Ps = []
    OA_Ds = []
    OA_IOIs = []
    for f in (os.listdir(folder)):
        oa_p, oa_d, oa_ioi = compute_overlapping_metric_one_sample(os.path.join(folder, f), file_name)
        OA_Ps.append(oa_p)
        OA_Ds.append(oa_d)
        OA_IOIs.append(oa_ioi)
    return np.mean(OA_Ps), np.mean(OA_Ds), np.mean(OA_IOIs)



if __name__=="__main__":
    print("===========Ground Truth Samples==============")
    gt_fpaths = get_target_fpaths("cond_1000","gt_wo_drum.mid")
    print(f"Task:/, Model:gt, DOA={compute_doa_fpaths(gt_fpaths)}")
    print(f"Task:/, Model:gt, PCE={compute_pce_fpaths(gt_fpaths)}")
    print(f"Task:/, Model:gt, GPS={compute_gps_fpaths(gt_fpaths)}")



    print("===========Uncond Generation====================")    
    folder = "uncond_1000"
    for filename in ["ours.mid", "atc.mid", "mmt.mid"]:
        fpaths = get_target_fpaths(folder, filename)
        print(f"Task:uncond, Model:{filename[:-4]}, DOA={compute_doa_fpaths(fpaths)}")
        print(f"Task:uncond, Model:{filename[:-4]}, PCE={compute_pce_fpaths(fpaths)}")
        print(f"Task:uncond, Model:{filename[:-4]}, GPS={compute_gps_fpaths(fpaths)}")


    print("===========Conditional Generation=================")
    folder = "cond_1000"

    for filename in ["output_wo_drum.mid", "qna_good_control.mid", "qna_simple_control.mid"]:
        fpaths = get_target_fpaths(folder, filename)
        print(f"Task:cond, Model:{filename[:-4]}, DOA={compute_doa_fpaths(fpaths)}")
        print(f"Task:cond, Model:{filename[:-4]}, PCE={compute_pce_fpaths(fpaths)}")
        print(f"Task:cond, Model:{filename[:-4]}, GPS={compute_gps_fpaths(fpaths)}")
        
        chord_acc, ins_acc = compute_chord_ins_acc_folder(folder, filename)
        print(f"Task:cond, Model:{filename[:-4]}, Chord Acc={chord_acc}, Instrumentation Acc={ins_acc}")
        oa_p, oa_d, oa_ioi = compute_overlapping_metric(folder, filename)
        print(f"Task:cond, Model:{filename[:-4]}, OAP={oa_p}, OAD={oa_d}, OAIOI={oa_ioi}")
        
    

