import numpy as np
import os
from chord_extractor import extract_chords_from_midi_file
import csv
import mir_eval as mir_eval
import torch

def get_chord_matrix(fpath):
    """
    chord matrix [M * 14], each line represent the chord of a beat
    same format as mir_eval.chord.encode():
        root_number(1), semitone_bitmap(12), bass_number(1)
    inputs are generated from junyan's algorithm
    """
    ONE_BEAT = 0.5
    file = csv.reader(open(fpath), delimiter="\t")
    beat_cnt = 0
    chords = []
    for line in file:
        start = float(line[0]) / ONE_BEAT
        end = float(line[1]) / ONE_BEAT
        chord = line[2]

        while beat_cnt < int(round(end)):
            beat_cnt += 1
            # see https://craffel.github.io/mir_eval/#mir_eval.chord.encode
            chd_enc = mir_eval.chord.encode(chord)
            root = chd_enc[0]
            # make chroma and bass absolute
            chroma_bitmap = chd_enc[1]
            chroma_bitmap = np.roll(chroma_bitmap, root)
            bass = (chd_enc[2] + root) % 12
            chord_line = [root]
            for _ in chroma_bitmap:
                chord_line.append(_)
            chord_line.append(bass)
            chords.append(chord_line)
    return chords


def chd_to_onehot(chd):
    n_step = chd.shape[0]
    onehot_chd = np.zeros((n_step, 36), dtype=bool)
    onehot_chd[np.arange(n_step), chd[:, 0]] = True
    onehot_chd[:, 12:24] = chd[:, 1:13]
    onehot_chd[np.arange(n_step), 24 + chd[:, -1]] = True
    onehot_chd = onehot_chd.reshape(-1, 3, 12)
    return onehot_chd


def midi_fpath_to_chords(midi_fpath, chord_fpath):
    extract_chords_from_midi_file(midi_fpath, chord_fpath)
    chord = np.array(get_chord_matrix(chord_fpath))
    np_chords = chd_to_onehot(chord)
    os.remove(chord_fpath)
    return np_chords

def chord_to_vector(chord):
    """
    将和弦代号转换为根音、和弦色彩和低音的向量表示，支持转位和7和弦。

    参数:
    chord (str): 和弦代号，如 'C', 'G7', 'Em/B', 'D7/F#' 等

    返回:
    numpy.ndarray: 形状为 (3, 12) 的数组，分别表示根音、和弦色彩和低音
    """
    note_to_num = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
                   'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}

    # 处理转位（低音）符号，如 Em/B
    if '/' in chord:
        chord, bass = chord.split('/')
    else:
        bass = None

    # 解析根音和和弦质量
    root = chord[0]
    if len(chord) > 1 and chord[1] in ['#', 'b']:
        root += chord[1]
        quality = chord[2:]
    else:
        quality = chord[1:]

    root_num = note_to_num[root]

    # 根音向量
    root_vector = np.zeros(12)
    root_vector[root_num] = 1

    # 和弦色彩向量，默认大三和弦
    chroma_vector = np.zeros(12)
    chroma_vector[root_num] = 1

    # 判断和弦类型，支持小三和弦、7和弦
    if quality in ['m', 'min']:
        # 小三和弦：根音 + 小三度 + 完全五度
        chroma_vector[(root_num + 3) % 12] = 1
        chroma_vector[(root_num + 7) % 12] = 1
    elif quality in ['7']:
        # 属七和弦：根音 + 大三度 + 完全五度 + 小七度
        chroma_vector[(root_num + 4) % 12] = 1
        chroma_vector[(root_num + 7) % 12] = 1
        chroma_vector[(root_num + 10) % 12] = 1
    elif quality in ['m7', 'min7']:
        # 小七和弦：根音 + 小三度 + 完全五度 + 小七度
        chroma_vector[(root_num + 3) % 12] = 1
        chroma_vector[(root_num + 7) % 12] = 1
        chroma_vector[(root_num + 10) % 12] = 1
    elif quality in ['maj7', 'M7']:
        # 大七和弦：根音 + 大三度 + 完全五度 + 大七度
        chroma_vector[(root_num + 4) % 12] = 1
        chroma_vector[(root_num + 7) % 12] = 1
        chroma_vector[(root_num + 11) % 12] = 1
    else:
        # 默认大三和弦
        chroma_vector[(root_num + 4) % 12] = 1
        chroma_vector[(root_num + 7) % 12] = 1

    # 低音向量，支持转位低音
    bass_vector = np.zeros(12)
    if bass:
        # 低音可能带升降号
        if len(bass) > 1 and bass[1] in ['#', 'b']:
            bass_note = bass[:2]
        else:
            bass_note = bass[0]
        bass_num = note_to_num[bass_note]
        bass_vector[bass_num] = 1
    else:
        bass_vector = root_vector.copy()

    return np.vstack([root_vector, chroma_vector, bass_vector])


def chords_to_sequence(chord_list, beats_per_bar=4, bars=8):
    """
    

    返回:
    numpy.ndarray: 形状为 (C, W, H) 的和弦向量序列
    """
    total_beats = beats_per_bar * bars
    if len(chord_list) != total_beats:
        raise ValueError(f"chord_list长度应为 {total_beats}，但得到 {len(chord_list)}")

    sequence = np.array([chord_to_vector(chord) for chord in chord_list])
    sequence = torch.from_numpy(sequence).to(torch.float32)
    return sequence.permute(1, 0, 2)
