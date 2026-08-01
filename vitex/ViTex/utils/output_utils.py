import matplotlib.patches as patches
from matplotlib.path import Path
import muspy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch


# ===== 2. 绘图设置 =====
CMAP = plt.cm.get_cmap('tab20', 11)  # 自动配色
COLORS = [CMAP(i) for i in range(11)]
INSTR_MAP = [
    'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Effect'
]


def chord_to_midi(chord, fpath):
    import pretty_midi as pm
    chord = chord.reshape(-1, 36)

    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    t = 0.0
    for beat, one_chord in enumerate(chord):
        root = int(one_chord[0:12].argmax())
        chroma = one_chord[12:24]
        bass = int(one_chord[24:].argmax())

        chroma = np.roll(chroma, -bass)
        c3 = 48
        for i, n in enumerate(chroma):
            if n == 1:
                note = pm.Note(
                    velocity=80,
                    pitch=c3 + i + bass,
                    start=t * 0.5,
                    end=(t + 1) * 0.5,
                )
                piano.notes.append(note)
        t += 1

    midi.instruments.append(piano)
    midi.write(fpath)


def pianoroll_to_midi(pianoroll: torch.Tensor, fpath, program=0):
    active_notes = torch.argwhere(pianoroll == 1)
    music = muspy.Music(resolution=4, tempos=[muspy.Tempo(
        0, 120)], time_signatures=[muspy.TimeSignature(0, 4, 4)])
    track = muspy.Track(program=program if program >=
                        0 else 0, is_drum=program < 0)
    for note in active_notes:
        onset, pitch = note

        duration = 1
        while True:
            if onset+duration >= 128:
                break
            if pianoroll[onset+duration, pitch] != 2:
                break

            duration += 1
        track.notes.append(muspy.Note(time=onset.item(),
                           pitch=pitch.item(), duration=duration, velocity=64))
    music.tracks.append(track)
    muspy.write_midi(fpath, music)


def notes_to_midi(tonal_notes, drum_notes, fpath):
    music = muspy.Music(resolution=4, tempos=[muspy.Tempo(
        0, 120)], time_signatures=[muspy.TimeSignature(0, 4, 4)])
    activate_drum_pitch = set()
    if len(drum_notes) > 0:
        drum_track = muspy.Track(program=0, is_drum=True)
        for note in drum_notes:
            onset, pitch, duration = note
            drum_track.notes.append(muspy.Note(time=int(onset),
                                               pitch=int(pitch),
                                               duration=int(duration),
                                               velocity=64))
            activate_drum_pitch.add(int(pitch))

        music.tracks.append(drum_track)
    tracks = defaultdict(list)
    if len(tonal_notes) > 0:
        for note in tonal_notes:
            onset, pitch, category, duration = note
            program_num = int(category * 8)
            tracks[program_num].append(muspy.Note(
                time=int(onset), pitch=int(pitch), duration=int(duration)))
        for program_num, notes in tracks.items():
            track = muspy.Track(program=program_num, is_drum=False)
            track.notes = notes
            music.tracks.append(track)
    muspy.write_midi(fpath, music)


def draw_wave(ax, x, y, color, is_long_note=True):
    """波浪线 / 短音时画成断开的波浪"""
    t = np.linspace(0, 1, 80)
    X = y + (t - 0.5) * 0.8
    Y = x + 0.3 * np.sin(2 * np.pi * t)

    if is_long_note:
        ax.plot(X, Y, color=color, alpha=0.6, linewidth=6.0)
    else:
        # 分段画，比如隔 10 个点断开一次
        step = 15
        for i in range(0, len(t)-step, step*2):
            ax.plot(X[i:i+step], Y[i:i+step],
                    color=color, alpha=0.6, linewidth=6.0)


def draw_round_rect(ax, x, y, color, is_long_note=True):
    """圆角矩形；短音符时在中间挖空两段"""
    # 整体矩形参数
    width, height = 0.8, 0.3
    base_rect = patches.FancyBboxPatch(
        (y - width/2, x - height/2),
        width, height,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        linewidth=0,
        facecolor=color, alpha=0.5
    )
    ax.add_patch(base_rect)

    if not is_long_note:
        # 在矩形上叠加两段白色遮挡条，制造断开的效果
        gap_w, gap_h = 0.15, height + 0.2
        offsets = [-0.15, 0.15]  # 两个空白的 x 偏移
        for off in offsets:
            gap = patches.Rectangle(
                (y + off - gap_w/2, x - gap_h/2),
                gap_w, gap_h,
                linewidth=0,
                facecolor="white", edgecolor="none",

            )
            ax.add_patch(gap)


def plot_tonal_hyperscore(hyperscore: np.ndarray, fpath):
    W, H, C = hyperscore.shape[0], hyperscore.shape[1], hyperscore.shape[2]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_aspect('equal')
    ax.set_ylim(0.0, 8.0)

    # 为不同通道生成 (dx, dy) 偏移，避免重叠
    angles = np.linspace(0, 2*np.pi, C, endpoint=False)
    radius = 0.2
    offsets = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]

    legend_handles = []
    # ===== 3. 绘制数据 =====
    for c in range(C):
        drawn = False
        for w in range(W):
            for h in range(8):
                is_melodic, is_harmonic, is_long_note = hyperscore[w, h, c]

                dx, dy = offsets[c]
                x = h + dx
                y = w + dy

                if is_melodic:  # 波浪线
                    draw_wave(ax, x, y, COLORS[c], is_long_note)
                    drawn = True

                if is_harmonic:  # 圆角矩形

                    draw_round_rect(ax, x, y, COLORS[c], is_long_note)
                    drawn = True
        if drawn:
            legend_handles.append(
                patches.Patch(color=COLORS[c], label=INSTR_MAP[c], alpha=0.5)
            )
    ax.legend(handles=legend_handles, title="Inst.",
              loc="upper right")

    plt.tight_layout()
    plt.axis("off")
    plt.savefig(fpath, dpi=300,bbox_inches="tight", pad_inches=0)
    plt.close(fig)
   


def plot_drum_hyperscore(hyperscore: np.ndarray, fpath):
    W, H = hyperscore.shape[0], hyperscore.shape[1]

    width_per_time = 0.01
    fig_width = max(6, W * 16 * width_per_time)  # 至少 6 英寸
    fig_height = 3  # 固定高度
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_aspect('equal')
    ax.set_xlim(-1.0, W)
    ax.set_ylim(-1.0, 4.0)

    # ===== 3. 绘制数据 =====

    for w in range(W):
        for h in range(3):
            has_note, is_long_note = hyperscore[w, h]
            if not has_note:
                continue
            x = float(h)
            y = float(w)

            draw_round_rect(ax, x, y, 'black', is_long_note)

    # ===== 4. 保存 & 显示 =====
    plt.tight_layout()
    plt.savefig(fpath, dpi=300)


def plot_pianoroll(tonal_notes, drum_notes, fpath):
    # 找最大时间
    max_time = 0
    for onset, pitch, category, duration in tonal_notes:
        max_time = max(max_time, onset + duration)
    for onset, pitch, duration in drum_notes:
        max_time = max(max_time, onset + duration)

    # 每个时间单位对应的宽度（英寸），可以根据需要调整
    width_per_time = 0.01
    fig_width = max(6, max_time * width_per_time)  # 至少 6 英寸
    fig_height = 6  # 固定高度
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_width, fig_height), sharex=True)

    # ===== Subplot 1: 彩色多乐器音符 =====
    for onset, pitch, category, duration in tonal_notes:
        color = COLORS[category % len(COLORS)]
        ax1.plot([onset, onset + duration],
                 [pitch, pitch], color=color, linewidth=2)

    ax1.set_ylabel("Pitch")
    ax1.set_title("Tonal Notes (Multi-instrument)")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # ===== Subplot 2: 黑色打击乐 =====
    for onset, pitch, duration in drum_notes:
        ax2.plot([onset, onset + duration], [pitch, pitch],
                 color="black", linewidth=2)

    ax2.set_ylabel("Pitch")
    ax2.set_xlabel("Time")
    ax2.set_title("Percussion Notes")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fpath, dpi=300)
    plt.close(fig)


def plot_pianoroll_from_multitrack_pianoroll(pianoroll: np.ndarray, fpath, plot_mask=False):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_ylim(0.0, 128.0)
    for w in range(128):
        for h in range(128):
            all_mask = True
            for c in range(11):
                if all_mask:
                    if pianoroll[w,h,c] != 3:
                        all_mask=False
                color = COLORS[c]
                if pianoroll[w, h, c] == 1:
                    dw = 1
                    while True:
                        if w+dw >= 128:
                            break
                        if pianoroll[w+dw, h, c] != 2:
                            break
                        dw += 1
                    ax.plot([w, w+dw], [h, h], color=color,
                            linewidth=5, alpha=0.5)
            if all_mask and plot_mask:
                ax.plot([w, w+1], [h, h], color='black',
                            linewidth=5, alpha=0.5)

    plt.tight_layout()
    plt.axis("off")
    plt.savefig(fpath, dpi=300,bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_pianoroll_from_drum_pianoroll(pianoroll: np.ndarray, fpath):
    fig, ax = plt.subplots(figsize=(12, 12))
    for w in range(128):
        for h in range(128):

            if pianoroll[w, h] > 0:
                plt.plot([w, w+1], [h, h], color='black', linewidth=5)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(fpath, dpi=300)
    plt.close(fig)


def multitrack_pianoroll_to_midi(pianoroll: np.ndarray, drum_pianoroll, fpath):
    music = muspy.Music(resolution=4, tempos=[muspy.Tempo(
        0, 120)], time_signatures=[muspy.TimeSignature(0, 4, 4)])
    tonal_tracks = defaultdict(list[muspy.Note])
    width = pianoroll.shape[0]
    for w in range(width):
        for h in range(128):
            for c in range(11):
                if pianoroll[w, h, c] == 1:
                    duration = 1
                    while True:
                        if w + duration >= width:
                            break
                        if pianoroll[w + duration, h, c] == 0 or pianoroll[w+duration,h,c] == 3:
                            break
                        duration += 1
                    note = muspy.Note(
                        time=w, pitch=h, duration=duration, velocity=100)
                    tonal_tracks[c].append(note)
    for k, v in tonal_tracks.items():
        track = muspy.Track(program=k*8, notes=v)
        music.tracks.append(track)
    if drum_pianoroll is not None:
        drum_track = muspy.Track(program=0, is_drum=True)
        for w in range(width):
            for h in range(128):
                if drum_pianoroll[w, h] == 1:
                    duration = 1
                    while True:
                        if w + duration >= width:
                            break
                        if drum_pianoroll[w + duration, h] == 0:
                            break
                        duration += 1
                    note = muspy.Note(
                        time=w, pitch=h, duration=duration, velocity=100)
                    drum_track.notes.append(note)
        music.tracks.append(drum_track)
    muspy.write_midi(fpath, music)
