import shutil
from pathlib import Path
from multiprocessing import Pool
import muspy
from tqdm import tqdm


# 1. 获取所有 MIDI 文件
def get_midi_files(root_dir):
    root_dir = Path(root_dir)
    return list(root_dir.rglob("*.mid")) + list(root_dir.rglob("*.midi"))


# 2. 处理单个 MIDI 文件（normalize + 保存）
def normalize_midi(file_path_output):
    file_path, output_dir, idx = file_path_output
    try:
        midi = muspy.read_midi(file_path, backend="pretty_midi")
        new_midi = muspy.Music(resolution=4, tempos=[muspy.Tempo(
            0, 120)], time_signatures=[muspy.TimeSignature(0, 4, 4)])
        # ===== 这里写你的 normalize 逻辑 =====
        midi.adjust_resolution(4)
        for track in midi.tracks:
            program_num = track.program // 8
            if program_num >= 10:
                program_num = 10
            new_track = muspy.Track(
                program=program_num*8, is_drum=track.is_drum)
            for note in track.notes:
                if note.duration == 0:
                    note.duration = 1
                note.velocity = 64
                new_track.notes.append(muspy.Note(
                    time=note.time, pitch=note.pitch, duration=note.duration, velocity=note.velocity))
            new_midi.tracks.append(new_track)
        # 写入新文件（按编号命名）
        out_path = output_dir / f"{idx:05d}.mid"
        new_midi.write(out_path, backend="pretty_midi")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


# 3. 主逻辑
def main(input_dir, output_dir, num_workers=8):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = get_midi_files(input_dir)

    # 组织任务列表 (文件路径, 输出目录, 序号)
    tasks = [(file, output_dir, idx) for idx, file in enumerate(midi_files)]

    # 多进程 normalize
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(normalize_midi, tasks), total=len(tasks)))


if __name__ == "__main__":
    main(r"/workspace/filtered_midi", r"/workspace/filtered_normalized_midi", num_workers=30)
