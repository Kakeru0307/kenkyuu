import muspy

import os
import shutil
from pathlib import Path
from multiprocessing import Pool
import muspy  # 或者用 mido、pretty_midi
from tqdm import tqdm


# ======================
# 1. 获取所有MIDI文件
# ======================
def get_midi_files(root_dir):
    root_dir = Path(root_dir)
    return list(root_dir.rglob("*.mid")) + list(root_dir.rglob("*.midi"))


# ======================
# 2. 处理单个文件（读取 + 过滤）
# ======================
def process_midi(file_path):
    """
    读取和过滤MIDI文件。
    返回 (文件路径, 是否保留)
    """
    try:
        midi = muspy.read_midi(file_path)

        # filter out non 4/4
        for time_signature in midi.time_signatures:
            if time_signature.denominator != 4:
                return file_path, False
            if time_signature.numerator != 4:
                return file_path, False

        if len(midi.time_signatures) > 1:
            return file_path, False

        # filter out non bpm 110-130
        for tempo in midi.tempos:
            if tempo.qpm >= 130:
                return file_path, False
            if tempo.qpm <= 110:
                return file_path, False

        if len(midi.tempos) > 1:
            return file_path, False

        # filter out key change
        if len(midi.key_signatures) > 1:
            return file_path, False

        # filter out less than 5 tracks
        track_num = 0
        for track in midi.tracks:
            if len(track.notes) > 0:
                track_num += 1
        if track_num <= 5:
            return file_path, False

        unique_instrument = set()
        for track in midi.tracks:
            if track.is_drum == False:
                program_category = track.program // 8
                if program_category >= 10:
                    program_category = 10
                unique_instrument.add(program_category)
        if len(unique_instrument) < 3:
            return file_path, False

        total_notes_num = 0
        for track in midi.tracks:
            total_notes_num += len(track.notes)
        if total_notes_num <= 50:
            return file_path, False

        midi.adjust_resolution(4, rounding="floor")
        end_time = midi.get_end_time()
        if end_time <= 16*40:
            return file_path, False

        has_piano = False
        has_guitar = False
        has_bass = False
        has_drum = False
        for track in midi.tracks:
            if track.is_drum == True:
                has_drum = True
                continue
            if track.program // 8 == 0:
                has_piano = True
                continue
            if track.program // 8 == 3:
                has_guitar = True
                continue
            if track.program // 8 == 4:
                has_bass = True
                continue
        if has_drum == False:
            return file_path, False
        if has_piano == False and has_guitar == False and has_bass == False:
            return file_path, False

        zero_note_num = 0
        for track in midi.tracks:
            if track.is_drum:
                continue
            for note in track.notes:
                if note.duration == 0:
                    zero_note_num += 1
        if zero_note_num > 30:
            return file_path, False

        return file_path, True

    except Exception:
        return file_path, False


# ======================
# 3. 主逻辑
# ======================
def main(input_dir, output_dir, num_workers=8):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = get_midi_files(input_dir)

    # 多进程处理
    with Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap(process_midi, midi_files), total=len(midi_files)))

    # 连续编号保存
    counter = 0
    for file_path, keep in results:
        if keep:
            new_path = output_dir / f"{counter:06d}.mid"
            shutil.copy(file_path, new_path)
            counter += 1


if __name__ == "__main__":
    main(r"/workspace/midi",
         r"/workspace/filtered_midi", num_workers=30)
