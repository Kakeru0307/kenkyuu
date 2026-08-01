from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing
from processor import MIDIProcessor


def process_midi(midi_fpath: Path, output_dir: Path):
    """单个 MIDI 文件处理"""
    try:
        processor = MIDIProcessor(midi_fpath)
        save_fpath = output_dir / midi_fpath.with_suffix(".pkl").name
        processor.save(save_fpath)
        return midi_fpath, None
    except Exception as e:
        return midi_fpath, str(e)


def batch_process_midi(input_dir: Path, output_dir: Path, max_workers: int = 4):
    """批量处理 MIDI 文件"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = list(input_dir.glob("*.mid"))

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_midi, midi, output_dir): midi
            for midi in midi_files
        }
        for future in tqdm(futures, total=len(futures)):
            midi_fpath, error = future.result()
            if error:
                print(f"Error: {midi_fpath} : {error}")
            else:
                results.append(midi_fpath)

    print(f"Successfully processed {len(results)}/{len(midi_files)} midis.")


if __name__ == "__main__":
    
    batch_process_midi(
        input_dir=Path("/workspace/filtered_normalized_midi"),
        output_dir=Path("/workspace/midi_pkl"),
        max_workers=30  # 也可以不写，自动使用 CPU 核心数-1
    )
