"""ドラム学習用 Dataset（input tonal+BPM+beat → target drum 1ch 二値）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from density_cond import make_bpm_cond_map
from makeData.constants import BEAT_TYPES
from midi_to_patch import normalize_pianoroll

N_BEAT_TYPES = len(BEAT_TYPES)


def make_beat_onehot_map(
    beat_id: int,
    *,
    height: int,
    width: int,
    n_types: int = N_BEAT_TYPES,
) -> np.ndarray:
    """beat_type を (N,H,W) の one-hot 定数マップにする。"""
    out = np.zeros((n_types, height, width), dtype=np.float32)
    idx = int(beat_id)
    if not (0 <= idx < n_types):
        raise ValueError(f"beat_id out of range: {beat_id}")
    out[idx] = 1.0
    return out


class DrumPairDataset(Dataset):
    """input/*_tonal.npy + *_cond.npy + *_beat.npy → target/*_drum.npy。

    *_beat.npy が無い旧サンプルはスキップする（in=24 再学習前提）。
    """

    def __init__(self, input_dir: str | Path, target_dir: str | Path):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        candidates = sorted(self.input_dir.rglob("*_tonal.npy"))
        self.files: list[Path] = []
        for path in candidates:
            beat_path = path.with_name(path.name.replace("_tonal.npy", "_beat.npy"))
            if beat_path.is_file():
                self.files.append(path)
        if not self.files:
            raise FileNotFoundError(
                f"*_beat.npy 付きパッチが見つかりません: {self.input_dir} "
                "(旧 energy データはスキップ。generate_drum_pairs で再生成してください)"
            )

    def __len__(self) -> int:
        return len(self.files)

    def _load_cond(self, input_path: Path, height: int, width: int) -> np.ndarray:
        cond_path = input_path.with_name(
            input_path.name.replace("_tonal.npy", "_cond.npy")
        )
        if cond_path.is_file():
            raw = np.load(cond_path)
            arr = np.asarray(raw)
            if arr.ndim >= 2 and arr.shape[-2:] == (height, width):
                if arr.ndim == 2:
                    return arr.reshape(1, height, width).astype(np.float32, copy=False)
                return arr.astype(np.float32, copy=False)[:1]
            unit = float(arr.reshape(-1)[0])
            return np.full((1, height, width), np.float32(unit), dtype=np.float32)
        return make_bpm_cond_map(105.0, height=height, width=width)

    def _load_beat(self, input_path: Path, height: int, width: int) -> np.ndarray:
        beat_path = input_path.with_name(
            input_path.name.replace("_tonal.npy", "_beat.npy")
        )
        beat_id = int(np.asarray(np.load(beat_path)).reshape(-1)[0])
        return make_beat_onehot_map(beat_id, height=height, width=width)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_path = self.files[idx]
        rel = input_path.relative_to(self.input_dir)
        target_path = self.target_dir / rel.with_name(
            rel.name.replace("_tonal.npy", "_drum.npy")
        )
        if not target_path.is_file():
            raise FileNotFoundError(f"drum target がありません: {target_path}")

        input_arr = normalize_pianoroll(np.load(input_path))
        drum = np.asarray(np.load(target_path), dtype=np.float32)
        if drum.ndim == 2:
            drum = drum[np.newaxis, ...]
        drum = (drum > 0).astype(np.float32)

        h, w = int(input_arr.shape[1]), int(input_arr.shape[2])
        cond = self._load_cond(input_path, h, w)
        beat = self._load_beat(input_path, h, w)
        model_in = np.concatenate(
            [input_arr.astype(np.float32), cond, beat],
            axis=0,
        )
        return (
            torch.from_numpy(model_in),
            torch.from_numpy(drum),
        )


def get_drum_dataloader(
    input_dir: Path,
    target_dir: Path,
    *,
    batch_size: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    ds = DrumPairDataset(input_dir, target_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
