"""パッチ .npy をピアノロール画像（PNG）として保存・表示する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent

# --- ここを書き換えて実行しても OK ---
NPY_PATH = SCRIPT_DIR / "data" / "patches" / "test1" / "bar0000_tonal.npy"
OUTPUT_PATH: Path | None = None  # None なら .npy と同じ場所に .png を保存
SHOW_IMAGE = False  # True なら保存後に画像ビューアで開く（環境依存）

VALUE_TO_BRIGHTNESS = {0: 0, 1: 170, 2: 255}

# tonal のチャンネル 0,1,2 を R,G,B に対応（ViTex / midiToPic と同じイメージ）
TONAL_RGB_CHANNELS = (0, 1, 2)


def _detect_layout(data: np.ndarray) -> tuple[str, np.ndarray]:
    """CHW 形式に正規化し、tonal / drum を判定する。"""
    if data.ndim != 3:
        raise ValueError(f"3 次元配列を想定していますが shape={data.shape} です")

    if data.shape[0] in (1, 11):
        chw = data
    elif data.shape[-1] in (1, 11):
        chw = data.transpose(2, 0, 1)
    else:
        raise ValueError(f"想定外の shape です: {data.shape}")

    kind = "drum" if chw.shape[0] == 1 else "tonal"
    return kind, chw.astype(np.int32)


def _chw_to_piano_roll(chw: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
    """(C, time, pitch) -> (pitch, time, 3) の RGB 画像。高音が上。"""
    time_len, pitch_len = chw.shape[1], chw.shape[2]
    canvas = np.zeros((pitch_len, time_len, 3), dtype=np.uint8)

    for t in range(time_len):
        for pitch in range(pitch_len):
            y = pitch_len - 1 - pitch
            for i, ch in enumerate(channels):
                value = int(chw[ch, t, pitch])
                canvas[y, t, i] = max(canvas[y, t, i], VALUE_TO_BRIGHTNESS.get(value, 0))

    return canvas


def _chw_to_grayscale(chw: np.ndarray) -> np.ndarray:
    """(1, time, pitch) -> (pitch, time) のグレースケール。高音が上。"""
    channel = chw[0]
    pitch_len, time_len = channel.shape[1], channel.shape[0]
    img = np.zeros((pitch_len, time_len), dtype=np.uint8)

    for t in range(time_len):
        for pitch in range(pitch_len):
            y = pitch_len - 1 - pitch
            img[y, t] = VALUE_TO_BRIGHTNESS.get(int(channel[t, pitch]), 0)

    return img


def _chw_to_composite(chw: np.ndarray) -> np.ndarray:
    """全チャンネルの最大値を白一色のピアノロールにする。"""
    merged = chw.max(axis=0)
    pitch_len, time_len = merged.shape[1], merged.shape[0]
    img = np.zeros((pitch_len, time_len, 3), dtype=np.uint8)

    for t in range(time_len):
        for pitch in range(pitch_len):
            y = pitch_len - 1 - pitch
            brightness = VALUE_TO_BRIGHTNESS.get(int(merged[t, pitch]), 0)
            img[y, t] = (brightness, brightness, brightness)

    return img


def npy_to_image(data: np.ndarray, *, mode: str = "auto") -> Image.Image:
    kind, chw = _detect_layout(data)

    if mode == "auto":
        mode = "rgb" if kind == "tonal" else "gray"

    if mode == "rgb":
        if kind != "tonal":
            raise ValueError("rgb モードは tonal.npy 用です")
        array = _chw_to_piano_roll(chw, TONAL_RGB_CHANNELS)
        return Image.fromarray(array, mode="RGB")

    if mode == "composite":
        array = _chw_to_composite(chw)
        return Image.fromarray(array, mode="RGB")

    if mode == "gray":
        array = _chw_to_grayscale(chw)
        return Image.fromarray(array, mode="L")

    raise ValueError(f"未知の mode: {mode}")


def default_output_path(npy_path: Path, mode: str) -> Path:
    suffix = "" if mode in ("auto", "rgb", "gray") else f"_{mode}"
    return npy_path.with_name(f"{npy_path.stem}{suffix}.png")


def visualize_npy(
    npy_path: str | Path,
    output_path: str | Path | None = None,
    *,
    mode: str = "auto",
    show: bool = False,
) -> Path:
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {npy_path}")

    data = np.load(npy_path)
    image = npy_to_image(data, mode=mode)

    if output_path is None:
        resolved_mode = "rgb" if "_tonal" in npy_path.stem else "gray"
        if mode != "auto":
            resolved_mode = mode
        out = default_output_path(npy_path, resolved_mode)
    else:
        out = Path(output_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)

    kind, chw = _detect_layout(data)
    print(f"入力: {npy_path}")
    print(f"種類: {kind}, shape: {tuple(chw.shape)}")
    print(f"保存: {out}")

    if show:
        image.show()

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="パッチ .npy を PNG 画像に変換")
    parser.add_argument(
        "npy_path",
        nargs="?",
        default=str(NPY_PATH),
        help=".npy ファイルのパス（省略時はスクリプト先頭の NPY_PATH）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="出力 PNG パス（省略時は .npy と同じ場所）",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "rgb", "composite", "gray"),
        default="auto",
        help="auto: tonal=rgb, drum=gray / composite: 全チャンネル合成",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=SHOW_IMAGE,
        help="保存後に画像を表示",
    )
    args = parser.parse_args()

    visualize_npy(
        args.npy_path,
        args.output,
        mode=args.mode,
        show=args.show,
    )


if __name__ == "__main__":
    main()
