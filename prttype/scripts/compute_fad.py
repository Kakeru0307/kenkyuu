"""FAD（Fréchet Audio Distance）を計算する。

使い方（リポジトリルートから）:
  .\\venv\\Scripts\\python.exe prttype\\scripts\\compute_fad.py <参照WAVディレクトリ> <生成WAVディレクトリ>

埋め込みは clap-laion-music 固定。1曲点数は出さず、フォルダ同士の距離を1つ出す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_MODEL = "clap-laion-music"


def _count_wavs(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


def _patch_torchaudio_soundfile() -> None:
    """torchaudio 2.11+ は load が torchcodec 固定。Windows では FFmpeg 無しだと死ぬので soundfile に差し替える。"""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    def load(uri, *args, **kwargs):  # noqa: ARG001
        data, sr = sf.read(str(uri), always_2d=True, dtype="float32")
        return torch.from_numpy(np.ascontiguousarray(data.T)), int(sr)

    def save(uri, src, sample_rate, **kwargs):  # noqa: ARG001
        arr = src.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T
        sf.write(str(uri), arr, int(sample_rate), subtype="PCM_16")

    torchaudio.load = load  # type: ignore[assignment]
    torchaudio.save = save  # type: ignore[assignment]


def _cache_embeddings_sequential(directory: Path, model) -> None:
    """fadtk の Pool(spawn) を避けて順次埋め込む（monkeypatch が子に渡らない対策）。"""
    from fadtk.fad import FrechetAudioDistance
    from fadtk.utils import get_cache_embedding_path

    wavs = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    pending = [p for p in wavs if not get_cache_embedding_path(model.name, p).exists()]
    if not pending:
        print(f"[fad] embeddings cached: {directory} ({len(wavs)} files)")
        return
    print(f"[fad] embedding {len(pending)} / {len(wavs)} files in {directory}")
    fad = FrechetAudioDistance(model, audio_load_worker=1, load_model=True)
    for i, path in enumerate(pending, 1):
        fad.cache_embedding_file(path)
        if i % 10 == 0 or i == len(pending):
            print(f"[fad]  {i}/{len(pending)}")


def compute_fad(ref_dir: Path, gen_dir: Path, *, model_name: str = DEFAULT_MODEL) -> float:
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"参照ディレクトリがありません: {ref_dir}")
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"生成ディレクトリがありません: {gen_dir}")

    n_ref = _count_wavs(ref_dir)
    n_gen = _count_wavs(gen_dir)
    if n_ref < 1:
        raise SystemExit(f"参照に WAV がありません: {ref_dir}")
    if n_gen < 1:
        raise SystemExit(f"生成に WAV がありません: {gen_dir}")

    _patch_torchaudio_soundfile()

    try:
        from fadtk.fad import FrechetAudioDistance
        from fadtk.model_loader import CLAPLaionModel
    except ImportError as exc:
        raise SystemExit(
            "fadtk がありません。venv で: python -m pip install fadtk"
        ) from exc

    if model_name == "clap-laion-music":
        model = CLAPLaionModel("music")
    elif model_name == "clap-laion-audio":
        model = CLAPLaionModel("audio")
    else:
        raise SystemExit(
            f"未知のモデル: {model_name}\n"
            "候補: clap-laion-music, clap-laion-audio"
        )

    _cache_embeddings_sequential(ref_dir.resolve(), model)
    _cache_embeddings_sequential(gen_dir.resolve(), model)

    fad = FrechetAudioDistance(model, audio_load_worker=1, load_model=False)
    score = float(fad.score(str(ref_dir.resolve()), str(gen_dir.resolve())))
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="FAD (CLAP music) between two WAV folders")
    parser.add_argument("ref_dir", type=Path, help="参照 WAV ディレクトリ")
    parser.add_argument("gen_dir", type=Path, help="生成 WAV ディレクトリ")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"fadtk モデル名（既定: {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="結果を書く JSON パス（省略時は stdout のみ）",
    )
    args = parser.parse_args()

    score = compute_fad(args.ref_dir, args.gen_dir, model_name=args.model)
    result = {
        "fad": score,
        "model": args.model,
        "ref_dir": str(args.ref_dir.resolve()),
        "gen_dir": str(args.gen_dir.resolve()),
        "n_ref": _count_wavs(args.ref_dir),
        "n_gen": _count_wavs(args.gen_dir),
    }
    line = f"FAD={score:.6f}  model={args.model}  n_ref={result['n_ref']}  n_gen={result['n_gen']}"
    print(line)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
