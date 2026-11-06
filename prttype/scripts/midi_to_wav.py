"""MIDI → WAV（FluidSynth + SoundFont）。

使い方:
  python scripts/midi_to_wav.py path/to/file.mid
  python scripts/midi_to_wav.py path/to/dir --out-dir wav_out

環境変数:
  SOUNDFONT  … .sf2 のパス（省略時は prttype/assets/soundfonts/default.sf2）
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SF2_CANDIDATES = (
    Path(os.environ["SOUNDFONT"]) if os.environ.get("SOUNDFONT") else None,
    ROOT / "assets" / "soundfonts" / "default.sf2",
    # Windows で非 ASCII パスだと fluidsynth が開けないことがある
    Path.home() / "soundfonts" / "default.sf2",
)


def find_fluidsynth() -> str:
    exe = shutil.which("fluidsynth")
    if exe:
        return exe
    raise FileNotFoundError(
        "fluidsynth が見つかりません。scoop install extras/fluidsynth などで入れてください。"
    )


def resolve_soundfont(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for p in DEFAULT_SF2_CANDIDATES:
        if p is not None:
            candidates.append(p)
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "SoundFont (.sf2) がありません。次のいずれかに置いてください:\n"
        f"  - {ROOT / 'assets' / 'soundfonts' / 'default.sf2'}\n"
        f"  - {Path.home() / 'soundfonts' / 'default.sf2'}\n"
        "または環境変数 SOUNDFONT を設定。"
    )


def midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    *,
    soundfont: Path,
    sample_rate: int = 44100,
) -> Path:
    fs = find_fluidsynth()
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    # FluidSynth 2.x: 出力オプションは MIDI より前
    cmd = [
        fs,
        "-ni",
        "-F",
        str(wav_path.resolve()),
        "-r",
        str(sample_rate),
        str(soundfont.resolve()),
        str(midi_path.resolve()),
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return wav_path


def main() -> None:
    parser = argparse.ArgumentParser(description="MIDI → WAV (FluidSynth)")
    parser.add_argument("input", type=Path, help="MIDI ファイルまたはディレクトリ")
    parser.add_argument("--out-dir", type=Path, default=None, help="出力ディレクトリ")
    parser.add_argument("--soundfont", type=Path, default=None)
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    sf2 = resolve_soundfont(args.soundfont)
    inp = args.input
    if inp.is_file():
        midis = [inp]
    elif inp.is_dir():
        midis = sorted(inp.rglob("*.mid")) + sorted(inp.rglob("*.midi"))
    else:
        raise FileNotFoundError(inp)
    if not midis:
        raise SystemExit(f"MIDI がありません: {inp}")

    out_dir = args.out_dir
    for mid in midis:
        if out_dir is None:
            wav = mid.with_suffix(".wav")
        else:
            wav = out_dir / (mid.stem + ".wav")
        midi_to_wav(mid, wav, soundfont=sf2, sample_rate=args.sample_rate)
        print(f"wrote {wav}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
