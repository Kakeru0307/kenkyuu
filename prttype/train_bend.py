"""BendModel 学習スクリプト（Phase 3）。

Basic Pitch で抽出した pitch bend MIDI から
  input:  (1, 128, 128) pianoroll + technique_map
  target: (128,) pitch bend カーブ（時系列）
を学習する。

【前提条件】
  scripts/extract_pitchbend_data.py を Colab で実行し、
  data/pitchbend_midi/ に pitch bend MIDI が存在すること。

Phase 3 の実装前に実行すると NotImplementedError が発生する。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="BendModel 学習 (Phase 3)")
    parser.add_argument(
        "--bend-midi-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "pitchbend_midi",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints"
        / "bend_model"
        / "bend_model_last.pth",
    )
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if not args.bend_midi_dir.exists() or not list(args.bend_midi_dir.glob("*.mid")):
        raise FileNotFoundError(
            f"pitch bend MIDI が見つかりません: {args.bend_midi_dir}\n"
            "先に scripts/extract_pitchbend_data.py を Colab で実行してください。"
        )

    raise NotImplementedError(
        "BendModel は Phase 3 で実装予定。\n"
        "Basic Pitch による pitch bend データ抽出後に実装します。"
    )


if __name__ == "__main__":
    main()
