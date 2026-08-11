"""Guitar-TECHS WAV から Basic Pitch で pitch bend MIDI を抽出する。

Basic Pitch（Spotify）でオーディオを解析し、pitch bend を含む MIDI を生成する。
この MIDI が Phase 3 の bend カーブモデルの学習データになる。

必要なパッケージ（Colab で事前インストール）:
  pip install basic-pitch

使い方:
  python scripts/extract_pitchbend_data.py \
      --audio-dir C:/Users/kake0/GitHub/研究/guitar_techs/P1_techniques/P1_techniques/audio/directinput \
      --out data/pitchbend_midi

出力: data/pitchbend_midi/
  Bendings_bend.mid
  Vibrato_bend.mid
  ...

注意:
  - Basic Pitch は CPU でも動作するが GPU があれば高速。
  - 処理時間目安: 1ファイル約 30〜90 秒（CPU）。
  - Colab での実行推奨。
"""

from __future__ import annotations

import argparse
from pathlib import Path

AUDIO_TO_TECHNIQUE = {
    "directinput_Bendings.wav": "bend",
    "directinput_Vibrato.wav": "vibrato",
    "directinput_PalmMute.wav": "palm_mute",
    "directinput_Harmonics.wav": "harmonic",
    "directinput_PinchHarmonics.wav": "pinch_harmonic",
}


def run_basic_pitch(audio_path: Path, out_dir: Path, technique: str) -> Path | None:
    """Basic Pitch で 1 ファイルを処理し、pitch bend 付き MIDI を返す。"""
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError:
        print(
            "  [ERROR] basic-pitch がインストールされていません。\n"
            "  pip install basic-pitch を実行してください。"
        )
        return None

    import numpy as np

    print(f"  処理中: {audio_path.name} → {technique}")

    model_output, midi_data, note_events = predict(
        audio_path,
        ICASSP_2022_MODEL_PATH,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=58,   # ms
        minimum_frequency=None,
        maximum_frequency=None,
        melodia_trick=True,
        midi_tempo=120,
    )

    out_path = out_dir / f"{technique}_basic_pitch.mid"
    midi_data.write(str(out_path))
    print(f"  -> 保存: {out_path} ({len(note_events)} notes)")
    return out_path


def analyze_midi_pitchbend(midi_path: Path) -> dict:
    """mido で pitch bend イベントを解析する。"""
    import mido

    mid = mido.MidiFile(str(midi_path))
    bend_events = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == "pitchwheel":
                bend_events.append(msg.pitch)

    if not bend_events:
        return {"n_bend_events": 0}

    import statistics
    return {
        "n_bend_events": len(bend_events),
        "mean": round(statistics.mean(bend_events), 1),
        "max": max(bend_events),
        "min": min(bend_events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Guitar-TECHS pitch bend 抽出")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path(
            r"C:\Users\kake0\GitHub\研究\guitar_techs"
            r"\P1_techniques\P1_techniques\audio\directinput"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "pitchbend_midi",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Audio フォルダ: {args.audio_dir}")
    print(f"出力フォルダ: {args.out}")

    results = []
    for filename, technique in AUDIO_TO_TECHNIQUE.items():
        audio_path = args.audio_dir / filename
        if not audio_path.exists():
            print(f"  [skip] {filename} not found")
            continue

        midi_path = run_basic_pitch(audio_path, args.out, technique)
        if midi_path is None:
            continue

        stats = analyze_midi_pitchbend(midi_path)
        results.append((technique, stats))
        print(f"  pitch bend 統計: {stats}")

    print("\n=== 完了 ===")
    for technique, stats in results:
        print(f"  {technique}: {stats}")


if __name__ == "__main__":
    main()
