"""【実験用】ふわっとした言葉から MIDI を生成するラッパー。

本実装ではない。学習済み U-Net の本線入口は `generate_backing.py`。

ここでの progression / bpm / key 決定は簡易ルール（既定は感情→カタログ、
`--use-manifest` 時のみ合成データの台帳からサンプリング）であり、
ネットワークが学習データから推論しているわけではない。

流れ:
  テキスト
    → WRIME 等で感情
    → 簡易ルールで progression / bpm / key
    → generate_song / generate_backing（ここからが本実装の U-Net）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompt_to_params import prompt_to_params

SCRIPT_DIR = Path(__file__).resolve().parent
BACKING_CKPT = SCRIPT_DIR / "checkpoints" / "backing" / "unet_last.pt"
LEAD_CKPT = SCRIPT_DIR / "checkpoints" / "lead" / "unet_last.pt"


def generate_from_prompt(
    prompt: str,
    *,
    backend: str = "wrime",
    seed: int | None = None,
    output: Path | None = None,
    with_lead: bool = True,
    with_backing: bool = True,
    show_params: bool = True,
    fallback_heuristic: bool = True,
    eval_muspy: bool = True,
    use_manifest: bool = False,
) -> Path:
    from generate_song import generate_song
    from inference import MIDI_DIR

    print(
        "[実験入口] パラメータ選択は本実装ではない。"
        "本実装は generate_backing.py（U-Net）。"
        f" selection={'manifest' if use_manifest else 'catalog'}"
    )

    params = prompt_to_params(
        prompt,
        backend=backend,  # type: ignore[arg-type]
        seed=seed,
        fallback_heuristic=fallback_heuristic,
        use_manifest=use_manifest,
    )

    if show_params:
        print("=== 内部パラメータ（簡易ルール・学習推論ではない）===")
        print(json.dumps(params.to_dict(), ensure_ascii=False, indent=2))

    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    if output is None:
        safe = "".join(c if c.isalnum() else "_" for c in prompt)[:24] or "prompt"
        output = MIDI_DIR / f"from_prompt_{safe}_bpm{int(params.bpm):03d}.mid"

    out = generate_song(
        progression=params.progression,
        key=params.key,
        bars=params.bars,
        bpm=params.bpm,
        bars_per_chord=params.bars_per_chord,
        backing_ckpt=BACKING_CKPT,
        lead_ckpt=LEAD_CKPT,
        output=output,
        with_lead=with_lead,
        with_backing=with_backing,
    )

    if eval_muspy:
        try:
            from evaluate_midi import evaluate_midi, format_report

            metrics = evaluate_midi(out, key=params.key, mode=params.mode or None)
            print(format_report(metrics))
        except Exception as exc:  # noqa: BLE001 — 評価失敗で生成結果は残す
            print(f"[evaluate_midi] skipped: {exc}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "【実験用】ふわっと言葉から MIDI を生成。"
            "本実装は generate_backing.py。"
            "ここでの進行/BPM選択は簡易ルール（既定はカタログ、--use-manifest で台帳）"
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="生成のきっかけになる言葉",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_opt",
        default=None,
        help="prompt の別名",
    )
    parser.add_argument(
        "--backend",
        choices=("wrime", "heuristic"),
        default="wrime",
        help="感情解析バックエンド（既定: wrime）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="進行・キー選択の乱数シード",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-lead", action="store_true")
    parser.add_argument("--no-backing", action="store_true")
    parser.add_argument(
        "--params-only",
        action="store_true",
        help="MIDI を生成せず、内部パラメータだけ表示",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="WRIME 失敗時に heuristic へ落とさない",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="生成後の MusPy 客観評価をスキップ",
    )
    parser.add_argument(
        "--use-manifest",
        action="store_true",
        help="合成データ manifest から進行/BPM/キーをサンプリング（実験用・既定OFF）",
    )
    args = parser.parse_args()

    prompt = args.prompt_opt or args.prompt
    if not prompt:
        parser.error("prompt を指定してください")

    if args.params_only:
        params = prompt_to_params(
            prompt,
            backend=args.backend,
            seed=args.seed,
            fallback_heuristic=not args.no_fallback,
            use_manifest=args.use_manifest,
        )
        print(json.dumps(params.to_dict(), ensure_ascii=False, indent=2))
        return

    generate_from_prompt(
        prompt,
        backend=args.backend,
        seed=args.seed,
        output=args.output,
        with_lead=not args.no_lead,
        with_backing=not args.no_backing,
        fallback_heuristic=not args.no_fallback,
        eval_muspy=not args.no_eval,
        use_manifest=args.use_manifest,
    )


if __name__ == "__main__":
    main()
