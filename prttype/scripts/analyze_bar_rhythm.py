"""バッキング MIDI / target npy の小節リズム統計を出力する評価スクリプト。

使用例（prttype/ ディレクトリで実行）:

  # 生成 MIDI を評価
  python scripts/analyze_bar_rhythm.py --midi midi/before.mid midi/after.mid

  # target pairs を 100 件サンプルして評価
  python scripts/analyze_bar_rhythm.py --npy-dir data/pairs/synthetic/target --n-samples 100

  # pairs モード（input も読んで precision も算出）
  python scripts/analyze_bar_rhythm.py --pairs-dir data/pairs/synthetic --n-samples 100

出力メトリクス（すべてギターカテゴリ対象）:
  onsets_per_bar         : 1小節あたりの onset **タイミング数**（pitch方向はORで潰す。
                            同時ストローク=1カウント。「何回弾いたか」に対応）
  bar_end_activity       : 小節の tick 12-15 に onset がある小節の割合
  bar_end_pattern_entropy: 小節末4tick(tick12-15)の onset パターンのエントロピー。
                            **全パッチ・全小節をプールしてから 1 回だけ計算**（パッチ単位で
                            計算して平均すると n=8/パッチしかなく最大値が飽和するため）。
                            0.0（常に同じ終わり方）〜 4.0bit（16パターンに均一分布）。
                            正規化値（/4.0）も併記
  same_pitch_retrigger   : 同一**音高のセル単位**で前tickも onset だった率（バグ的リトリガー検出）。
                            onsets_per_bar とは集計単位が異なる（pitch別、タイミング潰しなし）点に注意
  onset_precision        : [pairs モードのみ] target onset のうち入力骨格のコードトーン上にある割合
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PRTTYPE_DIR = SCRIPT_DIR.parent
if str(PRTTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PRTTYPE_DIR))

from midi_to_patch import TICKS_PER_BAR, PATCH_BARS, midi_to_patches  # noqa: E402
from program_utils import GUITAR_PROGRAM  # noqa: E402

GUITAR_CATEGORY = GUITAR_PROGRAM // 8
PATCH_TICKS = PATCH_BARS * TICKS_PER_BAR  # 128
BAR_END_WINDOW = 4  # tick 12-15 の 4tick


# ---------------------------------------------------------------------------
# 共通コア計算
# ---------------------------------------------------------------------------

def _guitar_channel(tonal_chw: np.ndarray) -> np.ndarray:
    """(11,128,128) から guitar カテゴリの (128,128) time×pitch を返す。"""
    return tonal_chw[GUITAR_CATEGORY]


def bar_metrics(ch: np.ndarray) -> dict:
    """(128,128) time×pitch 配列から小節リズム統計を計算する。

    値は uint8 の 0/1/2（onset=1, sustain=2）を想定。
    bar_end_patterns はこのパッチ内で観測された生パターン（1パッチ=8個）を返すのみで、
    エントロピーはここでは計算しない（全パッチをプールした後に aggregate_metrics で
    1 回だけ計算する。パッチ単位で計算すると n=8 しかなくエントロピーが飽和するため）。
    """
    assert ch.shape[0] == PATCH_TICKS, f"time 次元が {PATCH_TICKS} を期待: {ch.shape}"
    onset_map = (ch == 1)  # (128, 128) bool

    onsets_per_bar: list[int] = []
    bar_end_active: list[bool] = []
    bar_end_patterns: list[int] = []

    for b in range(PATCH_BARS):
        t0 = b * TICKS_PER_BAR
        t1 = t0 + TICKS_PER_BAR
        bar_onset = onset_map[t0:t1, :]       # (16, 128)

        # onsets per bar: どの pitch でも onset があれば 1tick カウント（同時ストローク=1）
        onsets_per_bar.append(int(bar_onset.any(axis=1).sum()))

        # bar end activity: tick 12-15 の 4tick に onset があるか
        end_slice = bar_onset[TICKS_PER_BAR - BAR_END_WINDOW:, :]  # (4, 128)
        bar_end_active.append(bool(end_slice.any()))

        # bar end pattern: 4bit (tick12,13,14,15 それぞれに onset があれば 1)
        pattern_bits = end_slice.any(axis=1)  # (4,) bool
        pat_int = int(sum(int(b) << i for i, b in enumerate(pattern_bits)))
        bar_end_patterns.append(pat_int)

    # same-pitch retrigger rate（音高セル単位。onsets_per_bar とは集計単位が異なる）
    total_onsets = int(onset_map.sum())
    retriggers = 0
    if total_onsets > 0:
        # onset[t, p] == True かつ onset[t-1, p] == True → retrigger
        retriggers = int((onset_map[1:, :] & onset_map[:-1, :]).sum())

    return {
        "onsets_per_bar": onsets_per_bar,
        "bar_end_active_frac": float(sum(bar_end_active)) / PATCH_BARS,
        "bar_end_patterns": bar_end_patterns,
        "same_pitch_retrigger_rate": retriggers / total_onsets if total_onsets > 0 else 0.0,
        "total_onsets": total_onsets,
    }


MAX_BAR_END_ENTROPY_BITS = 4.0  # 2^4 = 16 パターン


def pooled_entropy(patterns: list[int]) -> tuple[float, float]:
    """全パッチ・全小節をプールしたパターン列から (bit数, 正規化値0-1) を計算する。"""
    if not patterns:
        return float("nan"), float("nan")
    cnt = Counter(patterns)
    total = len(patterns)
    bits = -sum((v / total) * math.log2(v / total) for v in cnt.values() if v > 0)
    return bits, bits / MAX_BAR_END_ENTROPY_BITS


def onset_precision_vs_skeleton(
    target_chw: np.ndarray,
    input_chw: np.ndarray,
) -> float:
    """target の onset のうち、入力骨格(downbeat_chord)のコードトーン上にある割合。

    各小節のコードトーン = その小節で input に onset があるすべての pitch。
    """
    tgt_ch = _guitar_channel(target_chw)
    inp_ch = _guitar_channel(input_chw)

    total_onsets = 0
    on_chord = 0

    for b in range(PATCH_BARS):
        t0 = b * TICKS_PER_BAR
        t1 = t0 + TICKS_PER_BAR
        # 骨格のコードトーン(downbeat_chord は小節頭のみ onset)
        chord_pitches = set(int(p) for p in np.where(inp_ch[t0:t1, :] == 1)[1])

        bar_target = (tgt_ch[t0:t1, :] == 1)
        target_onsets = np.argwhere(bar_target)  # [(tick, pitch), ...]
        for _, pitch in target_onsets:
            total_onsets += 1
            if int(pitch) in chord_pitches:
                on_chord += 1

    return on_chord / total_onsets if total_onsets > 0 else float("nan")


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------

def load_from_midi(midi_paths: list[Path], first_patch_only: bool) -> list[np.ndarray]:
    """MIDI ファイルリストから guitar tonal_chw (uint8) のリストを返す。"""
    arrays: list[np.ndarray] = []
    for path in midi_paths:
        patches = midi_to_patches(path)
        for p in patches:
            if first_patch_only and p.bar_index > 0:
                continue
            arrays.append(p.tonal_chw.astype(np.uint8))
    return arrays


def load_from_npy_dir(
    target_dir: Path,
    input_dir: Path | None,
    first_patch_only: bool,
    n_samples: int | None,
    rng: random.Random,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """target npy ファイルを (target_chw, input_chw | None) のリストで返す。"""
    files = sorted(target_dir.rglob("*_tonal.npy"))
    if first_patch_only:
        files = [f for f in files if "bar0000" in f.name]
    if n_samples is not None and len(files) > n_samples:
        files = rng.sample(files, n_samples)

    result: list[tuple[np.ndarray, np.ndarray | None]] = []
    for tf in files:
        tgt = np.load(tf).astype(np.uint8)
        if tgt.ndim == 3 and tgt.shape[0] == PATCH_TICKS:
            tgt = tgt.transpose(2, 0, 1)  # (128,128,11) → (11,128,128)
        inp_arr: np.ndarray | None = None
        if input_dir is not None:
            rel = tf.relative_to(target_dir)
            inp_path = input_dir / rel
            if inp_path.exists():
                inp_arr = np.load(inp_path).astype(np.uint8)
                if inp_arr.ndim == 3 and inp_arr.shape[0] == PATCH_TICKS:
                    inp_arr = inp_arr.transpose(2, 0, 1)
        result.append((tgt, inp_arr))
    return result


# ---------------------------------------------------------------------------
# 集計・表示
# ---------------------------------------------------------------------------

def aggregate_metrics(
    samples: list[tuple[np.ndarray, np.ndarray | None]],
) -> dict:
    all_opb: list[int] = []
    all_bea: list[float] = []
    all_patterns: list[int] = []  # 全パッチ・全小節をプールしてから 1 回だけエントロピー計算
    all_rtr: list[float] = []
    all_prec: list[float] = []
    has_precision = False

    for tgt_chw, inp_chw in samples:
        ch = _guitar_channel(tgt_chw)
        m = bar_metrics(ch)
        all_opb.extend(m["onsets_per_bar"])
        all_bea.append(m["bar_end_active_frac"])
        all_patterns.extend(m["bar_end_patterns"])
        all_rtr.append(m["same_pitch_retrigger_rate"])
        if inp_chw is not None:
            prec = onset_precision_vs_skeleton(tgt_chw, inp_chw)
            if not math.isnan(prec):
                all_prec.append(prec)
                has_precision = True

    def p(arr: list, pct: float) -> float:
        return float(np.percentile(arr, pct)) if arr else float("nan")

    entropy_bits, entropy_norm = pooled_entropy(all_patterns)

    result: dict = {
        "n_patches": len(samples),
        "n_bars": len(all_opb),
        "onsets_per_bar": {
            "mean": float(np.mean(all_opb)) if all_opb else float("nan"),
            "p50": p(all_opb, 50),
            "p90": p(all_opb, 90),
            "p10": p(all_opb, 10),
        },
        "bar_end_activity_mean": float(np.mean(all_bea)) if all_bea else float("nan"),
        # 全小節プール後の 1 回だけの計算（パッチ単位平均ではない。n=len(all_patterns)）
        "bar_end_pattern_entropy_bits": entropy_bits,
        "bar_end_pattern_entropy_norm": entropy_norm,
        # same_pitch_retrigger_rate は音高セル単位の集計（onsets_per_bar とは単位が異なる）
        "same_pitch_retrigger_rate_mean": float(np.mean(all_rtr)) if all_rtr else float("nan"),
    }
    if has_precision:
        result["onset_precision_vs_skeleton"] = {
            "mean": float(np.mean(all_prec)),
            "p50": p(all_prec, 50),
        }
    return result


def print_report(label: str, m: dict) -> None:
    w = 60
    print(f"\n{'='*w}")
    print(f"  {label}")
    print(f"{'='*w}")
    opb = m["onsets_per_bar"]
    print(f"  onsets/bar             mean={opb['mean']:.2f}  p50={opb['p50']:.1f}  p90={opb['p90']:.1f}  p10={opb['p10']:.1f}")
    print(f"                         (同時ストロークは1カウント。pitch方向をORで潰した打鍵タイミング数)")
    print(f"  bar_end_activity       {m['bar_end_activity_mean']*100:.1f}%  (tick 12-15 に onset がある小節の割合)")
    print(
        f"  bar_end_entropy        {m['bar_end_pattern_entropy_bits']:.3f} bit"
        f"  (normalized={m['bar_end_pattern_entropy_norm']*100:.1f}%, max={MAX_BAR_END_ENTROPY_BITS:.1f}bit)"
    )
    print(f"                         (全小節をプールして1回だけ計算。高い=終わり方が多様)")
    print(f"  retrigger_rate         {m['same_pitch_retrigger_rate_mean']*100:.2f}%  (同一音高セル単位。onsets/barとは集計単位が異なる)")
    if "onset_precision_vs_skeleton" in m:
        p_ = m["onset_precision_vs_skeleton"]
        print(f"  precision_vs_chord     mean={p_['mean']*100:.1f}%  p50={p_['p50']*100:.1f}%")
    print(f"  (patches={m['n_patches']}, bars={m['n_bars']})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="バッキング MIDI / target npy のリズム統計を出力する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--midi",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="評価する MIDI ファイル（複数可）",
    )
    src.add_argument(
        "--npy-dir",
        type=Path,
        metavar="DIR",
        help="target npy ファイルのルートディレクトリ",
    )
    src.add_argument(
        "--pairs-dir",
        type=Path,
        metavar="DIR",
        help="pairs ルート（input/ と target/ を持つ）precision も算出",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        metavar="N",
        help="npy/pairs モードでサンプリングする最大ファイル数",
    )
    parser.add_argument(
        "--first-patch-only",
        action="store_true",
        help="各 MIDI / npy で bar_index=0 のパッチのみ使用（patches:2 重複除外）",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.midi:
        arrays = load_from_midi(args.midi, args.first_patch_only)
        samples: list[tuple[np.ndarray, np.ndarray | None]] = [(a, None) for a in arrays]
        label = f"MIDI ({len(args.midi)} files)"
    elif args.npy_dir:
        samples = load_from_npy_dir(
            args.npy_dir, None,
            args.first_patch_only, args.n_samples, rng,
        )
        label = f"npy target: {args.npy_dir}"
    else:
        target_dir = args.pairs_dir / "target"
        input_dir = args.pairs_dir / "input"
        if not target_dir.is_dir():
            parser.error(f"target ディレクトリが見つかりません: {target_dir}")
        samples = load_from_npy_dir(
            target_dir, input_dir,
            args.first_patch_only, args.n_samples, rng,
        )
        label = f"pairs: {args.pairs_dir}"

    if not samples:
        print("サンプルが 0 件です。パスを確認してください。", file=sys.stderr)
        sys.exit(1)

    m = aggregate_metrics(samples)
    print_report(label, m)


if __name__ == "__main__":
    main()
