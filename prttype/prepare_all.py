"""合成 MIDI + Guitar-TECHS を Colab 上で一括パッチ化する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from makeData.constants import DEFAULT_SYNTHETIC_COUNT
from makeData.generate import count_existing_midis, generate_dataset
from prepare_dataset import prepare_pairs
from remap_guitar_programs import remap_directory
from skeleton import PAIR_MODES

SCRIPT_DIR = Path(__file__).resolve().parent

RAW_SYNTHETIC = SCRIPT_DIR / "data" / "raw" / "synthetic"
RAW_GUITAR_TECHS = SCRIPT_DIR / "data" / "raw" / "guitar-techs"
PAIRS_SYNTHETIC = SCRIPT_DIR / "data" / "pairs" / "synthetic"
PAIRS_GUITAR_TECHS = SCRIPT_DIR / "data" / "pairs" / "guitar-techs"
PAIRS_GUITAR_TECHS_CURATED = SCRIPT_DIR / "data" / "pairs" / "guitar-techs-curated"

# 高価値サブセット: P3_music 全件 + chords を各フォルダから間引き
CURATED_TECHS_LIMITS: dict[str, int | None] = {
    "P3_music": None,  # 全件
    "P1_chords": 6,
    "P2_chords": 6,
}


def count_midi_files(raw_dir: Path) -> int:
    if not raw_dir.is_dir():
        return 0
    return len(list(raw_dir.rglob("*.mid")))


def list_curated_techs_midis(raw_dir: Path) -> list[Path]:
    """改善寄与の高い TECHS のみ（決定的に間引き）。"""
    selected: list[Path] = []
    for category, limit in CURATED_TECHS_LIMITS.items():
        cat_dir = raw_dir / category
        if not cat_dir.is_dir():
            print(f"警告: カテゴリがありません（スキップ）: {cat_dir}")
            continue
        files = sorted(cat_dir.rglob("*.mid"))
        if limit is not None:
            files = files[:limit]
        print(f"  curated {category}: {len(files)} MIDI" + ("" if limit is None else f" (max {limit})"))
        selected.extend(files)
    return selected


def ensure_synthetic_raw(
    *,
    count: int,
    seed: int,
    force_regenerate: bool = False,
    bars: int = 8,
) -> dict:
    existing = count_existing_midis(RAW_SYNTHETIC)
    if force_regenerate:
        print(f"合成 MIDI: {count} 本を先頭から再生成します（bars={bars}）")
        for path in RAW_SYNTHETIC.glob("synth_*.mid"):
            path.unlink()
        manifest_path = RAW_SYNTHETIC / "manifest.json"
        if manifest_path.is_file():
            manifest_path.unlink()
        return generate_dataset(
            RAW_SYNTHETIC,
            count=count,
            seed=seed,
            bars=bars,
            start_index=0,
        )

    if existing < count:
        print(f"合成 MIDI: 既存 {existing} 本 → 目標 {count} 本（+{count - existing} 本を追記, bars={bars}）")
        return generate_dataset(
            RAW_SYNTHETIC,
            count=count,
            seed=seed,
            bars=bars,
            start_index=existing,
        )

    print(f"合成 MIDI: 既存 {existing} 本を使用（再生成スキップ）")
    manifest_path = RAW_SYNTHETIC / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return {"count": existing, "seed": seed, "output_dir": str(RAW_SYNTHETIC)}


def prepare_all(
    *,
    synthetic_count: int = DEFAULT_SYNTHETIC_COUNT,
    synthetic_seed: int = 42,
    force_regenerate: bool = False,
    mode: str = "downbeat_chord",
    min_onsets: int = 1,
    skip_guitar_remap: bool = False,
    bars: int = 8,
    techs_full: bool = False,
) -> dict:
    print("=== Step 1: 合成 MIDI ===")
    synthetic_raw_manifest = ensure_synthetic_raw(
        count=synthetic_count,
        seed=synthetic_seed,
        force_regenerate=force_regenerate,
        bars=bars,
    )

    if force_regenerate:
        import shutil

        for pairs_dir in (PAIRS_SYNTHETIC, PAIRS_GUITAR_TECHS_CURATED):
            if pairs_dir.is_dir():
                print(f"旧ペア削除: {pairs_dir}")
                shutil.rmtree(pairs_dir)

    print("\n=== Step 2: Guitar-TECHS program 確認 ===")
    if not skip_guitar_remap and RAW_GUITAR_TECHS.is_dir():
        remap_directory(RAW_GUITAR_TECHS)
    else:
        print("Guitar-TECHS remap をスキップ")

    print("\n=== Step 3: 合成データをパッチ化 ===")
    synthetic_pairs = prepare_pairs(
        RAW_SYNTHETIC,
        PAIRS_SYNTHETIC,
        mode=mode,
        min_onsets=min_onsets,
        first_patch_only=True,  # 8小節ちょうどでも patches:2 になるバグ回避
    )

    if not RAW_GUITAR_TECHS.is_dir():
        raise FileNotFoundError(f"Guitar-TECHS がありません: {RAW_GUITAR_TECHS}")

    if techs_full:
        print("\n=== Step 4: Guitar-TECHS 全量をパッチ化 ===")
        techs_pairs_dir = PAIRS_GUITAR_TECHS
        guitar_pairs = prepare_pairs(
            RAW_GUITAR_TECHS,
            techs_pairs_dir,
            mode=mode,
            min_onsets=min_onsets,
        )
        techs_label = "full"
    else:
        print("\n=== Step 4: Guitar-TECHS 高価値サブセットをパッチ化 ===")
        print("  (P3_music 全件 + P1/P2_chords 各最大 6 本)")
        curated = list_curated_techs_midis(RAW_GUITAR_TECHS)
        if not curated:
            raise FileNotFoundError("curated TECHS MIDI が 0 件です")
        techs_pairs_dir = PAIRS_GUITAR_TECHS_CURATED
        guitar_pairs = prepare_pairs(
            RAW_GUITAR_TECHS,
            techs_pairs_dir,
            mode=mode,
            min_onsets=min_onsets,
            midi_files=curated,
        )
        techs_label = "curated"

    summary = {
        "mode": mode,
        "bars": bars,
        "techs_selection": techs_label,
        "curated_limits": None if techs_full else CURATED_TECHS_LIMITS,
        "synthetic": {
            "raw_dir": str(RAW_SYNTHETIC),
            "pairs_dir": str(PAIRS_SYNTHETIC),
            "midi_files": synthetic_raw_manifest.get("count", count_midi_files(RAW_SYNTHETIC)),
            "total_patches": synthetic_pairs["total_patches"],
        },
        "guitar_techs": {
            "raw_dir": str(RAW_GUITAR_TECHS),
            "pairs_dir": str(techs_pairs_dir),
            "midi_files": guitar_pairs["midi_files"],
            "total_patches": guitar_pairs["total_patches"],
        },
        "total_patches": synthetic_pairs["total_patches"] + guitar_pairs["total_patches"],
    }

    summary_path = SCRIPT_DIR / "data" / "pairs" / "manifest_all.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 完了 ===")
    print(f"合成パッチ: {summary['synthetic']['total_patches']}")
    print(f"TECHS ({techs_label}) パッチ: {summary['guitar_techs']['total_patches']}")
    print(f"合計パッチ: {summary['total_patches']}")
    print(f"manifest: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合成 + Guitar-TECHS を一括パッチ化（Colab 用）",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=DEFAULT_SYNTHETIC_COUNT,
        help=f"目標 MIDI 本数（既定 {DEFAULT_SYNTHETIC_COUNT}、8小節≈1パッチ）",
    )
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="合成 MIDI を必ず再生成する",
    )
    parser.add_argument(
        "--mode",
        choices=PAIR_MODES,
        default="downbeat_chord",
        help="学習タスク: downbeat_chord=小節頭コード骨格→フル（推奨）",
    )
    parser.add_argument("--min-onsets", type=int, default=1)
    parser.add_argument(
        "--bars",
        type=int,
        default=8,
        help="合成 MIDI の小節数（既定 8。長くするとパッチが爆発する）",
    )
    parser.add_argument(
        "--skip-guitar-remap",
        action="store_true",
        help="Guitar-TECHS の program 統一をスキップ",
    )
    parser.add_argument(
        "--techs-full",
        action="store_true",
        help="TECHS 全量をパッチ化（非推奨・リズム崩れリスク）。既定は curated のみ",
    )
    args = parser.parse_args()

    prepare_all(
        synthetic_count=args.synthetic_count,
        synthetic_seed=args.synthetic_seed,
        force_regenerate=args.force_regenerate,
        mode=args.mode,
        min_onsets=args.min_onsets,
        skip_guitar_remap=args.skip_guitar_remap,
        bars=args.bars,
        techs_full=args.techs_full,
    )


if __name__ == "__main__":
    main()
