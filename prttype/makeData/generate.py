"""合成テスト MIDI の一括生成 CLI。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import muspy

from .constants import DEFAULT_BARS, DEFAULT_SYNTHETIC_COUNT
from .patterns import generate_random_phrase

SCRIPT_DIR = Path(__file__).resolve().parent
PRTTYPE_DIR = SCRIPT_DIR.parent


def build_filename(index: int, meta: dict) -> str:
    pattern = meta["pattern"]
    key = meta["key"]
    bpm = meta["bpm"]
    if pattern in ("progression_strum", "progression_arpeggio"):
        suffix = meta["progression"]
    elif pattern in ("chord_strum", "arpeggio"):
        suffix = meta["quality"]
    else:
        suffix = meta.get("mode", "scale")
    return f"synth_{pattern}_{key}_{suffix}_bpm{bpm:03d}_{index:04d}.mid"


def _advance_rng(rng: random.Random, steps: int, *, bars: int | None) -> None:
    for _ in range(steps):
        generate_random_phrase(rng=rng, bars=bars)


def load_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def count_existing_midis(output_dir: Path) -> int:
    if not output_dir.is_dir():
        return 0
    return len(list(output_dir.glob("synth_*.mid")))


def generate_dataset(
    output_dir: Path,
    *,
    count: int,
    seed: int = 42,
    bars: int | None = None,
    start_index: int = 0,
    manifest_name: str = "manifest.json",
) -> dict:
    """合成 MIDI を生成する。

    ``start_index`` から ``count - 1`` までのインデックスを書き出す。
    既存データを残して増量するときは ``start_index=既存本数`` とする。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if start_index >= count:
        manifest_path = output_dir / manifest_name
        existing = load_manifest(manifest_path)
        if existing is not None:
            print(f"生成スキップ: 既に {existing.get('count', start_index)} 本あります")
            return existing
        return {
            "count": start_index,
            "seed": seed,
            "bars": bars,
            "output_dir": str(output_dir),
            "entries": [],
        }

    rng = random.Random(seed)
    if start_index > 0:
        _advance_rng(rng, start_index, bars=bars)

    new_entries: list[dict] = []
    for index in range(start_index, count):
        music, meta = generate_random_phrase(rng=rng, bars=bars)
        filename = build_filename(index, meta)
        midi_path = output_dir / filename
        muspy.write_midi(midi_path, music)
        new_entries.append(
            {
                "file": filename,
                "path": str(midi_path),
                **meta,
            }
        )
        if (index - start_index + 1) % 500 == 0:
            print(f"  生成中: {index + 1}/{count}")

    manifest_path = output_dir / manifest_name
    previous = load_manifest(manifest_path)
    entries = (previous.get("entries", []) if previous else []) + new_entries
    manifest = {
        "count": count,
        "seed": seed,
        "bars": bars,
        "output_dir": str(output_dir),
        "entries": entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"新規生成: {len(new_entries)} MIDI（合計 {count} 本）")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合成ギター MIDI（コード・音階・ストローク）を生成",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PRTTYPE_DIR / "data" / "raw" / "synthetic",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_SYNTHETIC_COUNT,
        help=f"目標 MIDI 本数（既定 {DEFAULT_SYNTHETIC_COUNT}）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bars",
        type=int,
        default=8,
        help="小節数（既定 8。Colab ディスク対策で 8 固定推奨）",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="既存 MIDI を残し、不足分だけ追記生成する",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存を無視して count 本を先頭から再生成する",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    existing = count_existing_midis(output_dir)
    if args.force:
        for path in output_dir.glob("synth_*.mid"):
            path.unlink()
        manifest_path = output_dir / "manifest.json"
        if manifest_path.is_file():
            manifest_path.unlink()
        start_index = 0
        print("既存の合成 MIDI を削除して再生成します")
    elif args.append:
        start_index = existing
    elif existing > 0:
        raise SystemExit(
            f"既存の合成 MIDI が {existing} 本あります。"
            "入れ替えは --force、追記は --append を明示してください。"
        )
    else:
        start_index = 0

    manifest = generate_dataset(
        output_dir,
        count=args.count,
        seed=args.seed,
        bars=args.bars,
        start_index=start_index,
    )
    print(f"目標: {manifest['count']} MIDI")
    print(f"保存先: {output_dir}")
    print(f"manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
