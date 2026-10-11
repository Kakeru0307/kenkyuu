"""PMEmo → structure prior 学習ペア（連続 VA + 音声推定 BPM）。

配置例:
  prttype/data/pmemo/
    chorus/                 # *.mp3
    metadata/static_annotations.csv  または static_annotations.csv

使い方:
  python pmemo_to_pairs.py
  python pmemo_to_pairs.py --root data/pmemo --out data/pmemo_pairs/structure_pairs.jsonl
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from audio_emotion_pairs import (
    count_skips,
    load_bpm_cache,
    log_va_bpm_stats,
    make_pair_row,
    normalize_va_to_pm1,
    resolve_bpm,
    save_bpm_cache,
    write_jsonl,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_annotations(root: Path) -> Path:
    candidates = [
        root / "metadata" / "static_annotations.csv",
        root / "static_annotations.csv",
        root / "annotations" / "static_annotations.csv",
        root / "PMEmo2019" / "annotations" / "static_annotations.csv",
        root / "PMEmo2019" / "metadata" / "static_annotations.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "static_annotations.csv が見つかりません。PMEmo 2019 を "
        f"{root} 以下に展開してください。候補: {[str(c) for c in candidates]}"
    )


def _find_chorus_dir(root: Path) -> Path:
    candidates = [
        root / "chorus",
        root / "PMEmo2019" / "chorus",
        root / "audio" / "chorus",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"chorus ディレクトリが見つかりません: {[str(c) for c in candidates]}"
    )


def _row_va(row: dict[str, str]) -> tuple[str, float, float] | None:
    # musicId / song_id など
    mid = (
        row.get("musicId")
        or row.get("music_id")
        or row.get("song_id")
        or row.get("SongID")
        or row.get("file_name")
        or row.get("filename")
    )
    if not mid:
        return None
    # static mean columns
    v_keys = (
        "Valence(mean)",
        "valence_mean",
        "Valence_mean",
        "valence",
        "Valence",
        "static valence",
    )
    a_keys = (
        "Arousal(mean)",
        "arousal_mean",
        "Arousal_mean",
        "arousal",
        "Arousal",
        "static arousal",
    )
    v = a = None
    for k in v_keys:
        if k in row and str(row[k]).strip() != "":
            try:
                v = float(row[k])
                break
            except ValueError:
                continue
    for k in a_keys:
        if k in row and str(row[k]).strip() != "":
            try:
                a = float(row[k])
                break
            except ValueError:
                continue
    if v is None or a is None:
        return None
    return str(mid).strip(), v, a


def _resolve_audio(chorus_dir: Path, music_id: str) -> Path | None:
    stem = Path(music_id).stem
    for name in (
        f"{music_id}",
        f"{stem}.mp3",
        f"{stem}.wav",
        f"{stem}.MP3",
        f"{music_id}.mp3",
    ):
        p = chorus_dir / name
        if p.is_file():
            return p
    # 部分一致（先頭 ID）
    hits = list(chorus_dir.glob(f"{stem}.*"))
    for p in hits:
        if p.suffix.lower() in (".mp3", ".wav", ".flac", ".ogg"):
            return p
    return None


def convert(*, root: Path, out_path: Path, cache_path: Path) -> None:
    ann_path = _find_annotations(root)
    chorus_dir = _find_chorus_dir(root)
    cache = load_bpm_cache(cache_path)
    stats: Counter[str] = Counter()
    rows: list[dict] = []

    with ann_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            parsed = _row_va(raw)
            if parsed is None:
                stats["skip_bad_va"] += 1
                continue
            mid, v0, a0 = parsed
            v, a = normalize_va_to_pm1(v0, a0, scale="01")
            audio = _resolve_audio(chorus_dir, mid)
            if audio is None:
                stats["skip_no_audio"] += 1
                continue
            bpm = resolve_bpm(audio, cache=cache, cache_key=mid)
            if bpm is None:
                stats["skip_no_bpm"] += 1
                continue
            rows.append(
                make_pair_row(
                    source="pmemo",
                    clip=audio.name,
                    valence=v,
                    arousal=a,
                    bpm=bpm,
                    va_source="pmemo_static",
                )
            )
            stats["ok"] += 1

    save_bpm_cache(cache_path, cache)
    n = write_jsonl(out_path, rows)
    print(f"書き出し: {n} 件 → {out_path}")
    print(f"内訳: {count_skips(stats)}")
    log_va_bpm_stats(rows, label="pmemo")


def main() -> None:
    parser = argparse.ArgumentParser(description="PMEmo → structure prior pairs")
    parser.add_argument(
        "--root",
        type=Path,
        default=SCRIPT_DIR / "data" / "pmemo",
        help="PMEmo 展開ルート（chorus + static_annotations.csv）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "data" / "pmemo_pairs" / "structure_pairs.jsonl",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=SCRIPT_DIR / "data" / "pmemo" / "bpm_cache.json",
    )
    args = parser.parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(
            f"PMEmo ルートがありません: {args.root}\n"
            "https://huisblog.cn/PMEmo/ または公式配布から 2019 更新版を取得し、"
            "chorus/ と static_annotations.csv を配置してください（音声は git 管理しない）。"
        )
    convert(root=args.root, out_path=args.out, cache_path=args.cache)


if __name__ == "__main__":
    main()
