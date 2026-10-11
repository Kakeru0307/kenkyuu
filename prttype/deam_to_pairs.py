"""DEAM → structure prior 学習ペア（連続 VA + 音声推定 BPM）。

配置例（Zenodo DEAM_Annotations + 公式 audio）:
  prttype/data/deam/
    audio/   または MEMD_audio/ 等（*.mp3）
    annotations/annotations averaged per song/song_level/
      static_annotations_averaged_songs_1_2000.csv
      static_annotations_averaged_songs_2000_2058.csv

使い方:
  python deam_to_pairs.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from audio_emotion_pairs import (
    count_skips,
    infer_corpus_scale,
    load_bpm_cache,
    log_va_bpm_stats,
    make_pair_row,
    normalize_va_to_pm1,
    resolve_bpm,
    save_bpm_cache,
    write_jsonl,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_song_level_csvs(root: Path) -> list[Path]:
    candidates_dirs = [
        root / "annotations" / "annotations averaged per song" / "song_level",
        root / "DEAM_Annotations" / "annotations averaged per song" / "song_level",
        root / "annotations averaged per song" / "song_level",
    ]
    paths: list[Path] = []
    for d in candidates_dirs:
        if d.is_dir():
            paths.extend(sorted(d.glob("static_annotations_averaged_songs_*.csv")))
    # legacy: separate valence.csv / arousal.csv
    legacy = [
        root / "annotations" / "annotations averaged per song",
        root / "annotations averaged per song",
    ]
    if not paths:
        for d in legacy:
            v = d / "valence.csv"
            a = d / "arousal.csv"
            if v.is_file() and a.is_file():
                return [v, a]  # special marker handled below — won't use this path
    if not paths:
        raise FileNotFoundError(
            "song_level static_annotations_averaged_songs_*.csv が見つかりません。"
            f"候補 dir: {[str(c) for c in candidates_dirs]}"
        )
    return paths


def _read_song_level(paths: list[Path]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            if reader.fieldnames is None:
                continue
            fields = {name.strip().lower(): name for name in reader.fieldnames}
            id_k = fields.get("song_id") or fields.get("id")
            v_k = fields.get("valence_mean") or fields.get("valence")
            a_k = fields.get("arousal_mean") or fields.get("arousal")
            if not id_k or not v_k or not a_k:
                raise RuntimeError(f"想定外の列: {path} fields={reader.fieldnames}")
            for row in reader:
                sid = str(row[id_k]).strip()
                if not sid:
                    continue
                try:
                    out[sid] = (float(row[v_k]), float(row[a_k]))
                except (TypeError, ValueError):
                    continue
    return out


def _find_audio_dir(root: Path) -> Path:
    candidates = [
        root / "audio",
        root / "DEAM_audio",
        root / "MEMD_audio",
        root / "audio" / "MEMD_audio",
    ]
    for p in candidates:
        if p.is_dir() and (
            any(p.glob("*.mp3"))
            or any(p.glob("*.wav"))
            or any(p.rglob("*.mp3"))
        ):
            return p
    raise FileNotFoundError(
        "音声ディレクトリが見つかりません（*.mp3/*.wav）。\n"
        "https://cvml.unige.ch/databases/DEAM/ の audio を "
        f"{root / 'audio'} に展開してください。候補: {[str(c) for c in candidates]}"
    )


def _resolve_audio(audio_dir: Path, song_id: str) -> Path | None:
    stem = Path(song_id).stem
    names: list[str] = [f"{stem}.mp3", f"{stem}.wav", f"{song_id}.mp3"]
    if stem.isdigit():
        n = int(stem)
        names.extend([f"{n}.mp3", f"{n:04d}.mp3", f"{n:03d}.mp3"])
    for name in names:
        p = audio_dir / name
        if p.is_file():
            return p
    for name in names:
        hits = list(audio_dir.rglob(name))
        if hits:
            return hits[0]
    return None


def convert(*, root: Path, out_path: Path, cache_path: Path) -> None:
    csvs = _find_song_level_csvs(root)
    va_map = _read_song_level(csvs)
    scale = infer_corpus_scale(list(va_map.values()))
    print(f"DEAM VA scale inferred: {scale} (n_ids={len(va_map)}) from {[p.name for p in csvs]}")
    audio_dir = _find_audio_dir(root)
    cache = load_bpm_cache(cache_path)
    stats: Counter[str] = Counter()
    rows: list[dict] = []

    for sid in sorted(va_map, key=lambda x: (len(x), x)):
        v0, a0 = va_map[sid]
        v, a = normalize_va_to_pm1(v0, a0, scale=scale)
        audio = _resolve_audio(audio_dir, sid)
        if audio is None:
            stats["skip_no_audio"] += 1
            continue
        bpm = resolve_bpm(audio, cache=cache, cache_key=sid)
        if bpm is None:
            stats["skip_no_bpm"] += 1
            continue
        try:
            clip = str(audio.relative_to(audio_dir))
        except ValueError:
            clip = audio.name
        rows.append(
            make_pair_row(
                source="deam",
                clip=clip,
                valence=v,
                arousal=a,
                bpm=bpm,
                va_source="deam_song_mean",
            )
        )
        stats["ok"] += 1

    save_bpm_cache(cache_path, cache)
    n = write_jsonl(out_path, rows)
    print(f"書き出し: {n} 件 → {out_path}")
    print(f"内訳: {count_skips(stats)}")
    log_va_bpm_stats(rows, label="deam")


def main() -> None:
    parser = argparse.ArgumentParser(description="DEAM → structure prior pairs")
    parser.add_argument("--root", type=Path, default=SCRIPT_DIR / "data" / "deam")
    parser.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "data" / "deam_pairs" / "structure_pairs.jsonl",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=SCRIPT_DIR / "data" / "deam" / "bpm_cache.json",
    )
    args = parser.parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(
            f"DEAM ルートがありません: {args.root}\n"
            "annotations は Zenodo DEAM_Annotations、audio は "
            "https://cvml.unige.ch/databases/DEAM/ から取得（音声は git 管理しない）。"
        )
    convert(root=args.root, out_path=args.out, cache_path=args.cache)


if __name__ == "__main__":
    main()
