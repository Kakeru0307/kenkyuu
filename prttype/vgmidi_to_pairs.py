"""VGMIDI → structure prior 学習ペア。

JSON 連続VA（アノテータ×時系列の平均）と labelled/midi の実 BPM を組にする。
phrases / CSV の ±1 は使わない。

使い方:
  python vgmidi_to_pairs.py --fetch-json
  python vgmidi_to_pairs.py --midi-dir data/vgmidi/labelled_midi --fetch-json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
KEYS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
GITHUB_RAW = "https://raw.githubusercontent.com/lucasnfe/vgmidi/master"


def _fetch_json(name: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return json.loads(dest.read_text(encoding="utf-8"))
    url = f"{GITHUB_RAW}/labelled/annotations/{name}"
    print(f"download {url}")
    data = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "vgmidi-to-pairs"})
    ).read()
    dest.write_bytes(data)
    return json.loads(data.decode("utf-8"))


def _piece_va_means(obj: dict) -> dict[str, tuple[float, float, str]]:
    """stem -> (valence, arousal, midi_filename)."""
    anns = obj["annotations"]
    pieces = obj["pieces"]
    by: dict[str, list[dict]] = defaultdict(list)
    for key, val in anns.items():
        pid = key.rsplit("_", 1)[0]
        by[pid].append(val)
    out: dict[str, tuple[float, float, str]] = {}
    for pid, rows in by.items():
        if pid not in pieces:
            continue
        mv = sum(sum(r["valence"]) / len(r["valence"]) for r in rows) / len(rows)
        ma = sum(sum(r["arousal"]) / len(r["arousal"]) for r in rows) / len(rows)
        midi_name = Path(pieces[pid]["midi"]).name
        out[Path(midi_name).stem] = (float(mv), float(ma), midi_name)
    return out


def _from_midi(midi_path: Path) -> tuple[float | None, str | None, str | None]:
    try:
        import muspy

        music = muspy.read_midi(str(midi_path))
    except Exception:
        return None, None, None
    bpm = float(music.tempos[0].qpm) if music.tempos else None
    key = mode = None
    if music.key_signatures:
        ks = music.key_signatures[0]
        if ks.root is not None:
            key = KEYS[int(ks.root) % 12]
        if ks.mode is not None:
            mode = (
                "natural_minor"
                if str(ks.mode).lower().startswith("min") or ks.mode == 1
                else "major"
            )
    return bpm, key, mode


def convert(*, midi_dir: Path, json_paths: list[Path], out_path: Path) -> None:
    cont: dict[str, tuple[float, float, str]] = {}
    for jp in json_paths:
        cont.update(_piece_va_means(json.loads(jp.read_text(encoding="utf-8"))))

    midi_by = {p.stem: p for p in midi_dir.glob("*.mid")}
    records: list[dict] = []
    stats: Counter[str] = Counter()
    for stem, (v, a, _midi_name) in cont.items():
        path = midi_by.get(stem)
        if path is None:
            stats["skip_no_midi"] += 1
            continue
        bpm, key, mode = _from_midi(path)
        if bpm is None or bpm <= 0:
            stats["skip_no_bpm"] += 1
            continue
        structure: dict[str, object] = {"bpm": round(float(bpm), 1), "bars": 8}
        if key:
            structure["key"] = key
        if mode:
            structure["mode"] = mode
        records.append({
            "source": "vgmidi",
            "clip": stem,
            "va": {"valence": v, "arousal": a},
            "va_source": "vgmidi_json_mean",
            "structure": structure,
        })
        stats["ok"] += 1
        stats["has_key"] += int(key is not None)
        stats["has_mode"] += int(mode is not None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"書き出し: {len(records)} 件 → {out_path}")
    print(f"内訳: {dict(stats)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VGMIDI → structure prior 学習ペア")
    parser.add_argument(
        "--midi-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "vgmidi" / "labelled_midi",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "vgmidi" / "annotations",
    )
    parser.add_argument("--fetch-json", action="store_true", help="GitHub から raw JSON を取得")
    parser.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "data" / "vgmidi_pairs" / "structure_pairs.jsonl",
    )
    args = parser.parse_args()

    names = ["vgmidi_raw_1.json", "vgmidi_raw_2.json"]
    paths: list[Path] = []
    for name in names:
        dest = args.json_dir / name
        if args.fetch_json or not dest.is_file():
            _fetch_json(name, dest)
        if not dest.is_file():
            raise FileNotFoundError(dest)
        paths.append(dest)

    if not args.midi_dir.is_dir():
        raise FileNotFoundError(
            f"MIDI ディレクトリがありません: {args.midi_dir}\n"
            "GitHub lucasnfe/vgmidi の labelled/midi を配置してください。"
        )
    convert(midi_dir=args.midi_dir, json_paths=paths, out_path=args.out)


if __name__ == "__main__":
    main()
