"""EMOPIA → structure prior 学習ペア変換。

キー・モードは key_mode_tempo.csv（keymode 1=major, 2=minor）または MIDI から。
BPM は学習に使わない（EMOPIA tempo がほぼ一定のため書かない）。
VA は VGMIDI 特徴転写（大きさ）× 象限符号。転写失敗行はスキップ（±0.6 フォールバックなし）。

使い方:
    python emopia_to_pairs.py \\
      --emopia-dir data/emopia/extracted/EMOPIA_2.2 \\
      --vgmidi-midi-dir data/vgmidi/labelled_midi \\
      --vgmidi-json-dir data/vgmidi/annotations
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from vgmidi_va_transfer import fit_va_regressor, transfer_va_for_midi

SCRIPT_DIR = Path(__file__).resolve().parent

_ENHARMONIC = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
    "Cb": "B", "Fb": "E", "E#": "F", "B#": "C",
}
KEYS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

_Q_RE = re.compile(r"^(Q[1-4])_", re.IGNORECASE)


def _norm_keyname(raw: str) -> str | None:
    """'C#', 'a', 'Db' 等 → KEYS 表記。モードは見ない（keymode 列を使う）。"""
    s = raw.strip()
    if not s:
        return None
    m = re.match(r"([A-Ga-g])([#b♯♭]?)", s)
    if not m:
        return None
    letter, acc = m.group(1), m.group(2).replace("♯", "#").replace("♭", "b")
    name = letter.upper() + acc
    name = _ENHARMONIC.get(name, name)
    return name if name in KEYS else None


def _norm_keymode(raw: str) -> str | None:
    """EMOPIA keymode: 1=major, 2=minor。文字列 major/minor も可。"""
    s = raw.strip().lower()
    if s in ("1", "major", "maj", "ionian"):
        return "major"
    if s in ("2", "minor", "min", "aeolian", "natural_minor"):
        return "natural_minor"
    return None


def _find_col(header: list[str], *needles: str) -> str | None:
    for col in header:
        c = col.strip().lower()
        if any(n in c for n in needles):
            return col
    return None


def _load_key_mode(emopia_dir: Path) -> dict[str, dict[str, str]]:
    paths = list(emopia_dir.rglob("key_mode_tempo.csv"))
    if not paths:
        print("[情報] key_mode_tempo.csv なし → MIDI からキー/モードを推定")
        return {}
    path = paths[0]
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    print(f"key_mode_tempo.csv: {path}  列={header}")
    id_col = _find_col(header, "name", "file", "id", "clip") or (header[0] if header else None)
    key_col = _find_col(header, "keyname", "key")
    mode_col = _find_col(header, "keymode", "mode")
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if id_col is None:
            break
        stem = Path(str(row.get(id_col, "")).strip()).stem
        if not stem:
            continue
        out[stem] = {
            "key": str(row.get(key_col, "")) if key_col else "",
            "mode": str(row.get(mode_col, "")) if mode_col else "",
        }
    return out


def _from_midi_key_mode(midi_path: Path) -> tuple[str | None, str | None]:
    try:
        import muspy

        music = muspy.read_midi(str(midi_path))
    except Exception:
        return None, None
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
    return key, mode


def download(root: Path) -> Path:
    import muspy

    root.mkdir(parents=True, exist_ok=True)
    muspy.EMOPIADataset(root, download_and_extract=True)
    return root


def convert(
    emopia_dir: Path,
    out_path: Path,
    *,
    vgmidi_midi_dir: Path,
    vgmidi_json_dir: Path,
) -> None:
    midis = sorted(p for p in emopia_dir.rglob("*.mid") if not p.name.startswith("._"))
    if not midis:
        raise FileNotFoundError(f"MIDI が見つかりません: {emopia_dir}")

    json_paths = sorted(vgmidi_json_dir.glob("vgmidi_raw_*.json"))
    if not json_paths:
        raise FileNotFoundError(
            f"VGMIDI JSON がありません: {vgmidi_json_dir}\n"
            "先に python vgmidi_to_pairs.py --fetch-json を実行してください。"
        )
    if not vgmidi_midi_dir.is_dir():
        raise FileNotFoundError(f"VGMIDI MIDI がありません: {vgmidi_midi_dir}")

    print("VGMIDI → VA 転写モデルを学習中...")
    W, meta = fit_va_regressor(midi_dir=vgmidi_midi_dir, json_paths=json_paths)
    print(f"転写モデル: samples fitted, W shape={W.shape}")

    kmt = _load_key_mode(emopia_dir)
    records: list[dict] = []
    stats: Counter[str] = Counter()
    for midi in midis:
        m = _Q_RE.match(midi.name)
        if not m:
            stats["skip_no_q"] += 1
            continue
        q = m.group(1).upper()

        transferred = transfer_va_for_midi(midi, q=q, W=W, meta=meta)
        if transferred is None:
            stats["skip_va_transfer"] += 1
            continue
        valence, arousal = transferred

        key: str | None = None
        mode: str | None = None
        meta_row = kmt.get(midi.stem)
        if meta_row:
            key = _norm_keyname(meta_row["key"])
            mode = _norm_keymode(meta_row["mode"])
            stats["csv_hit"] += 1
        if key is None or mode is None:
            m_key, m_mode = _from_midi_key_mode(midi)
            key = key or m_key
            mode = mode or m_mode

        # BPM は書かない（学習マスク対象）
        structure: dict[str, object] = {"bars": 8}
        if key:
            structure["key"] = key
        if mode:
            structure["mode"] = mode
        if key is None and mode is None:
            stats["skip_no_key_mode"] += 1
            continue

        records.append({
            "source": "emopia",
            "clip": midi.stem,
            "q_label": q,
            "va": {"valence": valence, "arousal": arousal},
            "va_source": "vgmidi_feature_transfer",
            "structure": structure,
        })
        stats[q] += 1
        stats["has_key"] += int(key is not None)
        stats["has_mode"] += int(mode is not None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"書き出し: {len(records)} 件 → {out_path}")
    print(f"内訳: {dict(stats)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EMOPIA → structure prior 学習ペア")
    parser.add_argument(
        "--emopia-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "emopia" / "extracted" / "EMOPIA_2.2",
    )
    parser.add_argument("--download", action="store_true", help="MuSpy 経由で EMOPIA 2.2 を取得")
    parser.add_argument(
        "--vgmidi-midi-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "vgmidi" / "labelled_midi",
    )
    parser.add_argument(
        "--vgmidi-json-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "vgmidi" / "annotations",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "data" / "emopia_pairs" / "structure_pairs.jsonl",
    )
    args = parser.parse_args()

    if args.download:
        download(args.emopia_dir)
    convert(
        args.emopia_dir,
        args.out,
        vgmidi_midi_dir=args.vgmidi_midi_dir,
        vgmidi_json_dir=args.vgmidi_json_dir,
    )


if __name__ == "__main__":
    main()
