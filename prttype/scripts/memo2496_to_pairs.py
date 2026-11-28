#!/usr/bin/env python3
"""Memo2496 MusicRaw → structure prior JSONL.

Chordino でコード列、librosa で BPM／キー。注釈 VA と結合して
`structure_pairs.jsonl` を書く。カタログ外の安定ループは亜種一覧へ。

WSL 想定:
  python3 memo2496_to_pairs.py \\
    --audio-dir MusicRawData \\
    --annotations Annotations \\
    --out-jsonl structure_pairs.jsonl \\
    --variants-out variant_candidates.jsonl \\
    --limit 30
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# 研究リポの progressions / KEYS（Windows マウント）
_REPO_CANDIDATES = (
    Path("/mnt/c/Users/kake0/github/研究/prttype"),
    Path.home() / "github/研究/prttype",
)
for _p in _REPO_CANDIDATES:
    if (_p / "makeData" / "progressions.py").is_file():
        sys.path.insert(0, str(_p))
        break

from makeData.constants import KEY_TO_SEMITONE, KEYS  # noqa: E402
from makeData.progressions import (  # noqa: E402
    PROGRESSION_BY_NAME,
    PROGRESSIONS,
    e3_canonical_mode,
)

NOTE_NAMES = list(KEYS)
PC_TO_KEY = {v: k for k, v in KEY_TO_SEMITONE.items()}
SHARP_TO_FLAT = {
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
    "A#": "Bb",
}

# Krumhansl–Kessler
_KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

_CHORD_RE = re.compile(
    r"^(?P<root>[A-G](?:#|b)?)(?P<qual>maj7|min7|m7|maj|min|m|dim|aug|7|sus[24]|)?$",
    re.IGNORECASE,
)

# Chordino quality → catalog quality
_QUAL_MAP = {
    "": "maj",
    "maj": "maj",
    "major": "maj",
    "min": "min",
    "m": "min",
    "minor": "min",
    "7": "7",
    "maj7": "maj",  # catalog に maj7 無し → maj に寄せる
    "m7": "m7",
    "min7": "m7",
    "dim": "dim",
    "aug": "maj",
    "sus2": "maj",
    "sus4": "maj",
}


def _canon_key(name: str) -> str | None:
    n = name.strip()
    if n in KEY_TO_SEMITONE:
        return n
    if n in SHARP_TO_FLAT:
        return SHARP_TO_FLAT[n]
    return None


def parse_chord_label(label: str) -> tuple[int, str] | None:
    """Chordino ラベル → (pitch_class, quality) 。N / 不明は None。"""
    s = (label or "").strip()
    if not s or s.upper() == "N" or s == "None":
        return None
    # Chordino は時々 "C:maj" / "Am7" / "Bb"
    s = s.replace(":", "")
    m = _CHORD_RE.match(s)
    if not m:
        return None
    root = _canon_key(m.group("root"))
    if root is None:
        return None
    raw_q = (m.group("qual") or "").lower()
    quality = _QUAL_MAP.get(raw_q, "maj")
    return KEY_TO_SEMITONE[root], quality


def estimate_bpm(path: Path) -> float:
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=90.0)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if tempo.size else 120.0
    bpm = float(tempo)
    if not math.isfinite(bpm) or bpm < 40.0 or bpm > 240.0:
        return 120.0
    # 半速／倍速の粗い補正
    if bpm < 60.0:
        bpm *= 2.0
    elif bpm > 200.0:
        bpm *= 0.5
    return float(np.clip(bpm, 40.0, 240.0))


def estimate_key_mode(path: Path) -> tuple[str, str, float]:
    """(key, mode, confidence)。mode は major / natural_minor。"""
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=90.0)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean = chroma.mean(axis=1).astype(np.float64)
    mean = mean - mean.mean()
    norm = float(np.linalg.norm(mean))
    if norm < 1e-8:
        return "C", "major", 0.0
    mean = mean / norm

    best = (-2.0, "C", "major")
    scores: list[float] = []
    for mode_name, prof in (("major", _KK_MAJOR), ("natural_minor", _KK_MINOR)):
        p = prof - prof.mean()
        p = p / (np.linalg.norm(p) + 1e-8)
        for shift in range(12):
            corr = float(np.dot(mean, np.roll(p, shift)))
            scores.append(corr)
            if corr > best[0]:
                best = (corr, PC_TO_KEY[shift], mode_name)
    scores.sort(reverse=True)
    conf = 0.0
    if len(scores) >= 2 and scores[0] > 1e-8:
        conf = float((scores[0] - scores[1]) / scores[0])
    return best[1], best[2], conf


def _sonic_annotator_bin() -> Path | None:
    root = Path(__file__).resolve().parent
    cand = root / "bin" / "sonic-annotator"
    return cand if cand.is_file() else None


def extract_chords_chordino(path: Path) -> list[tuple[float, str]]:
    """Chordino 経由の時間付きコード列。

    優先順: chord-extractor → vamp モジュール → sonic-annotator CLI
    （WSL Python 3.12 では前二者は入りにくいので CLI が本命）。
    """
    import os
    import subprocess
    import tempfile

    vamp_dir = Path(__file__).resolve().parent / "vamp_plugins"
    if vamp_dir.is_dir():
        os.environ["VAMP_PATH"] = str(vamp_dir)

    try:
        from chord_extractor.extractors import Chordino

        changes = Chordino(roll_on=1).extract(str(path))
        out: list[tuple[float, str]] = []
        for ch in changes:
            lab = getattr(ch, "chord", None) or getattr(ch, "label", None)
            ts = getattr(ch, "timestamp", None)
            if lab is None or ts is None:
                continue
            out.append((float(ts), str(lab)))
        return out
    except ImportError:
        pass

    try:
        import librosa
        import vamp

        y, sr = librosa.load(str(path), sr=None, mono=True)
        raw = vamp.collect(y, sr, "nnls-chroma:chordino")
        items = raw.get("list") if isinstance(raw, dict) else None
        if items is None and isinstance(raw, dict):
            for v in raw.values():
                if isinstance(v, list):
                    items = v
                    break
        out = []
        for it in items or []:
            lab = it.get("label") if isinstance(it, dict) else None
            ts = it.get("timestamp") if isinstance(it, dict) else None
            if lab is None or ts is None:
                continue
            try:
                t = float(ts)
            except (TypeError, ValueError):
                t = float(getattr(ts, "timestamp", 0.0) or 0.0)
            out.append((t, str(lab)))
        return out
    except ImportError:
        pass

    sa = _sonic_annotator_bin()
    if sa is None:
        raise RuntimeError("Chordino 不可: vamp / sonic-annotator が無い")

    # CSV: time,label
    with tempfile.TemporaryDirectory(prefix="chordino_") as td:
        out_csv = Path(td) / "chords.csv"
        # transform id は環境で違うことがあるので -w csv で chordino を指定
        cmd = [
            str(sa),
            "-d",
            "vamp:nnls-chroma:chordino:simplechord",
            "-w",
            "csv",
            "--csv-stdout",
            str(path),
        ]
        env = os.environ.copy()
        env["VAMP_PATH"] = str(vamp_dir)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        text = proc.stdout or ""
        if proc.returncode != 0 and not text.strip():
            # 別 output 名を試す
            cmd[2] = "vamp:nnls-chroma:chordino:chordino"
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
            text = proc.stdout or ""
        if proc.returncode != 0 and not text.strip():
            raise RuntimeError(
                f"sonic-annotator failed rc={proc.returncode}: {proc.stderr[-500:]}"
            )
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # sonic-annotator csv: often "file,time,label" or "time,label"
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) < 2:
                continue
            # last col = label, time is previous numeric
            lab = parts[-1]
            t_raw = parts[-2]
            try:
                t = float(t_raw)
            except ValueError:
                continue
            if lab.upper() == "N":
                continue
            out.append((t, lab))
        return out


def chords_to_bar_degrees(
    changes: list[tuple[float, str]],
    *,
    bpm: float,
    key: str,
    max_bars: int = 64,
) -> list[tuple[int, str] | None]:
    """小節ごとの (degree 1-7, quality)。不明は None。"""
    if not changes or bpm <= 0:
        return []
    tonic = KEY_TO_SEMITONE[key]
    bar_sec = 4.0 * 60.0 / bpm
    end_t = changes[-1][0] + bar_sec
    n_bars = min(max_bars, max(1, int(math.ceil(end_t / bar_sec))))
    # 各小節中央のコード
    bars: list[tuple[int, str] | None] = []
    for b in range(n_bars):
        t = (b + 0.5) * bar_sec
        lab = changes[0][1]
        for ts, L in changes:
            if ts <= t:
                lab = L
            else:
                break
        parsed = parse_chord_label(lab)
        if parsed is None:
            bars.append(None)
            continue
        pc, qual = parsed
        # 度数: 半音差 → メジャースケール度数（白鍵近似）。♯♭ は accidental 無視で丸め。
        # カタログ照合は degree+quality。tonic からの PC 差を diatonic に写す。
        rel = (pc - tonic) % 12
        # major scale pcs: 0,2,4,5,7,9,11 → degrees 1..7
        major_pcs = (0, 2, 4, 5, 7, 9, 11)
        if rel in major_pcs:
            deg = major_pcs.index(rel) + 1
        else:
            # 最も近い diatonic（同距離なら下）
            deg = min(range(7), key=lambda i: min((rel - major_pcs[i]) % 12, (major_pcs[i] - rel) % 12)) + 1
        bars.append((deg, qual))
    return bars


def find_stable_loop(
    bars: list[tuple[int, str] | None],
    lengths: tuple[int, ...] = (4, 8),
) -> tuple[list[tuple[int, str]], int, int] | None:
    """安定ループ (degrees, length, repeats)。None なら不安定。"""
    clean = [b for b in bars if b is not None]
    if len(clean) < 8:
        return None
    # None を飛ばさず、連続バーで探す（None が多いと失敗）
    seq = bars
    best: tuple[list[tuple[int, str]], int, int] | None = None
    for L in lengths:
        if len(seq) < L * 2:
            continue
        counts: Counter[tuple[tuple[int, str], ...]] = Counter()
        for i in range(0, len(seq) - L + 1, L):
            chunk = seq[i : i + L]
            if any(c is None for c in chunk):
                continue
            counts[tuple(chunk)] += 1  # type: ignore[arg-type]
        if not counts:
            continue
        loop, n = counts.most_common(1)[0]
        if n >= 2:
            degrees = list(loop)
            if best is None or n > best[2] or (n == best[2] and L < best[1]):
                best = (degrees, L, n)
    return best


def _deg_tuple(spec_degrees: tuple) -> list[tuple[int, str]]:
    return [(int(d), str(q)) for d, _acc, q in spec_degrees]


def _collapse_qual(q: str) -> str:
    if q in ("7", "maj"):
        return "maj"
    if q in ("m7", "min"):
        return "min"
    return q


def rotations(seq: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    return [seq[i:] + seq[:i] for i in range(len(seq))]


def match_progression(
    loop: list[tuple[int, str]],
) -> tuple[str | None, int]:
    """(name or None, edit_distance_to_nearest)。完全一致（回転・7th潰し可）を優先。"""
    if not loop:
        return None, 99
    n = len(loop)
    collapsed = [(_d, _collapse_qual(q)) for _d, q in loop]

    # exact / collapsed exact with rotation
    for spec in PROGRESSIONS:
        cat = _deg_tuple(spec.degrees)
        if len(cat) != n:
            continue
        cat_c = [(_d, _collapse_qual(q)) for _d, q in cat]
        for rot in rotations(loop):
            if rot == cat:
                return spec.name, 0
        for rot in rotations(collapsed):
            if rot == cat_c:
                return spec.name, 0

    # edit distance 1（置換1）
    best_dist = 99
    best_name: str | None = None
    for spec in PROGRESSIONS:
        cat = [(_d, _collapse_qual(q)) for _d, q in _deg_tuple(spec.degrees)]
        if len(cat) != n:
            continue
        for rot in rotations(collapsed):
            dist = sum(1 for a, b in zip(rot, cat) if a != b)
            if dist < best_dist:
                best_dist = dist
                best_name = spec.name
    return best_name, best_dist


def infer_bpc(loop: list[tuple[int, str]], bars: list[tuple[int, str] | None]) -> int | None:
    """ループが小節ごとに変わるなら 1。2小節同じが続けば 2。"""
    if not loop:
        return None
    # 簡易: 隣接バーが同じ割合
    pairs = [(bars[i], bars[i + 1]) for i in range(len(bars) - 1) if bars[i] and bars[i + 1]]
    if not pairs:
        return 1
    same = sum(1 for a, b in pairs if a == b)
    ratio = same / len(pairs)
    if ratio >= 0.45:
        return 2
    if ratio <= 0.15:
        return 1
    return None


def _mean_window_row(row: dict[str, str], *, t0_ms: int = 15000, t1_ms: int = 75000) -> float | None:
    """Memo2496 の sample_XXXms 列から区間平均。空なら全列平均。"""
    vals_win: list[float] = []
    vals_all: list[float] = []
    for k, raw in row.items():
        if not k.startswith("sample_") or not k.endswith("ms"):
            continue
        try:
            ms = int(k[len("sample_") : -len("ms")])
            x = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x):
            continue
        vals_all.append(x)
        if t0_ms <= ms <= t1_ms:
            vals_win.append(x)
    use = vals_win or vals_all
    if not use:
        return None
    return float(np.mean(use))


def load_song_va(annotations: Path | None) -> dict[str, tuple[float, float]]:
    """mp3 stem(UUID) → (v, a) in [-1,1]。

    Memo2496 Annotations:
      songs_info_all.csv … song_id,file_name
      valence_all_average.csv / arousal_all_average.csv … 時系列平均
    """
    if annotations is None or not annotations.exists():
        return {}
    root = annotations if annotations.is_dir() else annotations.parent
    info = root / "songs_info_all.csv"
    v_csv = root / "valence_all_average.csv"
    a_csv = root / "arousal_all_average.csv"
    if not (info.is_file() and v_csv.is_file() and a_csv.is_file()):
        return {}

    id_to_stem: dict[str, str] = {}
    with info.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("song_id", "")).strip()
            fn = str(row.get("file_name", "")).strip()
            if sid and fn:
                id_to_stem[sid] = Path(fn).stem

    v_map: dict[str, float] = {}
    with v_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("song_id", "")).strip()
            m = _mean_window_row(row)
            if sid and m is not None:
                v_map[sid] = m
    a_map: dict[str, float] = {}
    with a_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("song_id", "")).strip()
            m = _mean_window_row(row)
            if sid and m is not None:
                a_map[sid] = m

    agg: dict[str, tuple[float, float]] = {}
    for sid, stem in id_to_stem.items():
        if sid not in v_map or sid not in a_map:
            continue
        v = float(np.clip(v_map[sid], -1.0, 1.0))
        a = float(np.clip(a_map[sid], -1.0, 1.0))
        agg[stem] = (v, a)
    return agg


def list_audio(audio_dir: Path) -> list[Path]:
    files = sorted(audio_dir.rglob("*.mp3")) + sorted(audio_dir.rglob("*.wav"))
    # 重複除去
    seen: set[Path] = set()
    out: list[Path] = []
    for p in files:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def process_one(
    path: Path,
    *,
    va_map: dict[str, tuple[float, float]],
    key_conf_min: float,
    use_chordino: bool,
) -> dict[str, Any]:
    sid = path.stem
    row: dict[str, Any] = {
        "source": "memo2496",
        "clip": sid,
        "path": str(path),
    }
    if sid in va_map:
        v, a = va_map[sid]
        row["va"] = {"valence": v, "arousal": a}
        row["va_source"] = "memo2496_expert"
    else:
        row["va"] = None

    bpm = estimate_bpm(path)
    key, mode, key_conf = estimate_key_mode(path)
    st: dict[str, Any] = {"bpm": bpm, "key": key, "mode": mode, "bars": 8}
    mask = {
        "bpm": True,
        "mode": True,
        "key": key_conf >= key_conf_min,
        "prog": False,
        "bpc": False,
    }
    meta: dict[str, Any] = {
        "key_confidence": key_conf,
        "loop": None,
        "match": None,
        "edit_distance": None,
        "variant": False,
    }

    if use_chordino:
        try:
            changes = extract_chords_chordino(path)
        except Exception as e:  # noqa: BLE001
            changes = []
            meta["chordino_error"] = str(e)
        bars = chords_to_bar_degrees(changes, bpm=bpm, key=key)
        found = find_stable_loop(bars)
        if found is not None:
            loop, L, repeats = found
            meta["loop"] = [{"degree": d, "quality": q} for d, q in loop]
            meta["loop_len"] = L
            meta["loop_repeats"] = repeats
            name, dist = match_progression(loop)
            meta["edit_distance"] = dist
            bpc = infer_bpc(loop, bars)
            if bpc in (1, 2):
                st["bars_per_chord"] = bpc
                mask["bpc"] = True
            if name is not None and dist == 0:
                st["progression"] = name
                st["mode"] = e3_canonical_mode(name, mode)
                mask["prog"] = True
                mask["mode"] = True
                meta["match"] = name
            elif name is not None and dist == 1:
                meta["match"] = name
                meta["variant"] = True
                meta["near"] = name
            else:
                meta["variant"] = True
                meta["near"] = name

    row["structure"] = st
    row["_label_mask"] = mask
    row["_meta"] = meta
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, default=None)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--variants-out", type=Path, default=None)
    ap.add_argument("--estimates-out", type=Path, default=None, help="VA 無しも含む全推定")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--key-conf-min", type=float, default=0.05)
    ap.add_argument("--no-chordino", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="estimates-out 済み clip を飛ばして追記再開",
    )
    args = ap.parse_args()

    use_chordino = not args.no_chordino
    if use_chordino:
        ok = False
        try:
            import chord_extractor  # noqa: F401

            ok = True
        except ImportError:
            try:
                import vamp  # noqa: F401

                ok = True
            except ImportError:
                ok = _sonic_annotator_bin() is not None
        if not ok:
            print(
                "Chordino 用に vamp / sonic-annotator が無い → --no-chordino 相当",
                file=sys.stderr,
            )
            use_chordino = False

    va_map = load_song_va(args.annotations)
    print(f"VA songs={len(va_map)} chordino={use_chordino}", flush=True)

    files = list_audio(args.audio_dir)
    if args.limit > 0:
        files = files[: args.limit]

    done: set[str] = set()
    if args.resume and args.estimates_out and args.estimates_out.is_file():
        for line in args.estimates_out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line).get("clip") or ""))
            except json.JSONDecodeError:
                continue
        done.discard("")
    print(f"audio files={len(files)} resume_done={len(done)}", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.variants_out:
        args.variants_out.parent.mkdir(parents=True, exist_ok=True)
    if args.estimates_out:
        args.estimates_out.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume else "w"
    n_pair = 0
    n_var = 0
    n_skip = 0
    n_fail = 0
    with args.out_jsonl.open(mode, encoding="utf-8") as fout, (
        args.variants_out.open(mode, encoding="utf-8") if args.variants_out else open("/dev/null", "w")
    ) as fvar, (
        args.estimates_out.open(mode, encoding="utf-8") if args.estimates_out else open("/dev/null", "w")
    ) as fest:
        for i, path in enumerate(files, 1):
            if path.stem in done:
                n_skip += 1
                continue
            print(f"[{i}/{len(files)}] {path.name}", flush=True)
            try:
                row = process_one(
                    path,
                    va_map=va_map,
                    key_conf_min=args.key_conf_min,
                    use_chordino=use_chordino,
                )
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                fail_row = {
                    "source": "memo2496",
                    "clip": path.stem,
                    "path": str(path),
                    "va": (
                        {"valence": va_map[path.stem][0], "arousal": va_map[path.stem][1]}
                        if path.stem in va_map
                        else None
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                    "structure": {},
                    "_label_mask": {
                        "bpm": False,
                        "mode": False,
                        "key": False,
                        "prog": False,
                        "bpc": False,
                    },
                    "_meta": {"failed": True},
                }
                fest.write(json.dumps(fail_row, ensure_ascii=False) + "\n")
                fest.flush()
                print(f"  FAIL skip: {exc}", flush=True)
                continue
            fest.write(json.dumps(row, ensure_ascii=False) + "\n")
            fest.flush()
            meta = row.get("_meta") or {}
            if meta.get("variant") and meta.get("loop"):
                fvar.write(
                    json.dumps(
                        {
                            "clip": row["clip"],
                            "loop": meta["loop"],
                            "near": meta.get("near"),
                            "edit_distance": meta.get("edit_distance"),
                            "key": row["structure"].get("key"),
                            "mode": row["structure"].get("mode"),
                            "bpm": row["structure"].get("bpm"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fvar.flush()
                n_var += 1
            if row.get("va") is None:
                continue
            out = {
                "source": "memo2496",
                "clip": row["clip"],
                "va": row["va"],
                "va_source": row.get("va_source"),
                "structure": row["structure"],
                "_label_mask": row["_label_mask"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            n_pair += 1

    print(
        f"pairs+={n_pair} variants+={n_var} skipped={n_skip} failed={n_fail} -> {args.out_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
