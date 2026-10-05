"""VGMIDI の連続VAを MIDI 特徴から線形回帰し、EMOPIA へ転写する。

特徴: notes/sec, mean velocity, major_flag, mode_known
象限符号は EMOPIA の Q1〜Q4 を使い、大きさのみ回帰予測を採用。
転写失敗（特徴抽出失敗・異常値）は呼び出し側でスキップする（±0.6 フォールバックなし）。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from midi_emotion_features import extract_midi_features, feature_vector

Q_SIGNS: dict[str, tuple[float, float]] = {
    "Q1": (+1.0, +1.0),
    "Q2": (-1.0, +1.0),
    "Q3": (-1.0, -1.0),
    "Q4": (+1.0, -1.0),
}

_MIN_MAG = 0.05


def _piece_va_means(obj: dict) -> dict[str, tuple[float, float]]:
    anns = obj["annotations"]
    pieces = obj["pieces"]
    by: dict[str, list[dict]] = defaultdict(list)
    for key, val in anns.items():
        pid = key.rsplit("_", 1)[0]
        by[pid].append(val)
    out: dict[str, tuple[float, float]] = {}
    for pid, rows in by.items():
        if pid not in pieces:
            continue
        mv = sum(sum(r["valence"]) / len(r["valence"]) for r in rows) / len(rows)
        ma = sum(sum(r["arousal"]) / len(r["arousal"]) for r in rows) / len(rows)
        stem = Path(pieces[pid]["midi"]).stem
        out[stem] = (float(mv), float(ma))
    return out


def fit_va_regressor(
    *,
    midi_dir: Path,
    json_paths: list[Path],
    ridge: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """X @ W ≈ Y。戻り値 (W[d+1,2] with bias row, train_stats unused)."""
    cont: dict[str, tuple[float, float]] = {}
    for jp in json_paths:
        cont.update(_piece_va_means(json.loads(jp.read_text(encoding="utf-8"))))

    xs: list[list[float]] = []
    ys: list[list[float]] = []
    midi_by = {p.stem: p for p in midi_dir.glob("*.mid")}
    for stem, (v, a) in cont.items():
        path = midi_by.get(stem)
        if path is None:
            continue
        feat = extract_midi_features(path)
        if feat is None:
            continue
        xs.append(feature_vector(feat))
        ys.append([v, a])
    if len(xs) < 8:
        raise RuntimeError(f"VA転写の学習サンプルが足りません: {len(xs)}")

    X = np.asarray(xs, dtype=np.float64)
    Y = np.asarray(ys, dtype=np.float64)
    # 標準化して安定化
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    Xn = (X - x_mean) / x_std
    ones = np.ones((Xn.shape[0], 1), dtype=np.float64)
    A = np.concatenate([Xn, ones], axis=1)
    d = A.shape[1]
    W = np.linalg.solve(A.T @ A + ridge * np.eye(d), A.T @ Y)
    meta = np.stack([x_mean, x_std], axis=0)  # 2 x feat
    return W, meta


def predict_va(feat: dict[str, float], W: np.ndarray, meta: np.ndarray) -> tuple[float, float]:
    x = np.asarray(feature_vector(feat), dtype=np.float64)
    x_mean, x_std = meta[0], meta[1]
    xn = (x - x_mean) / x_std
    a = np.concatenate([xn, np.array([1.0])])
    pred = a @ W
    return float(pred[0]), float(pred[1])


def apply_quadrant_signs(v_mag: float, a_mag: float, q: str) -> tuple[float, float]:
    sv, sa = Q_SIGNS[q]
    mv = max(_MIN_MAG, abs(v_mag))
    ma = max(_MIN_MAG, abs(a_mag))
    # 回帰値は符号付き。大きさは |pred|、符号は象限。
    return sv * mv, sa * ma


def transfer_va_for_midi(
    midi_path: Path,
    *,
    q: str,
    W: np.ndarray,
    meta: np.ndarray,
) -> tuple[float, float] | None:
    feat = extract_midi_features(midi_path)
    if feat is None:
        return None
    pv, pa = predict_va(feat, W, meta)
    return apply_quadrant_signs(pv, pa, q)
