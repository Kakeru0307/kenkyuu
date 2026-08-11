"""VelocityMLP 学習スクリプト。

Guitar-TECHS MIDI から各テクニックの (pitch, duration, velocity) を抽出し、
per-note 特徴量 → velocity を学習する。

使い方:
  python train_velocity_mlp.py \
      --stats data/guitar_techs_stats.json \
      --out checkpoints/velocity_mlp/velocity_mlp_last.pth \
      --epochs 100
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from velocity_mlp import FEATURE_DIM, VelocityMLP, _make_feature

# テクニックID
TECHNIQUE_ID_MAP = {
    "normal": 0,
    "bend": 1,
    "vibrato": 2,
    "palm_mute": 3,
    "harmonic": 2,        # harmonic → vibrato クラスに分類
    "pinch_harmonic": 3,  # pinch_harmonic → palm_mute クラスに分類
}

BPM_RANGE = (80, 140)


def _build_dataset_from_stats(
    stats_all: dict,
    n_samples_per_tech: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """統計データから合成 (feature, target_velocity) ペアを生成する。"""
    random.seed(seed)
    np.random.seed(seed)

    features_list = []
    targets_list = []

    for tech_name, tech_id in TECHNIQUE_ID_MAP.items():
        if tech_name not in stats_all:
            continue

        stats = stats_all[tech_name]
        if stats.get("n_notes", 0) == 0:
            continue

        v = stats["velocity"]
        d = stats["duration_beats"]
        p = stats["pitch"]

        raw_velocities = stats.get("raw", {}).get("velocities", [])
        raw_pitches = stats.get("raw", {}).get("pitches", [])
        raw_durations = stats.get("raw", {}).get("duration_beats", [])

        for _ in range(n_samples_per_tech):
            # 実データからランダムサンプリング（あれば）、なければガウス
            if raw_velocities:
                velocity = random.choice(raw_velocities)
                pitch = random.choice(raw_pitches)
                duration_beats = random.choice(raw_durations)
            else:
                velocity = int(np.clip(random.gauss(v["mean"], v["stdev"]), 0, 127))
                pitch = int(np.clip(random.gauss(p["mean"], p["stdev"]), 0, 127))
                duration_beats = max(0.125, random.gauss(d["mean"], d["stdev"]))

            feat = _make_feature(tech_id, pitch, duration_beats)
            features_list.append(feat)
            targets_list.append(velocity / 127.0)

    X = np.stack(features_list).astype(np.float32)
    y = np.array(targets_list, dtype=np.float32).reshape(-1, 1)
    return X, y


def train(
    stats_path: Path,
    out_path: Path,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    stats_all = json.loads(stats_path.read_text(encoding="utf-8"))
    X, y = _build_dataset_from_stats(stats_all, seed=seed)
    print(f"データ数: {len(X)}")

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    dataset = TensorDataset(X_t, y_t)

    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VelocityMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val = float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item()

        scheduler.step()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train={avg_train:.6f} | val={avg_val:.6f}"
            )

        if avg_val < best_val:
            best_val = avg_val
            model.save(out_path)

    print(f"\n学習完了。最良モデル: {out_path} (val_loss={best_val:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="VelocityMLP 学習")
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "guitar_techs_stats.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints"
        / "velocity_mlp"
        / "velocity_mlp_last.pth",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        stats_path=args.stats,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
