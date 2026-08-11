"""WRIME 8感情スコア → prior 4ラベル 分類器の学習。

入力: data/prior_pairs/manifests/accept.jsonl
出力: checkpoints/emotion/emotion_clf.pt
      {'mu': ..., 'sd': ..., 'state_dict': ..., 'labels': (...)}

8次元 z-score → 16 → ReLU → 4 で学習し、holdout で確認する。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ACCEPT_JSONL = SCRIPT_DIR / "data" / "prior_pairs" / "manifests" / "accept.jsonl"
OUT = SCRIPT_DIR / "checkpoints" / "emotion" / "emotion_clf.pt"

WRIME_KEYS = ("joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust")
LABELS = ("joy", "sadness", "calm", "tension")

EPOCHS = 800
LR = 0.05
HIDDEN = 16
HOLDOUT = 0.2
SEED = 42
MIN_CONFIDENCE = 0.50  # softmax 最大値がこれ未満なら na


def _load_data():
    rows = [json.loads(l) for l in ACCEPT_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    X = [[r["emotion_wrime"][k] for k in WRIME_KEYS] for r in rows]
    y = [LABELS.index(r["emotion_target"]) for r in rows]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train(*, epochs: int = EPOCHS, lr: float = LR, seed: int = SEED, holdout: float = HOLDOUT):
    X, y = _load_data()
    mu = X.mean(0)
    sd = X.std(0).clamp_min(1e-6)
    Z = (X - mu) / sd

    torch.manual_seed(seed)
    random.seed(seed)
    idx = torch.randperm(len(y))
    n_te = max(1, int(len(y) * holdout))
    tr, te = idx[n_te:], idx[:n_te]

    model = nn.Sequential(
        nn.Linear(len(WRIME_KEYS), HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, len(LABELS)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Z[tr]), y[tr])
        loss.backward()
        opt.step()
        if (epoch + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                atr = (model(Z[tr]).argmax(1) == y[tr]).float().mean().item()
                ate = (model(Z[te]).argmax(1) == y[te]).float().mean().item()
            print(f"  epoch {epoch+1:4d} loss={loss.item():.4f} train={atr:.3f} holdout={ate:.3f}")

    model.eval()
    with torch.no_grad():
        atr = (model(Z[tr]).argmax(1) == y[tr]).float().mean().item()
        ate = (model(Z[te]).argmax(1) == y[te]).float().mean().item()
    print(f"final: train={atr:.3f} holdout={ate:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mu": mu,
        "sd": sd,
        "state_dict": model.state_dict(),
        "labels": LABELS,
        "hidden": HIDDEN,
        "min_confidence": MIN_CONFIDENCE,
    }, OUT)
    print(f"saved: {OUT}")
    return model, mu, sd


if __name__ == "__main__":
    train()
