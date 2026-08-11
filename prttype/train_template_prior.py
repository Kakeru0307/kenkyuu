"""template prior MLP 学習スクリプト。

generate_template_pairs.py で生成した JSONL を読み込み、
CrossEntropyLoss で TemplatePriorNet を学習する。
生成フェーズの BPM_MIN / BPM_MAX と一致させること。

使い方:
    python train_template_prior.py
    python train_template_prior.py --epochs 100 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from checkpoint_paths import TEMPLATE_PRIOR_NAME
from template_prior import TemplatePriorNet

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PAIRS = SCRIPT_DIR / "data" / "template_pairs" / "pairs.jsonl"
DEFAULT_CKPT_DIR = SCRIPT_DIR / "checkpoints" / "template_prior"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train(
    pairs_path: Path,
    checkpoint_dir: Path,
    *,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 64,
) -> None:
    records = load_jsonl(pairs_path)
    print(f"学習データ: {len(records)} 件")

    xs = torch.tensor(
        [[r["valence"], r["arousal"], r["bpm_unit"]] for r in records],
        dtype=torch.float32,
    )
    ys = torch.tensor([r["label"] for r in records], dtype=torch.long)

    n_templates = int(ys.max().item()) + 1
    dataset = TensorDataset(xs, ys)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TemplatePriorNet(n_templates=n_templates, hidden=hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n += 1
        if epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:03d}  loss={total / max(1, n):.4f}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / TEMPLATE_PRIOR_NAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_templates": n_templates,
            "hidden": hidden,
            "epochs": epochs,
            "bpm_unit": 0.5,
        },
        ckpt_path,
    )
    print(f"保存: {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="template prior MLP 学習")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    train(
        args.pairs,
        args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
    )


if __name__ == "__main__":
    main()
