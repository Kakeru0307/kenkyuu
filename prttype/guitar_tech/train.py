"""Train technique frame autoencoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .constants import CHECKPOINT_DIR, DATASET_DIR
from .model import TechFrameAE


def main() -> None:
    parser = argparse.ArgumentParser(description="Train guitar tech frame AE")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ds_dir = args.dataset_dir
    blob = np.load(ds_dir / "frames.npz")
    x = torch.from_numpy(blob["x"]).float()
    y = torch.from_numpy(blob["y"]).long()
    dictionary = json.loads((ds_dir / "dictionary.json").read_text(encoding="utf-8"))
    n_tech = len(dictionary["tech_names"])

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=min(args.batch_size, len(x)),
        shuffle=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TechFrameAE(
        n_tech, hidden=args.hidden, z_dim=args.z_dim
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb, yb)
            # pitch / gate / pb
            loss = (
                F.mse_loss(pred[..., 0], xb[..., 0])
                + F.mse_loss(pred[..., 1], xb[..., 1])
                + 0.5 * F.mse_loss(pred[..., 2], xb[..., 2])
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        avg = total / max(1, n)
        if avg < best:
            best = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d}  loss={avg:.5f}  best={best:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_dir = args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "guitar_tech_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "meta": {
                "n_tech": n_tech,
                "tech_names": dictionary["tech_names"],
                "hidden": args.hidden,
                "z_dim": args.z_dim,
                "best_loss": best,
                "epochs": args.epochs,
                "dataset": str(ds_dir),
                "n_train": int(x.shape[0]),
            },
        },
        path,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
