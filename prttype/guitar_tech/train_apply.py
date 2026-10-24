"""Train note→technique classifier on apply dataset labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .apply_model import TechApplyNet
from .carriers import NoteEvent
from .constants import APPLY_DATASET_DIR, CHECKPOINT_DIR
from .features import NOTE_FEATURE_DIM, note_features
from .params import TECH_NAMES, TECH_TO_ID

_NONE_ID = TECH_TO_ID["none"]


def load_xy(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    labels_dir = dataset_dir / "labels"
    xs: list[np.ndarray] = []
    ys: list[int] = []
    unknown = 0
    for path in sorted(labels_dir.glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        notes = [
            NoteEvent(
                pitch=int(n["pitch"]),
                start=float(n["start"]),
                end=float(n["end"]),
                velocity=int(n["velocity"]),
            )
            for n in blob["notes"]
        ]
        feats = note_features(notes)
        for lb in blob["labels"]:
            idx = int(lb["note_index"])
            tech = str(lb["tech"])
            if idx < 0 or idx >= len(feats):
                continue
            tid = TECH_TO_ID.get(tech)
            if tid is None:
                unknown += 1
                continue
            xs.append(feats[idx])
            ys.append(tid)
    if unknown:
        print(f"skipped unknown tech labels: {unknown}")
    if not xs:
        raise RuntimeError(f"no labels in {labels_dir}")
    return np.stack(xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train guitar tech apply classifier")
    parser.add_argument("--dataset-dir", type=Path, default=APPLY_DATASET_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x, y = load_xy(args.dataset_dir)
    print(f"notes={len(y)} techs={len(TECH_NAMES)}")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_ratio))
    if n_val >= len(idx):
        n_val = max(1, len(idx) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if len(train_idx) == 0:
        raise RuntimeError(f"train set empty (n={len(y)}); need more labels")

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    # class-balance sampling (none is majority)
    counts = np.bincount(y[train_idx], minlength=len(TECH_NAMES)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts[y[train_idx]]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(w).double(),
        num_samples=len(train_idx),
        replacement=True,
    )
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(args.batch_size, len(train_idx)),
        sampler=sampler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TechApplyNet(len(TECH_NAMES), hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        model.eval()
        with torch.no_grad():
            logits = model(x_val.to(device))
            pred = logits.argmax(-1).cpu()
            acc = float((pred == y_val).float().mean().item())
            # non-none recall-ish
            mask = y_val != _NONE_ID
            acc_pos = (
                float((pred[mask] == y_val[mask]).float().mean().item())
                if mask.any()
                else 0.0
            )
        train_loss = total / max(1, n)
        if acc_pos >= best:
            best = acc_pos
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d} loss={train_loss:.4f} "
                f"acc={acc:.3f} acc_nonzero={acc_pos:.3f} best_nz={best:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_dir = args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "guitar_tech_apply_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "meta": {
                "tech_names": list(TECH_NAMES),
                "hidden": args.hidden,
                "feature_dim": int(x.shape[1]),
                "note_feature_dim": NOTE_FEATURE_DIM,
                "n_train_notes": int(len(train_idx)),
                "n_val_notes": int(len(val_idx)),
                "best_acc_nonzero": best,
                "epochs": args.epochs,
                "dataset": str(args.dataset_dir),
            },
        },
        path,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
