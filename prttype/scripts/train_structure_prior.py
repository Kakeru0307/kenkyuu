"""Train structure prior on gated prior_pairs accept.jsonl.

Usage (local):
  python scripts/train_structure_prior.py \\
    --jsonl data/prior_pairs/manifests/accept.jsonl \\
    --checkpoint-dir checkpoints/structure_prior

Usage (colab_train cwd):
  python train_structure_prior.py \\
    --jsonl data/prior_pairs/accept.jsonl \\
    --checkpoint-dir checkpoints/structure_prior
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_va import wrime_to_va  # noqa: E402
from structure_prior import (  # noqa: E402
    BARS_PER_CHORD,
    BPM_HI,
    BPM_LO,
    ENERGIES,
    FEATURE_DIM,
    KEYS,
    MODES,
    PROGRESSIONS,
    StructurePriorNet,
    bpm_to_unit,
    encode_features,
    feature_dim,
    predict_structure,
)


def row_va(row: dict[str, Any]) -> tuple[float, float]:
    va = row.get("va")
    if isinstance(va, dict):
        return float(va.get("valence", 0.0)), float(va.get("arousal", 0.0))
    if isinstance(va, (list, tuple)) and len(va) >= 2:
        return float(va[0]), float(va[1])
    return wrime_to_va(row.get("emotion_wrime"))


def va_quadrant(v: float, a: float) -> str:
    hv = "pos" if v >= 0 else "neg"
    ha = "high" if a >= 0 else "low"
    return f"{hv}_{ha}"


def load_accept_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gate = (row.get("gate") or {}).get("status")
        if gate not in (None, "accept"):
            continue
        if gate is None and path.name != "accept.jsonl":
            continue
        st = row.get("structure") or {}
        if not st.get("progression") or st.get("bpm") is None or not st.get("key"):
            continue
        if st["progression"] not in PROGRESSIONS:
            continue
        if st["key"] not in KEYS:
            continue
        rows.append(row)
    if not rows:
        raise RuntimeError(f"学習可能な accept 行がありません: {path}")
    return rows


class PriorPairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.prog_i = {n: i for i, n in enumerate(PROGRESSIONS)}
        self.key_i = {n: i for i, n in enumerate(KEYS)}
        self.mode_i = {n: i for i, n in enumerate(MODES)}
        self.energy_i = {n: i for i, n in enumerate(ENERGIES)}
        self.bpc_i = {n: i for i, n in enumerate(BARS_PER_CHORD)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        st = row["structure"]
        energy = st.get("energy")
        if energy not in self.energy_i:
            bpm = float(st["bpm"])
            energy = "low" if bpm < 90 else ("high" if bpm >= 120 else "mid")
        mode = st.get("mode") or "major"
        if mode not in self.mode_i:
            mode = "major"
        bpc = int(st.get("bars_per_chord") or 1)
        if bpc not in self.bpc_i:
            bpc = 1 if bpc < 2 else 2

        x = encode_features(wrime=row.get("emotion_wrime"), va=row_va(row))
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "bpm": torch.tensor(bpm_to_unit(float(st["bpm"])), dtype=torch.float32),
            "energy": torch.tensor(self.energy_i[energy], dtype=torch.long),
            "mode": torch.tensor(self.mode_i[mode], dtype=torch.long),
            "key": torch.tensor(self.key_i[st["key"]], dtype=torch.long),
            "prog": torch.tensor(self.prog_i[st["progression"]], dtype=torch.long),
            "bpc": torch.tensor(self.bpc_i[bpc], dtype=torch.long),
        }


def stratified_split(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        v, a = row_va(r)
        lab = va_quadrant(v, a)
        by_label.setdefault(lab, []).append(i)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for _lab, idxs in by_label.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio))) if len(idxs) >= 5 else max(0, len(idxs) // 5)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    if not train_idx:
        train_idx, val_idx = val_idx[:-1] or val_idx, val_idx[-1:] if val_idx else []
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def batch_loss(
    model: StructurePriorNet,
    batch: dict[str, torch.Tensor],
    *,
    ce: nn.Module,
    mse: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(batch["x"])
    bpm_pred = torch.sigmoid(out["bpm"])
    losses = {
        "bpm": mse(bpm_pred, batch["bpm"]),
        "energy": ce(out["energy"], batch["energy"]),
        "mode": ce(out["mode"], batch["mode"]),
        "key": ce(out["key"], batch["key"]),
        "prog": ce(out["prog"], batch["prog"]),
        "bpc": ce(out["bpc"], batch["bpc"]),
    }
    total = (
        2.0 * losses["bpm"]
        + 1.5 * losses["energy"]
        + 1.0 * losses["mode"]
        + 0.8 * losses["key"]
        + 0.6 * losses["prog"]
        + 0.5 * losses["bpc"]
    )
    stats = {k: float(v.detach().item()) for k, v in losses.items()}
    stats["total"] = float(total.detach().item())
    return total, stats


@torch.inference_mode()
def evaluate(
    model: StructurePriorNet,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    if len(loader.dataset) == 0:
        return {}
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    totals: Counter[str] = Counter()
    n = 0
    correct = Counter()
    bpm_abs = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["x"])
        _, stats = batch_loss(model, batch, ce=ce, mse=mse)
        for k, v in stats.items():
            totals[k] += v
        pred_bpm = torch.sigmoid(out["bpm"])
        bpm_abs += float((pred_bpm - batch["bpm"]).abs().sum().item())
        for name in ("energy", "mode", "key", "prog", "bpc"):
            correct[name] += int((out[name].argmax(-1) == batch[name]).sum().item())
        n += batch["x"].shape[0]
    if n == 0:
        return {}
    nb = max(1, len(loader))
    metrics = {f"loss_{k}": totals[k] / nb for k in totals}
    metrics["mae_bpm"] = (bpm_abs / n) * (BPM_HI - BPM_LO)
    for name in ("energy", "mode", "key", "prog", "bpc"):
        metrics[f"acc_{name}"] = correct[name] / n
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WRIME+VA→structure prior")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "data" / "prior_pairs" / "manifests" / "accept.jsonl",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "structure_prior")
    parser.add_argument("--ckpt-name", type=str, default="structure_prior_last.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    jsonl = args.jsonl if args.jsonl.is_absolute() else (ROOT / args.jsonl)
    if not jsonl.is_file():
        alt = ROOT / "data" / "prior_pairs" / "accept.jsonl"
        if alt.is_file():
            jsonl = alt
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else (ROOT / args.checkpoint_dir)
    rows = load_accept_rows(jsonl)
    print(f"rows={len(rows)} from {jsonl}")
    print("va_quadrant", dict(Counter(va_quadrant(*row_va(r)) for r in rows)))

    train_idx, val_idx = stratified_split(rows, val_ratio=args.val_ratio, seed=args.seed)
    ds = PriorPairDataset(rows)
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=min(args.batch_size, max(1, len(train_idx))),
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=min(args.batch_size, max(1, len(val_idx) or 1)),
        shuffle=False,
    )
    print(f"split train={len(train_idx)} val={len(val_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = feature_dim()
    assert in_dim == FEATURE_DIM
    model = StructurePriorNet(in_dim, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = batch_loss(model, batch, ce=ce, mse=mse)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            steps += 1
        model.eval()
        val_m = evaluate(model, val_loader, device=device)
        train_loss = running / max(1, steps)
        val_loss = val_m.get("loss_total", train_loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                f"acc_energy={val_m.get('acc_energy', 0):.2f}  "
                f"acc_mode={val_m.get('acc_mode', 0):.2f}  "
                f"mae_bpm={val_m.get('mae_bpm', 0):.1f}  "
                f"acc_prog={val_m.get('acc_prog', 0):.2f}"
            )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = ckpt_dir / args.ckpt_name
    blob = {
        "model_state_dict": model.state_dict(),
        "meta": {
            "in_dim": in_dim,
            "hidden": args.hidden,
            "feature": "wrime8+va2",
            "epochs": args.epochs,
            "lr": args.lr,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "best_val_loss": best_val,
            "jsonl": str(jsonl),
            "progressions": list(PROGRESSIONS),
            "keys": list(KEYS),
            "modes": list(MODES),
            "energies": list(ENERGIES),
            "bpm_range": [BPM_LO, BPM_HI],
        },
    }
    torch.save(blob, ckpt_path)
    print(f"wrote {ckpt_path}")

    model.eval()
    shown: set[str] = set()
    for row in rows:
        v, a = row_va(row)
        lab = va_quadrant(v, a)
        if lab in shown:
            continue
        shown.add(lab)
        pred = predict_structure(
            model,
            wrime=row.get("emotion_wrime"),
            va=(v, a),
            device=device,
        )
        gold = row["structure"]
        print(
            f"sample[{lab} V={v:+.2f} A={a:+.2f}] pred bpm={pred.bpm} energy={pred.energy} "
            f"mode={pred.mode} prog={pred.progression} | gold bpm={gold['bpm']} "
            f"energy={gold.get('energy')} mode={gold.get('mode')} prog={gold['progression']}"
        )


if __name__ == "__main__":
    main()
