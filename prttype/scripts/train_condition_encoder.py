"""条件エンコーダ学習（VA + 音楽条件スコア）。

JSONL:
  VA行: {"text", "va": {"valence","arousal"}, "source"}
  スコア行: {"text", "scores": {axis: 0..1}, "source"}

損失: VA = MSE + (1-CCC)、music = BCEWithLogits（scores があるバッチのみ）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from condition_encoder import ConditionEncoder, save_condition_ckpt  # noqa: E402
from data.music_condition.axis_tag_map import MUSIC_AXES  # noqa: E402


def _ccc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Concordance correlation（バッチ平均）。pred/target: (B,)。"""
    pred = pred.float()
    target = target.float()
    mp = pred.mean()
    mt = target.mean()
    vp = pred.var(unbiased=False)
    vt = target.var(unbiased=False)
    cov = ((pred - mp) * (target - mt)).mean()
    return (2 * cov) / (vp + vt + (mp - mt) ** 2 + 1e-8)


class ConditionJsonlDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.rows: list[dict] = []
        for p in paths:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        if not self.rows:
            raise SystemExit(f"空データ: {paths}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def collate(batch: list[dict], tokenizer, max_length: int = 256):
    texts = [str(b["text"]) for b in batch]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    has_va = torch.zeros(len(batch), dtype=torch.bool)
    va = torch.zeros(len(batch), 2, dtype=torch.float32)
    has_music = torch.zeros(len(batch), dtype=torch.bool)
    music = torch.zeros(len(batch), len(MUSIC_AXES), dtype=torch.float32)
    for i, b in enumerate(batch):
        if "va" in b and b["va"] is not None:
            has_va[i] = True
            va[i, 0] = float(b["va"]["valence"])
            va[i, 1] = float(b["va"]["arousal"])
        if "scores" in b and b["scores"] is not None:
            has_music[i] = True
            sc = b["scores"]
            for j, ax in enumerate(MUSIC_AXES):
                music[i, j] = float(sc.get(ax, 0.0))
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "has_va": has_va,
        "va": va,
        "has_music": has_music,
        "music": music,
    }


def train(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    paths = [Path(p) for p in args.jsonl]
    ds = ConditionJsonlDataset(paths)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = ConditionEncoder(backbone_name=args.backbone, n_axes=len(MUSIC_AXES))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    def _collate(batch):
        return collate(batch, tokenizer, args.max_length)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        total = 0.0
        n = 0
        for batch in loader:
            step += 1
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(ids, mask)
            loss = torch.tensor(0.0, device=device)
            # VA
            if batch["has_va"].any():
                hv = batch["has_va"].to(device)
                pred = out["va"][hv]
                tgt = batch["va"].to(device)[hv]
                mse = F.mse_loss(pred, tgt)
                ccc_v = _ccc(pred[:, 0], tgt[:, 0])
                ccc_a = _ccc(pred[:, 1], tgt[:, 1])
                loss = loss + mse + 0.5 * ((1 - ccc_v) + (1 - ccc_a))
            # music
            if batch["has_music"].any():
                hm = batch["has_music"].to(device)
                logits = out["music_logits"][hm]
                tgt_m = batch["music"].to(device)[hm]
                loss = loss + args.music_weight * F.binary_cross_entropy_with_logits(
                    logits, tgt_m
                )
            if float(loss.item()) == 0.0:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")
        avg = total / max(n, 1)
        print(f"epoch {epoch} done avg_loss={avg:.4f}")

    out_dir = Path(args.out)
    save_condition_ckpt(
        model,
        out_dir,
        tokenizer_name=args.backbone,
        meta={"epochs": args.epochs, "jsonl": [str(p) for p in paths]},
    )
    print(f"saved → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "va_encoder" / "jp_condition_last")
    ap.add_argument("--backbone", default="cl-tohoku/bert-base-japanese-v3")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--music-weight", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
