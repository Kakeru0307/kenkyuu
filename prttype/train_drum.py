"""ドラム専用 U-Net 学習（in=24 tonal+BPM+beat → out=1 drum、BCEWithLogitsLoss）。

入力: tonal11 + BPM1 + beat_type one-hot12 = 24ch。
--cvae を付けると CVAEUNet で学習（同じ入力でも出力が変わる多様生成）。
学習実行は Colab 側で行う。ローカルでのデフォルト実行は想定しない。
旧 in=12 ckpt はチャネル不一致で非互換。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim

from checkpoint_paths import part_ckpt_path, role_from_checkpoint_dir
from dataset_drum import get_drum_dataloader
from model import build_cvae, build_unet

SCRIPT_DIR = Path(__file__).resolve().parent


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """q(z|x,y) と標準正規 N(0, I) の KL ダイバージェンス（バッチ平均）。"""
    per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return per_sample.mean()


def kl_weight(epoch: int, *, beta: float, anneal_epochs: int) -> float:
    """β を 0 から beta まで線形に増やす（posterior collapse 回避）。"""
    if anneal_epochs <= 0:
        return beta
    return beta * min(1.0, epoch / anneal_epochs)


def train_drum(
    pairs_dir: Path,
    checkpoint_dir: Path,
    *,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    pos_weight: float = 20.0,
    resume: Path | None = None,
    encoder_weights: str | None = None,
    cvae: bool = False,
    latent_dim: int = 16,
    beta: float = 1.0,
    kl_anneal_epochs: int = 10,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    input_dir = pairs_dir / "input"
    target_dir = pairs_dir / "target"
    if not input_dir.is_dir() or not target_dir.is_dir():
        raise FileNotFoundError(
            f"pairs がありません: {pairs_dir} (input/ と target/ が必要)"
        )

    loader = get_drum_dataloader(
        input_dir,
        target_dir,
        batch_size=batch_size,
        shuffle=True,
    )
    sample_x, sample_y = next(iter(loader))
    in_ch = int(sample_x.shape[1])
    out_ch = int(sample_y.shape[1])
    print(f"device={device} in_channels={in_ch} out_channels={out_ch} batches={len(loader)} cvae={cvae}")

    if cvae:
        model = build_cvae(
            in_channels=in_ch,
            out_channels=out_ch,
            latent_dim=latent_dim,
            encoder_weights=encoder_weights,
        ).to(device)
    else:
        model = build_unet(
            in_channels=in_ch,
            out_channels=out_ch,
            encoder_weights=encoder_weights,
        ).to(device)

    if resume is not None and Path(resume).is_file():
        blob = torch.load(resume, map_location=device, weights_only=False)
        state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
        model.load_state_dict(state)
        print(f"resumed from {resume}")

    pw = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n = 0
        beta_t = kl_weight(epoch, beta=beta, anneal_epochs=kl_anneal_epochs)
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if cvae:
                logits, mu, logvar = model(x, y)
                recon = criterion(logits, y)
                kl = kl_divergence(mu, logvar)
                loss = recon + beta_t * kl
                total_recon += float(recon.item())
                total_kl += float(kl.item())
            else:
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n += 1

        mean_loss = total / max(1, n)
        if cvae:
            print(
                f"epoch {epoch:03d}  loss={mean_loss:.4f}  "
                f"recon={total_recon / max(1, n):.4f}  kl={total_kl / max(1, n):.4f}  beta={beta_t:.3f}"
            )
        else:
            print(f"epoch {epoch:03d}  loss={mean_loss:.4f}")

        role = role_from_checkpoint_dir(checkpoint_dir)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_type": "cvae_drum" if cvae else "unet_drum",
            "in_channels": in_ch,
            "out_channels": out_ch,
            "epochs": epoch,
            "lr": lr,
            "pos_weight": pos_weight,
            "loss": mean_loss,
            "role": role,
        }
        if cvae:
            ckpt["latent_dim"] = latent_dim
        last_path = part_ckpt_path(role, checkpoint_dir, cvae=cvae)
        torch.save(ckpt, last_path)
        if epoch % 5 == 0 or epoch == epochs:
            kind = "cvae" if cvae else "unet"
            torch.save(ckpt, checkpoint_dir / f"drum_{kind}_ep{epoch:03d}.pt")

    print(f"wrote {part_ckpt_path(role_from_checkpoint_dir(checkpoint_dir), checkpoint_dir, cvae=cvae)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train drum U-Net (in=24 → out=1)")
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "pairs" / "drum",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "drum",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=20.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--encoder-weights", type=str, default=None)
    parser.add_argument(
        "--cvae",
        action="store_true",
        help="確率的生成（CVAE）で学習する。同じ入力でも毎回異なる出力を得る",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help="CVAE の潜在次元数",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="CVAE の KL 項の重み（大きいほど多様・小さいほど再構成優先）",
    )
    parser.add_argument(
        "--kl-anneal-epochs",
        type=int,
        default=10,
        help="beta を 0 から線形に増やす epoch 数（posterior collapse 回避）",
    )
    args = parser.parse_args()

    train_drum(
        args.pairs_dir,
        args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pos_weight=args.pos_weight,
        resume=args.resume,
        encoder_weights=args.encoder_weights,
        cvae=args.cvae,
        latent_dim=args.latent_dim,
        beta=args.beta,
        kl_anneal_epochs=args.kl_anneal_epochs,
    )


if __name__ == "__main__":
    main()
