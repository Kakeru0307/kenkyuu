"""smp U-Net の学習スクリプト。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim

from checkpoint_paths import part_ckpt_path, role_from_checkpoint_dir
from dataset import PatchPairDataset, SinglePatchDataset, get_dataloader
from model import DEFAULT_LATENT_DIM, build_cvae, build_unet
from program_utils import GUITAR_PROGRAM

SCRIPT_DIR = Path(__file__).resolve().parent
GUITAR_CATEGORY = GUITAR_PROGRAM // 8


def weighted_mse_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: float = 10.0,
    onset_weight: float = 1.0,
    midbar_onset_bonus: float = 1.0,
    ticks_per_bar: int = 16,
) -> torch.Tensor:
    """音符セル（非ゼロ）に重みを付けて、無音ばかり予測するのを防ぐ。

    pos_weight: 無音以外セル全般の重み（既存）
    onset_weight: onset(正規化値0.5)セルへの追加倍率 — >1 でストローク再現を強化
    midbar_onset_bonus: 小節頭以外の onset への追加倍率 — >1 で中拍打鍵を強化
    デフォルトはすべて 1.0 で既存動作と同一。出力レンジは raw 回帰値のまま変えない。
    """
    onset_mask = targets == 0.5  # 正規化後の値1（打ち込み onset）
    weights = torch.ones_like(targets)
    weights[targets > 0] = pos_weight
    if onset_weight != 1.0:
        weights[onset_mask] *= onset_weight
    if midbar_onset_bonus != 1.0:
        time_idx = torch.arange(targets.shape[-2], device=targets.device)
        midbar = (time_idx % ticks_per_bar != 0).view(1, 1, -1, 1).expand_as(onset_mask)
        weights[onset_mask & midbar] *= midbar_onset_bonus
    return (weights * (outputs - targets) ** 2).mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """q(z|x,y) と標準正規 N(0, I) の KL ダイバージェンス（バッチ平均）。"""
    per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return per_sample.mean()


def kl_weight(epoch: int, *, beta: float, anneal_epochs: int) -> float:
    """β を 0 から beta まで線形に増やす（posterior collapse 回避）。"""
    if anneal_epochs <= 0:
        return beta
    return beta * min(1.0, epoch / anneal_epochs)


def resolve_pair_dirs(
    data_dir: Path | None,
    pairs_dir: Path | None,
) -> tuple[Path, Path | None]:
    if pairs_dir is not None:
        input_dir = pairs_dir / "input"
        target_dir = pairs_dir / "target"
        if not input_dir.is_dir():
            raise FileNotFoundError(f"input ディレクトリがありません: {input_dir}")
        if not target_dir.is_dir():
            raise FileNotFoundError(f"target ディレクトリがありません: {target_dir}")
        return input_dir, target_dir

    if data_dir is None:
        raise ValueError("--data-dir または --pairs-dir を指定してください")

    return data_dir, None


def train(
    data_dir: Path | None,
    checkpoint_dir: Path,
    *,
    pairs_dir: Path | None = None,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    overfit_single: bool = False,
    encoder_weights: str | None = None,
    pos_weight: float = 10.0,
    onset_weight: float = 1.0,
    midbar_onset_bonus: float = 1.0,
    resume: Path | None = None,
    cvae: bool = False,
    latent_dim: int = DEFAULT_LATENT_DIM,
    beta: float = 1.0,
    kl_anneal_epochs: int = 10,
    require_power_cond: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    input_dir, target_dir = resolve_pair_dirs(data_dir, pairs_dir)

    if overfit_single:
        patch_files = sorted(input_dir.rglob("*_tonal.npy"))
        if not patch_files:
            raise FileNotFoundError(f"パッチが見つかりません: {input_dir}")
        dataset = SinglePatchDataset(patch_files[0], length=32)
    else:
        dataset = PatchPairDataset(
            input_dir,
            target_dir,
            require_power_cond=require_power_cond,
        )
    print(f"学習パッチ数: {len(dataset)}")
    sample_input, _ = dataset[0]
    input_channels = int(sample_input.shape[0])
    print(f"入力チャンネル数: {input_channels}")

    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)
    if cvae:
        model = build_cvae(
            in_channels=input_channels,
            out_channels=11,
            latent_dim=latent_dim, encoder_weights=encoder_weights
        ).to(device)
    else:
        model = build_unet(
            in_channels=input_channels,
            encoder_weights=encoder_weights,
        ).to(device)
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        prev_epochs = checkpoint.get("epochs", "?")
        print(f"再開: {resume} から重みを読み込み（前 epochs={prev_epochs}）")
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        beta_t = kl_weight(epoch, beta=beta, anneal_epochs=kl_anneal_epochs)
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            if cvae:
                outputs, mu, logvar = model(inputs, targets)
                recon = weighted_mse_loss(
                    outputs, targets,
                    pos_weight=pos_weight,
                    onset_weight=onset_weight,
                    midbar_onset_bonus=midbar_onset_bonus,
                )
                kl = kl_divergence(mu, logvar)
                loss = recon + beta_t * kl
                total_recon += recon.item()
                total_kl += kl.item()
            else:
                outputs = model(inputs)
                loss = weighted_mse_loss(
                    outputs, targets,
                    pos_weight=pos_weight,
                    onset_weight=onset_weight,
                    midbar_onset_bonus=midbar_onset_bonus,
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        n = len(dataloader)
        avg_loss = total_loss / n
        if cvae:
            print(
                f"epoch {epoch}/{epochs}  loss={avg_loss:.6f}  "
                f"recon={total_recon / n:.6f}  kl={total_kl / n:.6f}  beta={beta_t:.3f}"
            )
        else:
            print(f"epoch {epoch}/{epochs}  loss={avg_loss:.6f}")

    role = role_from_checkpoint_dir(checkpoint_dir)
    ckpt_path = part_ckpt_path(role, checkpoint_dir, cvae=cvae)
    payload = {
        "model_state_dict": model.state_dict(),
        "epochs": epochs,
        "lr": lr,
        "model_type": "cvae" if cvae else "unet",
        "in_channels": input_channels,
        "out_channels": 11,
        "role": role,
    }
    if cvae:
        payload["latent_dim"] = latent_dim
    torch.save(payload, ckpt_path)
    print(f"チェックポイント保存: {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="単一フォルダ学習（input=target）",
    )
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=None,
        help="prepare_dataset.py で作った pairs ルート（input/ と target/）",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=SCRIPT_DIR / "checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--overfit-single",
        action="store_true",
        help="1 パッチで過学習テスト",
    )
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument(
        "--onset-weight",
        type=float,
        default=1.0,
        help="onset(値1)セルへの追加重み倍率。fine-tune 推奨値: 2.0",
    )
    parser.add_argument(
        "--midbar-onset-bonus",
        type=float,
        default=1.0,
        help="小節頭以外の onset への追加重み倍率。fine-tune 推奨値: 3.0",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="既存チェックポイントから重みを読み込んで学習を継続する",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default=None,
        help="例: imagenet（ネットワーク接続が必要）",
    )
    parser.add_argument(
        "--require-power-cond",
        action="store_true",
        help="全inputに*_power.npyを必須化（13ch lead学習用）",
    )
    parser.add_argument(
        "--cvae",
        action="store_true",
        help="確率的生成（条件付きVAE）で学習する。同じ入力から多様な出力を得る",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=DEFAULT_LATENT_DIM,
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

    data_dir = args.data_dir
    if data_dir is None and args.pairs_dir is None:
        data_dir = (
            SCRIPT_DIR / "stash" / "data_legacy" / "patches" / "test1"
        )

    train(
        data_dir,
        args.checkpoint_dir,
        pairs_dir=args.pairs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        overfit_single=args.overfit_single,
        encoder_weights=args.encoder_weights,
        pos_weight=args.pos_weight,
        onset_weight=args.onset_weight,
        midbar_onset_bonus=args.midbar_onset_bonus,
        resume=args.resume,
        cvae=args.cvae,
        latent_dim=args.latent_dim,
        beta=args.beta,
        kl_anneal_epochs=args.kl_anneal_epochs,
        require_power_cond=args.require_power_cond,
    )


if __name__ == "__main__":
    main()
