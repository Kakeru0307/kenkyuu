"""segmentation-models-pytorch ベースの U-Net と、確率的生成用の CVAE。"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

DEFAULT_ENCODER = "resnet18"
DEFAULT_LATENT_DIM = 16


def build_unet(
    in_channels: int = 12,
    out_channels: int = 11,
    encoder: str = DEFAULT_ENCODER,
    encoder_weights: str | None = None,
) -> nn.Module:
    """既定: 11 tonal + 1 BPM 条件 → 11 tonal。"""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,
    )


class PosteriorEncoder(nn.Module):
    """事後分布 q(z | x, y) を推定する小さな CNN。

    入力骨格 x と正解 y を結合したパッチから潜在ベクトルの
    平均 mu と対数分散 logvar を返す。学習時のみ使用する。
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        base: int = 32,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_mu = nn.Linear(base * 4, latent_dim)
        self.fc_logvar = nn.Linear(base * 4, latent_dim)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(torch.cat([x, y], dim=1))
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class CVAEUNet(nn.Module):
    """条件付き VAE。 骨格 x を条件に、潜在 z から演奏 y を生成する。

    - 学習時: q(z|x,y) から z をサンプリングし、decoder(x, z) で再構成。
    - 推論時: z ~ N(0, I) をサンプリングするため、同じ x でも出力が変わる。
    """

    def __init__(
        self,
        channels: int = 11,
        latent_dim: int = DEFAULT_LATENT_DIM,
        encoder: str = DEFAULT_ENCODER,
        encoder_weights: str | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.posterior = PosteriorEncoder(
            in_channels=channels * 2, latent_dim=latent_dim
        )
        self.decoder = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=channels + latent_dim,
            classes=channels,
            activation=None,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _inject(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """潜在ベクトル z を空間方向にブロードキャストして x に連結する。"""
        b, _, h, w = x.shape
        z_map = z.view(b, self.latent_dim, 1, 1).expand(b, self.latent_dim, h, w)
        return torch.cat([x, z_map], dim=1)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.posterior(x, y)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(self._inject(x, z))
        return out, mu, logvar

    def sample(
        self,
        x: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """推論用: 事前分布 N(0, I) から z を引いて生成する。"""
        if z is None:
            z = torch.randn(x.shape[0], self.latent_dim, device=x.device)
            z = z * temperature
        return self.decoder(self._inject(x, z))


def build_cvae(
    channels: int = 11,
    latent_dim: int = DEFAULT_LATENT_DIM,
    encoder: str = DEFAULT_ENCODER,
    encoder_weights: str | None = None,
) -> CVAEUNet:
    return CVAEUNet(
        channels=channels,
        latent_dim=latent_dim,
        encoder=encoder,
        encoder_weights=encoder_weights,
    )
