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
    """条件付き VAE（後方互換維持用・drum など旧モデル向け）。

    - 学習時: q(z|x,y) から z をサンプリングし、decoder(cat(x,z)) で再構成。
    - 推論時: z ~ N(0, I) をサンプリング。

    新規学習には FiLMCVAEUNet を推奨。
    PosteriorEncoder: cat(x, y) = in_channels + out_channels
    Decoder:          cat(x, z) = in_channels + latent_dim → out_channels
    """

    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 11,
        latent_dim: int = DEFAULT_LATENT_DIM,
        encoder: str = DEFAULT_ENCODER,
        encoder_weights: str | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.posterior = PosteriorEncoder(
            in_channels=in_channels + out_channels, latent_dim=latent_dim
        )
        self.decoder = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels + latent_dim,
            classes=out_channels,
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
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_dec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.posterior(x, y)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(self._inject(x if x_dec is None else x_dec, z))
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


# ---------------------------------------------------------------------------
# FiLM CVAE U-Net
# ---------------------------------------------------------------------------

def _get_block_out_channels(block: nn.Module) -> int:
    """DecoderBlock 内の最後の Conv2d から出力チャンネル数を取得する。

    smp のバージョン差を吸収するため、サブモジュールを逆順に走査する。
    """
    for mod in reversed(list(block.modules())):
        if isinstance(mod, nn.Conv2d):
            return mod.out_channels
    raise ValueError(f"Conv2d が見つかりません: {type(block)}")


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: z から γ, β を生成して feature map を変調する。

    output = γ(z) * feat + β(z)
    """

    def __init__(self, latent_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(latent_dim, channels)
        self.beta = nn.Linear(latent_dim, channels)
        # 初期値: γ=1, β=0 → 学習初期は変調なしと等価
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        g = self.gamma(z).view(-1, feat.shape[1], 1, 1)
        b = self.beta(z).view(-1, feat.shape[1], 1, 1)
        return g * feat + b


class FiLMCVAEUNet(nn.Module):
    """FiLM conditioning 付き CVAE U-Net。

    CVAEUNet との違い:
    - デコーダ入力は x のみ（z を concat しない）。in_channels = in_channels。
    - z は各 DecoderBlock の後に FiLM（γ*feat+β）として全階層に注入する。
    - スキップ接続は維持するため再構成精度は保ちつつ z を無視できない構造になる。

    PosteriorEncoder: cat(x, y) → (mu, logvar) → z
    Decoder:          smp.Unet(in_channels) + FiLM(z) @ each DecoderBlock
    """

    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 11,
        latent_dim: int = DEFAULT_LATENT_DIM,
        encoder: str = DEFAULT_ENCODER,
        encoder_weights: str | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        self.posterior = PosteriorEncoder(
            in_channels=in_channels + out_channels, latent_dim=latent_dim
        )
        # デコーダ入力は x のみ（CVAEUNet と異なり latent_dim を足さない）
        self.decoder = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

        # 各 DecoderBlock の出力チャンネル数を自動取得して FiLM 層を生成
        self.film_layers = nn.ModuleList([
            FiLMLayer(latent_dim, _get_block_out_channels(block))
            for block in self.decoder.decoder.blocks
        ])

        # forward 中に z を渡すための一時領域（hook 経由で参照）
        self._z_current: torch.Tensor | None = None

        # 各 DecoderBlock の後段に FiLM を差し込む
        for i, block in enumerate(self.decoder.decoder.blocks):
            block.register_forward_hook(self._make_film_hook(i))

    def _make_film_hook(self, idx: int):
        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
            if self._z_current is not None:
                return self.film_layers[idx](output, self._z_current)
            return output
        return hook

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_dec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """学習時フォワード。x_dec を指定するとデコーダに渡す骨格をマスクできる。"""
        mu, logvar = self.posterior(x, y)
        z = self.reparameterize(mu, logvar)
        self._z_current = z
        try:
            out = self.decoder(x if x_dec is None else x_dec)
        finally:
            self._z_current = None
        return out, mu, logvar

    def sample(
        self,
        x: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """推論用: z ~ N(0, I) からサンプリングして生成する。"""
        if z is None:
            z = torch.randn(x.shape[0], self.latent_dim, device=x.device) * temperature
        self._z_current = z
        try:
            out = self.decoder(x)
        finally:
            self._z_current = None
        return out


def build_cvae(
    in_channels: int = 11,
    out_channels: int = 11,
    latent_dim: int = DEFAULT_LATENT_DIM,
    encoder: str = DEFAULT_ENCODER,
    encoder_weights: str | None = None,
    use_film: bool = False,
) -> CVAEUNet | FiLMCVAEUNet:
    """CVAE を構築する。use_film=True で FiLMCVAEUNet を返す。"""
    cls = FiLMCVAEUNet if use_film else CVAEUNet
    return cls(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_dim=latent_dim,
        encoder=encoder,
        encoder_weights=encoder_weights,
    )
