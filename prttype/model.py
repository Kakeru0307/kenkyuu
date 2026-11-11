"""segmentation-models-pytorch ベースの U-Net と、確率的生成用の CVAE。"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """事後分布 q(z | x, y) を推定する CNN。

    [改善] AdaptiveAvgPool2d(1) による全空間平均を廃止し、
    ピッチ方向のみ集約したあと時間方向を 1D conv で処理する。
    これにより「フレーズの時間的な輪郭（上昇型・下降型など）」を
    z に符号化できるようになり、乱数を変えると輪郭が変わる素地ができる。
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        base: int = 32,
        temporal_steps: int = 8,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.temporal_steps = temporal_steps
        # ピッチ方向を集約し時間方向を temporal_steps に揃える
        self.pitch_pool = nn.AdaptiveAvgPool2d((temporal_steps, 1))
        # 時間方向の特徴を軽量 1D conv で集約
        self.time_conv = nn.Sequential(
            nn.Conv1d(base * 4, base * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_mu = nn.Linear(base * 4, latent_dim)
        self.fc_logvar = nn.Linear(base * 4, latent_dim)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(torch.cat([x, y], dim=1))  # (B, C, T', P')
        h = self.pitch_pool(h).squeeze(-1)            # (B, C, temporal_steps)
        h = self.time_conv(h).squeeze(-1)             # (B, C)
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
# AdaIN 層（FiLM の改善版）
# ---------------------------------------------------------------------------

def _get_block_out_channels(block: nn.Module) -> int:
    """DecoderBlock 内の最後の Conv2d から出力チャンネル数を取得する。

    smp のバージョン差を吸収するため、サブモジュールを逆順に走査する。
    """
    for mod in reversed(list(block.modules())):
        if isinstance(mod, nn.Conv2d):
            return mod.out_channels
    raise ValueError(f"Conv2d が見つかりません: {type(block)}")


class AdaINLayer(nn.Module):
    """Adaptive Instance Normalization による z 条件付け。

    output = γ(z) * (feat - μ(feat)) / σ(feat) + β(z)

    FiLM との違い:
    - feat を正規化してから z を乗算するため、feat の絶対スケールに依存しない。
    - 層ごとのスケール差に関係なく z の影響量が安定する。
    - γ=1, β=0 の初期値が全層で一貫して「変調なし」として機能する。
    """

    def __init__(self, latent_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(latent_dim, channels)
        self.beta = nn.Linear(latent_dim, channels)
        # 初期値: γ=1, β=0 → 正規化後に変調なしと等価
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        mu = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
        g = self.gamma(z).view(b, c, 1, 1)
        bv = self.beta(z).view(b, c, 1, 1)
        return g * (feat - mu) / std + bv


# 後方互換エイリアス（旧コードが FiLMLayer を参照している場合）
FiLMLayer = AdaINLayer


class DensityHead(nn.Module):
    """z からノート密度（onset 比率）を予測する補助ヘッド。

    z が音楽的な変化量を符号化する動機を与える補助損失用。
    train.py で mu に対して呼び出し、MSE 損失を加算する。
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """0–1 の密度スコアを返す。"""
        return torch.sigmoid(self.fc(z))


# ---------------------------------------------------------------------------
# FiLM（AdaIN）+ GatedSkip CVAE U-Net
# ---------------------------------------------------------------------------

class FiLMCVAEUNet(nn.Module):
    """AdaIN conditioning + GatedSkip 付き CVAE U-Net。

    CVAEUNet との違い:
    - デコーダ入力は x のみ（z を concat しない）。
    - z は各 DecoderBlock 後に AdaIN として注入する（旧 FiLM より安定）。
    - エンコーダのスキップ接続を z でゲーティングする（推論時の骨格支配を防ぐ）。
    - PosteriorEncoder が時間的フレーズ差を保持するようになった。
    - DensityHead で z に音楽的性質を符号化する補助損失を持つ。

    PosteriorEncoder: cat(x, y) → 時間保持集約 → (mu, logvar) → z
    Decoder:          smp.Unet(in_channels) + AdaIN(z) @ each DecoderBlock
    Skip:             encoder skip ← sigmoid(gate(z)) でゲーティング
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

        # AdaIN 層: 各 DecoderBlock の出力チャンネル数を自動取得
        self.adain_layers = nn.ModuleList([
            AdaINLayer(latent_dim, _get_block_out_channels(block))
            for block in self.decoder.decoder.blocks
        ])

        # GatedSkip: エンコーダスキップ接続を z でゲーティング
        # encoder.out_channels = (in_ch, stage1_ch, ..., bottleneck_ch)
        # bottleneck は z で直接制御するためゲート不要。in_ch も除外。
        # チャンネル数 0 のエントリ（smp バージョン差）もスキップする。
        enc_out_ch = list(self.decoder.encoder.out_channels)
        skip_channels = [ch for ch in enc_out_ch[1:-1] if ch > 0]
        self.skip_gates = nn.ModuleList([
            nn.Linear(latent_dim, ch) for ch in skip_channels
        ])
        # 初期値: sigmoid(3.0) ≈ 0.95 → ほぼ全開。学習で徐々に絞られる
        for gate in self.skip_gates:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, 3.0)
        self._skip_channels = skip_channels

        # 補助損失ヘッド（z → ノート密度）
        self.density_head = DensityHead(latent_dim)

        # forward 中に z を渡すための一時領域（hook 経由で参照）
        self._z_current: torch.Tensor | None = None

        # 各 DecoderBlock の後段に AdaIN を差し込む
        for i, block in enumerate(self.decoder.decoder.blocks):
            block.register_forward_hook(self._make_adain_hook(i))

    def _make_adain_hook(self, idx: int):
        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
            if self._z_current is not None:
                return self.adain_layers[idx](output, self._z_current)
            return output
        return hook

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _gated_decoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """z によるスキップゲーティングを適用したデコーダフォワード。

        smp.Unet の encoder / decoder / segmentation_head を直接呼ぶことで、
        encoder の中間特徴（スキップ接続）に z ゲートをかけられる。
        """
        features = list(self.decoder.encoder(x))

        # スキップ接続（in_ch と bottleneck を除いた中間特徴）をゲーティング
        # features[0] = in_ch 特徴（smp が内部で除外）
        # features[1:-1] = 中間スキップ
        # features[-1] = bottleneck
        if self._z_current is not None and self.skip_gates:
            skip_indices = list(range(1, len(features) - 1))
            for gate_idx, feat_idx in enumerate(skip_indices):
                if gate_idx >= len(self.skip_gates):
                    break
                feat = features[feat_idx]
                g = torch.sigmoid(
                    self.skip_gates[gate_idx](self._z_current)
                ).view(-1, feat.shape[1], 1, 1)
                features[feat_idx] = feat * g

        decoder_output = self.decoder.decoder(features)  # smp>=0.5: list 1本
        return self.decoder.segmentation_head(decoder_output)

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
            out = self._gated_decoder_forward(x if x_dec is None else x_dec)
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
            out = self._gated_decoder_forward(x)
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
