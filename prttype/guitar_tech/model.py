"""Class-conditional autoencoder for guitar technique frames."""

from __future__ import annotations

import torch
import torch.nn as nn

from .constants import FEATURE_DIM, N_FRAMES


class TechFrameAE(nn.Module):
    """frames → z ; (z, tech) → frames."""

    def __init__(
        self,
        n_tech: int,
        *,
        n_frames: int = N_FRAMES,
        feat_dim: int = FEATURE_DIM,
        z_dim: int = 16,
        hidden: int = 128,
        tech_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_frames = int(n_frames)
        self.feat_dim = int(feat_dim)
        self.z_dim = int(z_dim)
        flat = n_frames * feat_dim
        self.tech_emb = nn.Embedding(n_tech, tech_dim)
        self.encoder = nn.Sequential(
            nn.Linear(flat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + tech_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, flat),
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames.reshape(frames.shape[0], -1))

    def decode(self, tech_id: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        e = self.tech_emb(tech_id.long())
        out = self.decoder(torch.cat([z, e], dim=-1))
        return out.view(-1, self.n_frames, self.feat_dim)

    def forward(self, frames: torch.Tensor, tech_id: torch.Tensor) -> torch.Tensor:
        z = self.encode(frames)
        return self.decode(tech_id, z)

    @torch.no_grad()
    def sample(
        self,
        tech_id: int | torch.Tensor,
        *,
        n: int = 1,
        device: torch.device | None = None,
        z_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = device or next(self.parameters()).device
        if isinstance(tech_id, int):
            tid = torch.full((n,), tech_id, device=device, dtype=torch.long)
        else:
            tid = tech_id.to(device).long()
            n = int(tid.shape[0])
        z = torch.randn(n, self.z_dim, device=device, generator=generator) * float(z_scale)
        frames = self.decode(tid, z)
        frames = frames.clone()
        frames[..., 0] = frames[..., 0].clamp(0.0, 1.0)
        frames[..., 1] = frames[..., 1].clamp(0.0, 1.0)
        frames[..., 2] = frames[..., 2].clamp(-1.0, 1.0)
        return frames
