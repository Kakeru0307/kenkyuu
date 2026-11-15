"""Note-context → technique classifier + vel_scale regressor."""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import NOTE_FEATURE_DIM
from .params import TECH_NAMES


class TechApplyNet(nn.Module):
    def __init__(
        self,
        n_tech: int = len(TECH_NAMES),
        *,
        in_dim: int = NOTE_FEATURE_DIM,
        hidden: int = 128,
        predict_vel: bool = True,
    ) -> None:
        super().__init__()
        self.predict_vel = predict_vel
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.tech_head = nn.Linear(hidden, n_tech)
        self.vel_head = nn.Linear(hidden, 1) if predict_vel else None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.tech_head(h)
        if self.vel_head is None:
            return logits
        vel = self.vel_head(h).squeeze(-1)
        return logits, vel
