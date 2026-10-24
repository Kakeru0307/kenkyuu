"""Note-context → technique classifier."""

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
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_tech),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
