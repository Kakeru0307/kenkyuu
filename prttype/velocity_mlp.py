"""VelocityMLP: per-note 特徴量 → velocity 予測モデル。

入力特徴量（3次元）:
  0: technique_id (0 〜 N_TECHNIQUE_TYPES-1)  正規化: / (N_TECHNIQUE_TYPES - 1)
  1: pitch (0-127)                             正規化: /127
  2: duration_beats                            正規化: /4 (clamp)

Guitar-TECHS から学べる情報だけを入力とする。
section_energy は Guitar-TECHS と相関がないためモデル入力には含めない。

出力: velocity (0-127) → sigmoid × 127 で予測

使い方:
  model = VelocityMLP.load("velocity_mlp.pth")
  vel = model.predict(technique_id=1, pitch=64, duration_beats=0.5)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from makeData.constants import N_TECHNIQUE_TYPES

FEATURE_DIM = 3   # (technique_id, pitch, duration_beats)
HIDDEN = 128


class VelocityMLP(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, FEATURE_DIM) → (B, 1) in [0, 1]"""
        return self.net(x)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | None = None) -> "VelocityMLP":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device)
        model = cls()
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(
        self,
        technique_id: int,
        pitch: int,
        duration_beats: float,
    ) -> int:
        """テクニック・ピッチ・音長から Guitar-TECHS ベースの velocity を予測する (0-127)。

        section_energy によるスケーリングは呼び出し元（articulation_layer）が担当。
        """
        feat = _make_feature(technique_id, pitch, duration_beats)
        device = next(self.parameters()).device
        x = torch.from_numpy(feat).unsqueeze(0).to(device)
        raw = self.forward(x)[0, 0].item()
        return int(round(raw * 127))


def _make_feature(
    technique_id: int,
    pitch: int,
    duration_beats: float,
) -> np.ndarray:
    max_id = max(N_TECHNIQUE_TYPES - 1, 1)
    return np.array(
        [
            technique_id / max_id,
            pitch / 127.0,
            min(duration_beats / 4.0, 1.0),
        ],
        dtype=np.float32,
    )
