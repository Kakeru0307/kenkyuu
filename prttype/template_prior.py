"""テンプレ選択 MLP。入力: (valence, arousal, bpm_unit)、出力: テンプレ logit。

VA + 正規化 BPM の 3 次元から各テンプレの選択確率を学習する小 MLP。
ckpt がなければ song_form._apply_template_mlp がスキップするため、
学習前でも song_form は正常動作する（VA ルールベース重みで動く）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemplatePriorNet(nn.Module):
    """3 → hidden → hidden → n_templates の全結合 MLP。"""

    def __init__(self, n_templates: int = 7, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_templates),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3) → logits: (B, n_templates)"""
        return self.net(x)
