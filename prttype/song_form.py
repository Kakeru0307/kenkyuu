"""曲全体の並び（構成レイヤ）。短い条件 prior とは別物。

v1: 定番テンプレから抽選。区間注釈データが揃ったら学習サンプラに差し替え可能。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

EnergyLevel = Literal["low", "mid", "high"]
SectionRole = Literal["intro", "a", "b", "bridge", "chorus", "outro"]

ROLE_LABEL_JA: dict[SectionRole, str] = {
    "intro": "イントロ",
    "a": "Aメロ",
    "b": "Bメロ",
    "bridge": "間奏",
    "chorus": "サビ",
    "outro": "アウトロ",
}

ROLE_ENERGY: dict[SectionRole, EnergyLevel] = {
    "intro": "low",
    "a": "mid",
    "b": "mid",
    "bridge": "mid",
    "chorus": "high",
    "outro": "low",
}

# 低／中／高 → chord-peak デコード疎密
ENERGY_DECODE_PARAMS: dict[EnergyLevel, dict[str, float | int]] = {
    "low": {
        "onset_th": 0.40,
        "peak_min_distance": 3,
        "release_gap_ticks": 2,
        "lead_onset_th": 0.40,
    },
    "mid": {
        "onset_th": 0.30,
        "peak_min_distance": 2,
        "release_gap_ticks": 1,
        "lead_onset_th": 0.30,
    },
    "high": {
        "onset_th": 0.22,
        "peak_min_distance": 2,
        "release_gap_ticks": 1,
        "lead_onset_th": 0.22,
    },
}

BARS_PER_BLOCK = 8


@dataclass(frozen=True)
class FormSection:
    role: SectionRole
    bars: int
    energy: EnergyLevel
    label: str

    @property
    def decode_params(self) -> dict[str, float | int]:
        return dict(ENERGY_DECODE_PARAMS[self.energy])


@dataclass(frozen=True)
class SongForm:
    template_id: str
    sections: tuple[FormSection, ...]

    @property
    def total_bars(self) -> int:
        return sum(s.bars for s in self.sections)

    def describe(self) -> str:
        parts = " → ".join(s.label for s in self.sections)
        return f"{self.template_id} ({self.total_bars}bars): {parts}"


def _section(role: SectionRole, bars: int = BARS_PER_BLOCK) -> FormSection:
    return FormSection(
        role=role,
        bars=bars,
        energy=ROLE_ENERGY[role],
        label=ROLE_LABEL_JA[role],
    )


# (template_id, weight, roles...)
_TEMPLATES: list[tuple[str, float, tuple[SectionRole, ...]]] = [
    ("short", 1.0, ("intro", "a", "chorus", "outro")),
    ("standard", 2.0, ("intro", "a", "b", "a", "chorus", "outro")),
    ("extended", 1.0, ("intro", "a", "bridge", "b", "chorus", "a", "outro")),
]


def sample_song_form(*, seed: int | None = None) -> SongForm:
    """重み付きでテンプレを1つ選ぶ。"""
    rng = random.Random(seed)
    ids = [t[0] for t in _TEMPLATES]
    weights = [t[1] for t in _TEMPLATES]
    chosen = rng.choices(ids, weights=weights, k=1)[0]
    for tid, _, roles in _TEMPLATES:
        if tid == chosen:
            sections = tuple(_section(r) for r in roles)
            return SongForm(template_id=tid, sections=sections)
    raise RuntimeError("template not found")


def list_templates() -> list[tuple[str, int, str]]:
    """(id, total_bars, path_ja) の一覧。"""
    out: list[tuple[str, int, str]] = []
    for tid, _, roles in _TEMPLATES:
        sections = [_section(r) for r in roles]
        path = " → ".join(s.label for s in sections)
        out.append((tid, sum(s.bars for s in sections), path))
    return out
