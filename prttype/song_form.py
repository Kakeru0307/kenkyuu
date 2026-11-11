"""曲全体の並び（構成レイヤ）。短い条件 prior とは別物。

v2: テンプレ抽選 + 区間ごとの進行（同 mode 対比）+ beat_type。
区間 role は SongFormer 語彙に統一。
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

from makeData.progressions import (
    PROGRESSIONS,
    ProgressionSpec,
    modes_allowed_for_spec,
)

EnergyLevel = Literal["low", "mid", "high"]
SectionRole = Literal[
    "intro",
    "verse",
    "pre_chorus",
    "chorus",
    "bridge",
    "inst",
    "outro",
]

SECTION_ROLES: tuple[SectionRole, ...] = (
    "intro",
    "verse",
    "pre_chorus",
    "chorus",
    "bridge",
    "inst",
    "outro",
)

ROLE_LABEL_JA: dict[SectionRole, str] = {
    "intro": "イントロ",
    "verse": "Aメロ",
    "pre_chorus": "Bメロ",
    "bridge": "間奏",
    "chorus": "サビ",
    "inst": "インスト",
    "outro": "アウトロ",
}

ROLE_ENERGY: dict[SectionRole, EnergyLevel] = {
    "intro": "low",
    "verse": "mid",
    "pre_chorus": "mid",
    "bridge": "mid",
    "chorus": "high",
    "inst": "mid",
    "outro": "low",
}

# role → リード奏法。guitar_tech.params.TECH_NAMES の名前を使う。
ROLE_TECHNIQUE_CANDIDATES: dict[SectionRole, list[tuple[str, float]]] = {
    "intro": [("none", 0.60), ("vibrato", 0.30), ("slide", 0.10)],
    "verse": [("none", 0.50), ("hammer", 0.25), ("vibrato", 0.15), ("slide", 0.10)],
    "pre_chorus": [("none", 0.35), ("vibrato", 0.25), ("choke", 0.20), ("hammer", 0.20)],
    "bridge": [("vibrato", 0.40), ("choke", 0.30), ("none", 0.20), ("slide", 0.10)],
    "chorus": [("choke", 0.30), ("choke_strong", 0.30), ("vibrato", 0.20), ("none", 0.20)],
    "inst": [("vibrato", 0.35), ("choke", 0.25), ("slide", 0.20), ("none", 0.20)],
    "outro": [("none", 0.50), ("vibrato", 0.30), ("choke", 0.20)],
}

ROLE_BEAT_CANDIDATES: dict[SectionRole, list[tuple[str, float]]] = {
    "intro": [
        ("ballad_sparse", 0.70),
        ("eight_basic", 0.30),
    ],
    "verse": [
        ("eight_basic", 0.55),
        ("sixteen_basic", 0.25),
        ("shuffle_eight", 0.20),
    ],
    "pre_chorus": [
        ("halftime", 0.40),
        ("eight_basic", 0.30),
        ("sixteen_basic", 0.30),
    ],
    "bridge": [
        ("halftime", 0.50),
        ("halftime_shuffle", 0.30),
        ("ballad_sparse", 0.20),
    ],
    "chorus": [
        ("four_floor", 0.35),
        ("sixteen_funk", 0.25),
        ("eight_basic", 0.20),
        ("metal_double", 0.15),
        ("disco", 0.05),
    ],
    "inst": [
        ("halftime", 0.40),
        ("eight_basic", 0.30),
        ("sixteen_basic", 0.30),
    ],
    "outro": [
        ("ballad_sparse", 0.70),
        ("eight_basic", 0.30),
    ],
}

ENERGY_DECODE_PARAMS: dict[EnergyLevel, dict[str, float | int]] = {
    "low": {
        "onset_th": 0.40,
        "peak_min_distance": 3,
        "release_gap_ticks": 2,
        "lead_onset_th": 0.40,
        "bass_onset_th": 0.35,
        "drum_onset_th": 0.40,
    },
    "mid": {
        "onset_th": 0.30,
        "peak_min_distance": 2,
        "release_gap_ticks": 1,
        "lead_onset_th": 0.30,
        "bass_onset_th": 0.30,
        "drum_onset_th": 0.35,
    },
    "high": {
        "onset_th": 0.22,
        "peak_min_distance": 2,
        "release_gap_ticks": 1,
        "lead_onset_th": 0.22,
        "bass_onset_th": 0.25,
        "drum_onset_th": 0.30,
    },
}

BARS_PER_BLOCK = 8
CHORUS_BARS = 8

# AnimeTAB Clips ファイル名サフィックス → role
ANIMETAB_CLIP_TO_ROLE: dict[str, SectionRole] = {
    "I": "intro",
    "A": "verse",
    "B": "chorus",
    "C": "bridge",
}

# SongFormer 出力 → role
SONGFORMER_TO_ROLE: dict[str, SectionRole | None] = {
    "intro": "intro",
    "verse": "verse",
    "pre-chorus": "pre_chorus",
    "prechorus": "pre_chorus",
    "pre_chorus": "pre_chorus",
    "chorus": "chorus",
    "bridge": "bridge",
    "inst": "inst",
    "instrumental": "inst",
    "outro": "outro",
    "silence": None,
}


def section_role_to_id(role: str) -> int:
    if role not in SECTION_ROLES:
        raise ValueError(f"unknown section role: {role!r}")
    return SECTION_ROLES.index(role)  # type: ignore[arg-type]


def make_section_onehot_map(
    section_id: int,
    *,
    height: int,
    width: int,
    n_types: int | None = None,
) -> np.ndarray:
    """section id を (N,H,W) の one-hot 定数マップにする。"""
    n = len(SECTION_ROLES) if n_types is None else int(n_types)
    out = np.zeros((n, height, width), dtype=np.float32)
    idx = int(section_id)
    if not (0 <= idx < n):
        raise ValueError(f"section_id out of range: {section_id}")
    out[idx] = 1.0
    return out


def map_songformer_label(label: str) -> SectionRole | None:
    key = str(label).strip().lower().replace(" ", "-")
    if key in SONGFORMER_TO_ROLE:
        return SONGFORMER_TO_ROLE[key]
    key2 = key.replace("-", "_")
    return SONGFORMER_TO_ROLE.get(key2)


@dataclass(frozen=True)
class FormSection:
    role: SectionRole
    bars: int
    energy: EnergyLevel
    beat_type: str
    technique_type: str
    progression: str
    label: str

    @property
    def decode_params(self) -> dict[str, float | int]:
        return dict(ENERGY_DECODE_PARAMS[self.energy])


@dataclass(frozen=True)
class SongForm:
    template_id: str
    home_progression: str
    home_mode: str
    sections: tuple[FormSection, ...]

    @property
    def total_bars(self) -> int:
        return sum(s.bars for s in self.sections)

    def describe(self) -> str:
        parts = " → ".join(
            f"{s.label}[{s.progression}|{s.beat_type}|{s.technique_type}|{s.bars}]"
            for s in self.sections
        )
        return (
            f"{self.template_id} ({self.total_bars}bars, home={self.home_progression}): "
            f"{parts}"
        )


def list_progressions_for_mode(mode: str) -> list[ProgressionSpec]:
    """同 mode で選んでよい進行（borrowed/blues は両 mode）。"""
    return [s for s in PROGRESSIONS if mode in modes_allowed_for_spec(s)]


def pick_contrast(
    home: str,
    mode: str,
    rng: random.Random,
    *,
    exclude: set[str] | frozenset[str] | None = None,
) -> str:
    """同 mode から home 以外の進行を重み付き抽選。候補が無ければ home。"""
    blocked = set(exclude or ())
    blocked.add(home)
    cands = [s for s in list_progressions_for_mode(mode) if s.name not in blocked]
    if not cands:
        cands = [s for s in list_progressions_for_mode(mode) if s.name != home]
    if not cands:
        return home
    weights = [max(0.01, float(s.weight)) for s in cands]
    return rng.choices([s.name for s in cands], weights=weights, k=1)[0]


def _pick_beat_type(role: SectionRole, rng: random.Random | None) -> str:
    cands = ROLE_BEAT_CANDIDATES[role]
    if rng is None:
        return cands[0][0]
    names = [c[0] for c in cands]
    weights = [c[1] for c in cands]
    return rng.choices(names, weights=weights, k=1)[0]


def _pick_technique_type(role: SectionRole, rng: random.Random | None) -> str:
    cands = ROLE_TECHNIQUE_CANDIDATES[role]
    if rng is None:
        return cands[0][0]
    names = [c[0] for c in cands]
    weights = [c[1] for c in cands]
    return rng.choices(names, weights=weights, k=1)[0]


def _bars_for_role(role: SectionRole) -> int:
    if role == "chorus":
        return CHORUS_BARS
    return BARS_PER_BLOCK


def _pick_progression_for_role(
    role: SectionRole,
    *,
    occurrence: int,
    home: str,
    mode: str,
    rng: random.Random,
    used_contrast: set[str],
) -> str:
    """role から区間進行を決める。

    intro / verse / outro → home
    pre_chorus / bridge / chorus / inst → contrast
    """
    del occurrence
    if role in ("intro", "outro", "verse"):
        return home
    chosen = pick_contrast(home, mode, rng, exclude=used_contrast)
    if chosen != home:
        used_contrast.add(chosen)
    return chosen


def _section(
    role: SectionRole,
    *,
    occurrence: int,
    home_progression: str,
    home_mode: str,
    rng: random.Random,
    used_contrast: set[str],
) -> FormSection:
    return FormSection(
        role=role,
        bars=_bars_for_role(role),
        energy=ROLE_ENERGY[role],
        beat_type=_pick_beat_type(role, rng),
        technique_type=_pick_technique_type(role, rng),
        progression=_pick_progression_for_role(
            role,
            occurrence=occurrence,
            home=home_progression,
            mode=home_mode,
            rng=rng,
            used_contrast=used_contrast,
        ),
        label=ROLE_LABEL_JA[role],
    )


_TEMPLATES: list[tuple[str, float, tuple[SectionRole, ...]]] = [
    ("short", 1.0, ("intro", "verse", "chorus", "outro")),
    ("standard", 2.0, ("intro", "verse", "pre_chorus", "verse", "chorus", "outro")),
    ("extended", 1.0, ("intro", "verse", "bridge", "pre_chorus", "chorus", "verse", "outro")),
    ("ballad", 1.0, ("intro", "verse", "verse", "bridge", "chorus", "outro")),
    ("chorus_repeat", 1.0, ("intro", "verse", "pre_chorus", "chorus", "verse", "chorus", "outro")),
    ("drop_chorus", 0.5, ("intro", "verse", "bridge", "chorus", "outro")),
    ("extended_long", 1.0, ("intro", "verse", "pre_chorus", "bridge", "chorus", "verse", "outro")),
]


def _template_weights(
    va: tuple[float, float],
    base_weights: list[float],
) -> list[float]:
    """VA 座標でテンプレ選択確率を調整する（Phase1 ルールベース）。"""
    ids = [t[0] for t in _TEMPLATES]
    weights = list(base_weights)
    valence, arousal = float(va[0]), float(va[1])
    for i, tid in enumerate(ids):
        if valence < 0 and arousal < 0:
            if tid == "ballad":
                weights[i] *= 2.5
            elif tid in ("chorus_repeat", "drop_chorus"):
                weights[i] *= 0.4
        if arousal > 0.3:
            if tid in ("chorus_repeat", "drop_chorus"):
                weights[i] *= 2.0
            elif tid == "ballad":
                weights[i] *= 0.4
    return weights


def sample_song_form(
    *,
    home_progression: str,
    home_mode: str,
    seed: int | None = None,
    va: tuple[float, float] = (0.0, 0.0),
) -> SongForm:
    """重み付きでテンプレを1つ選ぶ。区間進行・beat_type もシード付き抽選。"""
    if not home_progression:
        raise ValueError("home_progression is required")
    if not home_mode:
        raise ValueError("home_mode is required")
    if not list_progressions_for_mode(home_mode):
        raise ValueError(f"unknown home_mode: {home_mode!r}")

    rng = random.Random(seed)
    ids = [t[0] for t in _TEMPLATES]
    base_weights = [t[1] for t in _TEMPLATES]
    weights = _template_weights(va, base_weights)

    chosen = rng.choices(ids, weights=weights, k=1)[0]
    for tid, _, roles in _TEMPLATES:
        if tid != chosen:
            continue
        role_counts: Counter[SectionRole] = Counter()
        used_contrast: set[str] = set()
        sections: list[FormSection] = []
        for role in roles:
            role_counts[role] += 1
            sections.append(
                _section(
                    role,
                    occurrence=role_counts[role],
                    home_progression=home_progression,
                    home_mode=home_mode,
                    rng=rng,
                    used_contrast=used_contrast,
                )
            )
        return SongForm(
            template_id=tid,
            home_progression=home_progression,
            home_mode=home_mode,
            sections=tuple(sections),
        )
    raise RuntimeError("template not found")


def list_templates() -> list[tuple[str, int, str]]:
    """(id, total_bars, path_ja) の一覧。"""
    out: list[tuple[str, int, str]] = []
    for tid, _, roles in _TEMPLATES:
        bars = sum(_bars_for_role(r) for r in roles)
        path = " → ".join(ROLE_LABEL_JA[r] for r in roles)
        out.append((tid, bars, path))
    return out
