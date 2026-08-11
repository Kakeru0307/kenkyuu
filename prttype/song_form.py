"""曲全体の並び（構成レイヤ）。短い条件 prior とは別物。

v2: テンプレ抽選 + 区間ごとの進行（同 mode 対比）+ beat_type。
評価データが溜まったら学習サンプラに差し替え可能。
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from makeData.progressions import PROGRESSIONS, ProgressionSpec

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

# role → beat_type 候補（重み付き）。energy / progression とは独立。
ROLE_BEAT_CANDIDATES: dict[SectionRole, list[tuple[str, float]]] = {
    "intro": [
        ("ballad_sparse", 0.70),
        ("eight_basic", 0.30),
    ],
    "a": [
        ("eight_basic", 0.55),
        ("sixteen_basic", 0.25),
        ("shuffle_eight", 0.20),
    ],
    "b": [
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
    "outro": [
        ("ballad_sparse", 0.70),
        ("eight_basic", 0.30),
    ],
}

# 低／中／高 → chord-peak デコード疎密（guitar）+ bass/drum onset しきい値
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
# v1: chorus も 8（パッチ層の bars=16 は未検証）。後で有効化可。
CHORUS_BARS = 8


@dataclass(frozen=True)
class FormSection:
    role: SectionRole
    bars: int
    energy: EnergyLevel
    beat_type: str
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
            f"{s.label}[{s.progression}|{s.beat_type}|{s.bars}]" for s in self.sections
        )
        return (
            f"{self.template_id} ({self.total_bars}bars, home={self.home_progression}): "
            f"{parts}"
        )


def list_progressions_for_mode(mode: str) -> list[ProgressionSpec]:
    """同 mode の進行カタログを返す。"""
    return [s for s in PROGRESSIONS if s.mode == mode]


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

    intro / a / outro → home（a は回数によらず主題提示と回帰）
    b / bridge / chorus → contrast（同 mode、既出を避ける）
    """
    del occurrence
    if role in ("intro", "outro", "a"):
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


def _apply_template_mlp(
    va: tuple[float, float],
    base_weights: list[float],
) -> list[float]:
    """template_prior MLP ckpt があれば softmax 確率で重みを上書き。なければ base_weights をそのまま返す。"""
    from checkpoint_paths import resolve_template_prior_checkpoint
    ckpt_path = resolve_template_prior_checkpoint()
    if not ckpt_path.is_file():
        return base_weights
    try:
        import torch as _torch
        from template_prior import TemplatePriorNet
        n = len(_TEMPLATES)
        checkpoint = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = TemplatePriorNet(n_templates=n)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        bpm_unit = float(checkpoint.get("bpm_unit", 0.5))
        x = _torch.tensor([[va[0], va[1], bpm_unit]], dtype=_torch.float32)
        with _torch.no_grad():
            logits = model(x)
            probs = _torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return [max(1e-6, float(p)) for p in probs]
    except Exception:
        return base_weights


# (template_id, weight, roles...)
_TEMPLATES: list[tuple[str, float, tuple[SectionRole, ...]]] = [
    ("short", 1.0, ("intro", "a", "chorus", "outro")),
    ("standard", 2.0, ("intro", "a", "b", "a", "chorus", "outro")),
    ("extended", 1.0, ("intro", "a", "bridge", "b", "chorus", "a", "outro")),
    ("ballad", 1.0, ("intro", "a", "a", "bridge", "chorus", "outro")),
    ("chorus_repeat", 1.0, ("intro", "a", "b", "chorus", "a", "chorus", "outro")),
    ("drop_chorus", 0.5, ("intro", "a", "bridge", "chorus", "outro")),
    ("extended_long", 1.0, ("intro", "a", "b", "bridge", "chorus", "a", "outro")),
]

# VA に基づくテンプレ重み調整。base_weights は _TEMPLATES の順番に対応。
def _template_weights(
    va: tuple[float, float],
    base_weights: list[float],
) -> list[float]:
    """VA 座標でテンプレ選択確率を調整する（Phase1 ルールベース）。

    valence<0 & arousal<0 → ballad を強化（暗く静か）
    arousal>0.3            → chorus_repeat / drop_chorus を強化（活発）
    """
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
    """重み付きでテンプレを1つ選ぶ。区間進行・beat_type もシード付き抽選。

    home_progression / home_mode は必須。渡さなければ TypeError。
    va を渡すと VA ルールベースでテンプレ選択確率を調整する。
    checkpoints/template_prior/template_prior_last.pt が存在すれば MLP で重みを上書き。
    """
    if not home_progression:
        raise ValueError("home_progression is required")
    if not home_mode:
        raise ValueError("home_mode is required")
    if not list_progressions_for_mode(home_mode):
        raise ValueError(f"unknown home_mode: {home_mode!r}")

    rng = random.Random(seed)
    ids = [t[0] for t in _TEMPLATES]
    base_weights = [t[1] for t in _TEMPLATES]

    # VA ルールベース調整
    weights = _template_weights(va, base_weights)

    # MLP フォールバック（ckpt があれば上書き）
    weights = _apply_template_mlp(va, weights)

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
