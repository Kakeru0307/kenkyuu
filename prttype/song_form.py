"""曲全体の並び（構成レイヤ）。短い条件 prior とは別物。

v2: テンプレ抽選 + 区間ごとの進行（同 mode 対比）+ beat_type + 編成レシピ。
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
PartName = Literal["backing", "lead", "drum", "bass"]
SectionRole = Literal[
    "intro",
    "verse",
    "pre_chorus",
    "chorus",
    "bridge",
    "inst",
    "outro",
]

ALL_PARTS: frozenset[PartName] = frozenset({"backing", "lead", "drum", "bass"})

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

# 区間内の相対段階（曲 energy とは別軸。intro は常に「区間low」）
ROLE_ENERGY: dict[SectionRole, EnergyLevel] = {
    "intro": "low",
    "verse": "mid",
    "pre_chorus": "mid",
    "bridge": "mid",
    "chorus": "high",
    "inst": "mid",
    "outro": "low",
}

_ENERGY_LEVELS: tuple[EnergyLevel, ...] = ("low", "mid", "high")


def _as_energy(level: EnergyLevel | str | None, default: EnergyLevel = "mid") -> EnergyLevel:
    if level in _ENERGY_LEVELS:
        return level  # type: ignore[return-value]
    return default


def blend_energy_decode_params(
    song_energy: EnergyLevel | str,
    section_energy: EnergyLevel | str,
) -> dict[str, float | int]:
    """曲 energy × 区間 energy。

    区間段階を土台に、曲 energy の mid 差分を足す。
    → mid の low と low の low が別パラメータになる（clamp で潰さない）。
    """
    song = _as_energy(song_energy)
    section = _as_energy(section_energy)
    base = ENERGY_DECODE_PARAMS[section]
    song_p = ENERGY_DECODE_PARAMS[song]
    mid_p = ENERGY_DECODE_PARAMS["mid"]
    out: dict[str, float | int] = {}
    for key, val in base.items():
        delta = float(song_p[key]) - float(mid_p[key])
        merged = float(val) + delta
        if isinstance(val, int):
            out[key] = int(round(merged))
        else:
            out[key] = merged
    return out


def effective_energy(
    song_energy: EnergyLevel | str,
    section_energy: EnergyLevel | str,
) -> EnergyLevel:
    """3値しか受け取れない箇所用。2*song + section（0..8）を low/mid/high に落とす。"""
    song_i = _ENERGY_LEVELS.index(_as_energy(song_energy))
    sec_i = _ENERGY_LEVELS.index(_as_energy(section_energy))
    score = 2 * song_i + sec_i
    if score <= 1:
        return "low"
    if score <= 5:
        return "mid"
    return "high"

# role → メタ用候補（本線の付与はノート単位 Apply。区間一色には使わない）
ROLE_TECHNIQUE_CANDIDATES: dict[SectionRole, list[tuple[str, float]]] = {
    "intro": [("normal", 0.60), ("vibrato", 0.30), ("slide", 0.10)],
    "verse": [("normal", 0.50), ("hammer", 0.25), ("vibrato", 0.15), ("slide", 0.10)],
    "pre_chorus": [("normal", 0.35), ("vibrato", 0.25), ("bend", 0.20), ("hammer", 0.20)],
    "bridge": [("vibrato", 0.40), ("bend", 0.30), ("normal", 0.20), ("slide", 0.10)],
    "chorus": [("bend", 0.30), ("choke_strong", 0.30), ("vibrato", 0.20), ("normal", 0.20)],
    "inst": [("vibrato", 0.35), ("bend", 0.25), ("slide", 0.20), ("normal", 0.20)],
    "outro": [("normal", 0.50), ("vibrato", 0.30), ("bend", 0.20)],
}

ROLE_BEAT_CANDIDATES: dict[SectionRole, list[tuple[str, float]]] = {
    "intro": [
        ("eight_basic", 0.65),
        ("ballad_sparse", 0.20),
        ("sixteen_basic", 0.15),
    ],
    "verse": [
        ("eight_basic", 0.55),
        ("sixteen_basic", 0.30),
        ("shuffle_eight", 0.15),
    ],
    "pre_chorus": [
        ("eight_basic", 0.45),
        ("sixteen_basic", 0.35),
        ("halftime", 0.20),
    ],
    "bridge": [
        ("eight_basic", 0.45),
        ("sixteen_basic", 0.30),
        ("halftime", 0.15),
        ("halftime_shuffle", 0.10),
    ],
    "chorus": [
        ("four_floor", 0.40),
        ("eight_basic", 0.35),
        ("metal_double", 0.15),
        ("sixteen_funk", 0.08),
        ("disco", 0.02),
    ],
    "inst": [
        ("eight_basic", 0.45),
        ("sixteen_basic", 0.35),
        ("halftime", 0.20),
    ],
    "outro": [
        ("eight_basic", 0.60),
        ("ballad_sparse", 0.25),
        ("sixteen_basic", 0.15),
    ],
}

# role → 編成レシピ（区間マスク + 区間内ビルド）
# bass_only は廃止。薄いレシピは spans で短尺キャップ
ROLE_ARRANGEMENT_CANDIDATES: dict[SectionRole, list[tuple[str, float]]] = {
    "intro": [
        ("lead_build", 0.45),
        ("full_sparse", 0.40),
        ("backing_only", 0.10),
        ("lead_only", 0.05),
    ],
    "verse": [
        ("full", 0.88),
        ("no_lead", 0.12),
    ],
    "pre_chorus": [
        ("full", 0.75),
        ("drop_bass", 0.25),
    ],
    "chorus": [
        ("full", 1.0),
    ],
    "bridge": [
        ("full", 0.60),
        ("lead_feature", 0.30),
        ("bass_break", 0.10),
    ],
    "inst": [
        ("full", 0.55),
        ("lead_feature", 0.35),
        ("bass_break", 0.10),
    ],
    "outro": [
        ("full_sparse", 0.45),
        ("strip_down", 0.30),
        ("lead_fade", 0.25),
    ],
}

# 単独パート／ブレイクは区間全程にせず、この小節数で打ち切ってフルへ戻す
MAX_THIN_SPAN_BARS = 4

ENERGY_DECODE_PARAMS: dict[EnergyLevel, dict[str, float | int]] = {
    "low": {
        "onset_th": 0.40,
        "peak_min_distance": 3,
        "release_gap_ticks": 2,
        "lead_onset_th": 0.50,
        "lead_sustain_th": 0.68,
        "bass_onset_th": 0.42,
        "drum_onset_th": 0.40,
    },
    "mid": {
        "onset_th": 0.30,
        "peak_min_distance": 2,
        "release_gap_ticks": 1,
        "lead_onset_th": 0.41,
        "lead_sustain_th": 0.68,
        "bass_onset_th": 0.38,
        "drum_onset_th": 0.35,
    },
    "high": {
        "onset_th": 0.22,
        "peak_min_distance": 2,
        "release_gap_ticks": 1,
        "lead_onset_th": 0.35,
        "lead_sustain_th": 0.68,
        "bass_onset_th": 0.34,
        "drum_onset_th": 0.30,
    },
}

# lead onset: 曲 energy × 区間 energy の固定表（ブレンドなし）
# 行=song (VA), 列=section (role) の low/mid/high
LEAD_ONSET_FIXED: dict[EnergyLevel, dict[EnergyLevel, float]] = {
    "low": {"low": 0.53, "mid": 0.43, "high": 0.37},
    "mid": {"low": 0.50, "mid": 0.41, "high": 0.35},
    "high": {"low": 0.47, "mid": 0.39, "high": 0.33},
}

# 互換（旧疎ビート床。lead 本線では未使用）
LEAD_SPARSE_BEAT_TYPES: frozenset[str] = frozenset(
    {
        "ballad_sparse",
        "halftime",
        "halftime_shuffle",
    }
)
LEAD_SPARSE_ONSET_FLOOR = 0.47
LEAD_ONSET_TH_MAX = 0.50


def resolve_lead_onset_th(
    *,
    energy: EnergyLevel,
    beat_type: str = "",
    arrangement_id: str | None = None,
    base_th: float | None = None,
    song_energy: EnergyLevel | str | None = None,
) -> float:
    """lead onset_th。song × section の固定値（床・ブレンドなし）。"""
    del beat_type, arrangement_id, base_th
    song = _as_energy(song_energy if song_energy is not None else "mid")
    section = _as_energy(energy)
    return float(LEAD_ONSET_FIXED[song][section])

BARS_PER_BLOCK = 8
CHORUS_BARS = 8

# 4/4: duration_sec = total_bars * 240 / bpm
TARGET_DURATION_SEC_LO = 180.0
TARGET_DURATION_SEC_HI = 240.0
_BAR_QUANTUM = 4
_MIN_SECTION_BARS = 4

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
class ArrSpan:
    """区間内の連続サブ区間。bars 分だけ parts のパートを生成する。"""

    bars: int
    parts: frozenset[PartName]

    def has(self, part: PartName) -> bool:
        return part in self.parts


@dataclass(frozen=True)
class FormSection:
    role: SectionRole
    bars: int
    energy: EnergyLevel  # 区間段階（role 由来: intro=low …）
    beat_type: str
    technique_type: str
    progression: str
    label: str
    arrangement_id: str = "full"
    stroke_groove: str = "eight_even"
    spans: tuple[ArrSpan, ...] = ()
    song_energy: EnergyLevel = "mid"  # 曲段階（VA）

    def __post_init__(self) -> None:
        if self.spans:
            return
        object.__setattr__(
            self,
            "spans",
            (ArrSpan(bars=self.bars, parts=ALL_PARTS),),
        )

    @property
    def parts_mask(self) -> frozenset[PartName]:
        mask: set[PartName] = set()
        for span in self.spans:
            mask |= set(span.parts)
        return frozenset(mask)

    @property
    def effective_energy(self) -> EnergyLevel:
        return effective_energy(self.song_energy, self.energy)

    @property
    def decode_params(self) -> dict[str, float | int]:
        params = blend_energy_decode_params(self.song_energy, self.energy)
        params["lead_onset_th"] = resolve_lead_onset_th(
            energy=self.energy,
            song_energy=self.song_energy,
        )
        return params


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
            f"{s.label}[{s.progression}|{s.beat_type}|{s.stroke_groove}|{s.arrangement_id}|{s.bars}]"
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


def _pick_arrangement_id(role: SectionRole, rng: random.Random | None) -> str:
    cands = ROLE_ARRANGEMENT_CANDIDATES[role]
    if rng is None:
        return cands[0][0]
    names = [c[0] for c in cands]
    weights = [c[1] for c in cands]
    return rng.choices(names, weights=weights, k=1)[0]


def _parts(*names: PartName) -> frozenset[PartName]:
    return frozenset(names)


def spans_for_arrangement(arrangement_id: str, total_bars: int) -> tuple[ArrSpan, ...]:
    """レシピ ID から区間内 spans を作る。bars 合計は total_bars。

    lead_only / bass_only / bass_break 等は MAX_THIN_SPAN_BARS で打ち切り、
    残りはバンド編成に戻す（長尺区間で単独パートが伸びすぎないようにする）。
    """
    bars = max(1, int(total_bars))
    half = max(1, bars // 2)
    rest = max(1, bars - half)
    # half+rest が bars を超えないよう調整（奇数 bars）
    if half + rest > bars:
        rest = bars - half
    if rest < 1:
        half = bars
        rest = 0

    def one(parts: frozenset[PartName]) -> tuple[ArrSpan, ...]:
        return (ArrSpan(bars=bars, parts=parts),)

    def two(
        first_parts: frozenset[PartName],
        second_parts: frozenset[PartName],
        *,
        first_bars: int | None = None,
    ) -> tuple[ArrSpan, ...]:
        fb = half if first_bars is None else max(1, min(int(first_bars), bars))
        rb = bars - fb
        if rb < 1:
            return (ArrSpan(bars=bars, parts=first_parts),)
        return (
            ArrSpan(bars=fb, parts=first_parts),
            ArrSpan(bars=rb, parts=second_parts),
        )

    def thin_then(
        thin_parts: frozenset[PartName],
        then_parts: frozenset[PartName],
    ) -> tuple[ArrSpan, ...]:
        thin = min(MAX_THIN_SPAN_BARS, max(1, bars // 4), bars)
        return two(thin_parts, then_parts, first_bars=thin)

    def full_then_thin(
        full_parts: frozenset[PartName],
        thin_parts: frozenset[PartName],
    ) -> tuple[ArrSpan, ...]:
        """outro 系: 本体はフル、末尾だけ短く薄くする。"""
        thin = min(MAX_THIN_SPAN_BARS, max(1, bars // 4), bars)
        fb = bars - thin
        if fb < 1:
            return (ArrSpan(bars=bars, parts=thin_parts),)
        return (
            ArrSpan(bars=fb, parts=full_parts),
            ArrSpan(bars=thin, parts=thin_parts),
        )

    if arrangement_id == "lead_only":
        return thin_then(_parts("lead"), _parts("lead", "drum", "bass"))
    if arrangement_id == "backing_only":
        return thin_then(_parts("backing"), _parts("backing", "drum", "bass"))
    if arrangement_id == "bass_only":
        # 廃止済み ID のフォールバック（旧 manifest 互換）
        return one(ALL_PARTS)
    if arrangement_id == "lead_build":
        # 導入のリードは短く、すぐリズム隊を足す
        return thin_then(_parts("lead"), _parts("lead", "drum", "bass"))
    if arrangement_id == "full_sparse":
        return one(_parts("backing", "lead", "bass"))
    if arrangement_id == "no_lead":
        return one(_parts("backing", "drum", "bass"))
    if arrangement_id == "drop_bass":
        return one(_parts("backing", "lead", "drum"))
    if arrangement_id == "bass_break":
        return thin_then(_parts("bass"), ALL_PARTS)
    if arrangement_id == "lead_feature":
        # バッキング無しは短く、残りはフルバンド
        return thin_then(_parts("lead", "drum", "bass"), ALL_PARTS)
    if arrangement_id == "strip_down":
        return full_then_thin(ALL_PARTS, _parts("lead", "bass"))
    if arrangement_id == "lead_fade":
        return full_then_thin(ALL_PARTS, _parts("lead"))
    # full / unknown
    return one(ALL_PARTS)


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


def _pick_stroke_groove(
    rng: random.Random | None,
    *,
    bpm: float | None = None,
) -> str:
    from makeData.patterns import choose_stroke_groove

    if rng is None:
        return "eight_even"
    return choose_stroke_groove(rng, bpm=bpm)


def _section(
    role: SectionRole,
    *,
    occurrence: int,
    home_progression: str,
    home_mode: str,
    rng: random.Random,
    used_contrast: set[str],
    bpm: float | None = None,
    bars: int | None = None,
    song_energy: EnergyLevel | str = "mid",
) -> FormSection:
    n_bars = int(bars) if bars is not None else _bars_for_role(role)
    n_bars = max(_MIN_SECTION_BARS, n_bars)
    arrangement_id = _pick_arrangement_id(role, rng)
    return FormSection(
        role=role,
        bars=n_bars,
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
        arrangement_id=arrangement_id,
        stroke_groove=_pick_stroke_groove(rng, bpm=bpm),
        spans=spans_for_arrangement(arrangement_id, n_bars),
        song_energy=_as_energy(song_energy),
    )


_TEMPLATES: list[tuple[str, float, tuple[SectionRole, ...]]] = [
    # J-pop
    ("jpop_double_chorus", 1.0, ("intro", "verse", "pre_chorus", "chorus", "verse", "pre_chorus", "chorus", "outro")),
    ("jpop_sabi_inst_sabi", 1.0, ("intro", "verse", "pre_chorus", "chorus", "inst", "chorus", "outro")),
    ("jpop_late_bridge", 1.0, ("intro", "verse", "pre_chorus", "chorus", "verse", "chorus", "bridge", "chorus", "outro")),
    ("jpop_cold_open", 1.0, ("verse", "pre_chorus", "chorus", "verse", "pre_chorus", "chorus", "outro")),
    ("jpop_inst_open", 1.0, ("inst", "verse", "pre_chorus", "chorus", "verse", "chorus", "outro")),
    ("jpop_quiet_loud", 1.0, ("intro", "verse", "verse", "pre_chorus", "chorus", "bridge", "chorus", "outro")),
    ("jpop_pre_twice", 1.0, ("intro", "verse", "pre_chorus", "pre_chorus", "chorus", "verse", "chorus", "outro")),
    ("jpop_outro_sabi", 1.0, ("intro", "verse", "pre_chorus", "chorus", "verse", "pre_chorus", "chorus", "chorus", "outro")),
    ("jpop_bridge_pre", 1.0, ("intro", "verse", "bridge", "pre_chorus", "chorus", "verse", "chorus", "outro")),
    ("jpop_long_story", 1.0, ("intro", "verse", "pre_chorus", "chorus", "verse", "pre_chorus", "bridge", "chorus", "inst", "outro")),
    # Western
    ("west_vcvc", 1.0, ("intro", "verse", "chorus", "verse", "chorus", "outro")),
    ("west_vcvcb", 1.0, ("intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro")),
    ("west_vpcvpc", 1.0, ("intro", "verse", "pre_chorus", "chorus", "verse", "pre_chorus", "chorus", "outro")),
    ("west_middle8", 1.0, ("intro", "verse", "chorus", "verse", "chorus", "bridge", "verse", "chorus", "outro")),
    ("west_solo", 1.0, ("intro", "verse", "chorus", "verse", "chorus", "inst", "chorus", "outro")),
    ("west_double_verse", 1.0, ("intro", "verse", "verse", "chorus", "verse", "chorus", "outro")),
    ("west_break_chorus", 1.0, ("intro", "verse", "chorus", "bridge", "chorus", "outro")),
    ("west_fade_cycle", 1.0, ("intro", "verse", "chorus", "verse", "chorus", "verse", "chorus", "outro")),
    ("west_cold_chorus", 1.0, ("chorus", "verse", "chorus", "verse", "chorus", "outro")),
    ("west_post_inst", 1.0, ("intro", "verse", "chorus", "verse", "chorus", "bridge", "inst", "outro")),
    # Variant
    ("sabi_sandwich", 1.0, ("intro", "chorus", "verse", "chorus", "verse", "chorus", "outro")),
    ("inst_feature", 1.0, ("intro", "verse", "inst", "chorus", "verse", "inst", "chorus", "outro")),
    ("bridge_centric", 1.0, ("intro", "verse", "bridge", "chorus", "bridge", "chorus", "outro")),
]


def _template_weights(
    va: tuple[float, float],
    base_weights: list[float],
) -> list[float]:
    """VA 座標でテンプレ選択確率を調整する（Phase1 ルールベース）。"""
    ids = [t[0] for t in _TEMPLATES]
    weights = list(base_weights)
    valence, arousal = float(va[0]), float(va[1])
    quiet = {
        "jpop_quiet_loud",
        "jpop_cold_open",
        "west_double_verse",
        "west_break_chorus",
    }
    energetic = {
        "jpop_outro_sabi",
        "jpop_late_bridge",
        "west_fade_cycle",
        "west_cold_chorus",
        "sabi_sandwich",
        "west_solo",
    }
    for i, tid in enumerate(ids):
        if valence < 0 and arousal < 0:
            if tid in quiet:
                weights[i] *= 2.0
            elif tid in energetic:
                weights[i] *= 0.5
        if arousal > 0.3:
            if tid in energetic:
                weights[i] *= 2.0
            elif tid in quiet:
                weights[i] *= 0.5
    return weights


def _quantize_bars(n: float) -> int:
    q = _BAR_QUANTUM
    return max(_MIN_SECTION_BARS, int(round(float(n) / q) * q))


def _scale_section_bars(
    base_bars: list[int],
    *,
    bpm: float,
    rng: random.Random,
) -> list[int]:
    """区間小節を比例スケールし、総尺をおおむね 3〜4 分にする。"""
    tempo = max(40.0, float(bpm))
    target_sec = float(rng.uniform(TARGET_DURATION_SEC_LO, TARGET_DURATION_SEC_HI))
    target_total = target_sec * tempo / 240.0
    base_total = float(sum(base_bars)) or float(len(base_bars) * BARS_PER_BLOCK)
    scale = target_total / base_total
    scaled = [_quantize_bars(b * scale) for b in base_bars]

    def _duration_sec(bars: list[int]) -> float:
        return float(sum(bars)) * 240.0 / tempo

    # 合計が帯から外れたらコーラス／長い区間から ±quantum で一度調整
    sec = _duration_sec(scaled)
    if sec < TARGET_DURATION_SEC_LO or sec > TARGET_DURATION_SEC_HI:
        order = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)
        delta = _BAR_QUANTUM if sec < TARGET_DURATION_SEC_LO else -_BAR_QUANTUM
        for i in order:
            nxt = scaled[i] + delta
            if nxt < _MIN_SECTION_BARS:
                continue
            scaled[i] = nxt
            sec = _duration_sec(scaled)
            if TARGET_DURATION_SEC_LO <= sec <= TARGET_DURATION_SEC_HI:
                break
    return scaled


def sample_song_form(
    *,
    home_progression: str,
    home_mode: str,
    seed: int | None = None,
    va: tuple[float, float] = (0.0, 0.0),
    bpm: float | None = None,
    song_energy: EnergyLevel | str | None = None,
) -> SongForm:
    """重み付きでテンプレを1つ選ぶ。区間進行・beat_type・stroke_groove も抽選。

    bpm があるとき区間小節を伸ばし、総尺をおよそ 3〜4 分にする。
    song_energy: 曲 energy（VA）。省略時は va の arousal から決める。
    区間 energy は role 由来の別軸（intro=low 等）。デコードは両軸ブレンド。
    """
    if not home_progression:
        raise ValueError("home_progression is required")
    if not home_mode:
        raise ValueError("home_mode is required")
    if not list_progressions_for_mode(home_mode):
        raise ValueError(f"unknown home_mode: {home_mode!r}")

    from structure_prior import energy_from_va

    if song_energy in _ENERGY_LEVELS:
        resolved_song: EnergyLevel = song_energy  # type: ignore[assignment]
    else:
        resolved_song = _as_energy(energy_from_va(va))

    rng = random.Random(seed)
    ids = [t[0] for t in _TEMPLATES]
    base_weights = [t[1] for t in _TEMPLATES]
    weights = _template_weights(va, base_weights)

    chosen = rng.choices(ids, weights=weights, k=1)[0]
    for tid, _, roles in _TEMPLATES:
        if tid != chosen:
            continue
        base_bars = [_bars_for_role(r) for r in roles]
        if bpm is not None:
            section_bars = _scale_section_bars(base_bars, bpm=float(bpm), rng=rng)
        else:
            section_bars = base_bars
        role_counts: Counter[SectionRole] = Counter()
        used_contrast: set[str] = set()
        sections: list[FormSection] = []
        for role, n_bars in zip(roles, section_bars):
            role_counts[role] += 1
            sections.append(
                _section(
                    role,
                    occurrence=role_counts[role],
                    home_progression=home_progression,
                    home_mode=home_mode,
                    rng=rng,
                    used_contrast=used_contrast,
                    bpm=bpm,
                    bars=n_bars,
                    song_energy=resolved_song,
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
    """(id, base_total_bars, path_ja) の一覧。base はスケール前（各区間8小節）。"""
    out: list[tuple[str, int, str]] = []
    for tid, _, roles in _TEMPLATES:
        bars = sum(_bars_for_role(r) for r in roles)
        path = " → ".join(ROLE_LABEL_JA[r] for r in roles)
        out.append((tid, bars, path))
    return out
