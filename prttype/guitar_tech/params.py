"""Technique vocabulary, role filters, and parametric ranges."""

from __future__ import annotations

import random
from typing import Any

# EG-Solo tech names + residual classes (依頼MIDI / 合成)
TECH_NAMES: tuple[str, ...] = (
    "normal",
    "slide",
    "bend",
    "vibrato",
    "mute",
    "pull",
    "harmonic",
    "hammer",
    "tap",
    "tremolo",
    "ghost",
    "choke_strong",
    "prebend",
    "bend_release",
    "arming",
)

TECH_TO_ID = {n: i for i, n in enumerate(TECH_NAMES)}

# Legacy → unified
TECH_ALIASES: dict[str, str] = {
    "none": "normal",
    "palm_mute": "mute",
    "choke": "bend",
    "trill": "vibrato",
    "staccato": "normal",  # dropped class
}

# ノートを複数に割る奏法（和音バッキングだと同時発音が爆発する）
NOTE_MULTIPLY_TECHS = frozenset(
    {"tremolo", "hammer", "pull", "tap"}
)

LEAD_ALLOWED_TECHS = frozenset(TECH_NAMES)
BACKING_ALLOWED_TECHS = frozenset(TECH_NAMES) - NOTE_MULTIPLY_TECHS

# 推論: normal(none) logit 加点（大きいほど控えめ）。本線・CLI の既定。
NONE_BIAS = 1.6

# 前ノートから音程が動くレガート／重なりで slide logit を足す（0=補正なし）
SLIDE_CONTEXT_BIAS = 1.0
SLIDE_CONTEXT_INTERVAL = (1, 7)  # 半音
SLIDE_CONTEXT_GAP_MAX = 0.12  # 秒。負＝重なり

# レア奏法一律加点（0=補正なし）
RARE_TECH_BIAS = 0.0
RARE_TECHS: frozenset[str] = frozenset(
    {
        "pull",
        "hammer",
        "tap",
        "tremolo",
        "ghost",
        "choke_strong",
        "prebend",
        "bend_release",
        "arming",
    }
)

# リード基準のトラック間 velocity ゲイン（伴奏を下げ、lead / 低音を前に）
TRACK_VEL_GAINS: dict[str, float] = {
    "lead": 1.5,
    "backing": 0.76,
    "bass": 0.90,
    "drums": 0.95,
}


def canonicalize_tech(name: str) -> str:
    t = str(name).strip()
    t = TECH_ALIASES.get(t, t)
    return t if t in TECH_TO_ID else "normal"


def allowed_techs_for_role(role: str) -> frozenset[str]:
    if role == "backing":
        return BACKING_ALLOWED_TECHS
    return LEAD_ALLOWED_TECHS


def sample_params(tech: str, rng: random.Random) -> dict[str, Any]:
    tech = canonicalize_tech(tech)
    if tech == "normal":
        return {"vel_scale": 1.0}
    if tech == "vibrato":
        return {
            "depth": rng.uniform(0.12, 0.45),
            "rate_hz": rng.uniform(4.0, 7.5),
            "delay": rng.uniform(0.0, 0.35),
            "phase": rng.uniform(0.0, 6.28318),
            "fade_in_ms": rng.uniform(0.0, 150.0),
            "vel_scale": rng.uniform(0.9, 1.05),
        }
    if tech == "tremolo":
        return {
            "note_len_ms": rng.uniform(25.0, 70.0),
            "gap_ms": rng.uniform(0.0, 25.0),
            "vel_mean": rng.uniform(70.0, 110.0),
            "vel_jitter": rng.uniform(0.0, 12.0),
            "accent_every": rng.choice([0, 2, 3, 4]),
            "vel_scale": 1.0,
        }
    if tech == "ghost":
        return {
            "len_ms": rng.uniform(8.0, 25.0),
            "vel": rng.uniform(25.0, 55.0),
            "pitch_mode": rng.choice(["keep", "drop12"]),
            "vel_scale": rng.uniform(0.25, 0.45),
        }
    if tech == "mute":
        return {
            "vel_scale": rng.uniform(0.35, 0.7),
            "len_ratio": rng.uniform(0.4, 0.85),
            "low_bias": rng.uniform(0.0, 0.3),
        }
    if tech == "bend":
        return {
            "target_semitone": rng.uniform(0.5, 2.0),
            "rise_ms": rng.uniform(60.0, 180.0),
            "hold_frac": rng.uniform(0.3, 0.7),
            "overshoot": rng.uniform(0.0, 0.15),
            "vel_scale": rng.uniform(0.9, 1.1),
        }
    if tech == "choke_strong":
        return {
            "target_semitone": rng.uniform(1.5, 3.0),
            "rise_ms": rng.uniform(40.0, 120.0),
            "hold_frac": rng.uniform(0.4, 0.8),
            "overshoot": rng.uniform(0.0, 0.25),
            "vel_scale": rng.uniform(0.9, 1.1),
        }
    if tech == "prebend":
        return {
            "pre_semitone": rng.uniform(0.5, 2.0),
            "pre_hold_ms": rng.uniform(40.0, 200.0),
            "release_ms": rng.choice([0.0, rng.uniform(80.0, 250.0)]),
            "vel_scale": 1.0,
        }
    if tech == "bend_release":
        return {
            "peak_semitone": rng.uniform(0.5, 2.0),
            "rise_ms": rng.uniform(40.0, 120.0),
            "release_ms": rng.uniform(80.0, 300.0),
            "vel_scale": 1.0,
        }
    if tech == "arming":
        return {
            "direction": rng.choice(["down", "dip"]),
            "depth": rng.uniform(0.3, 1.0),
            "dip_ms": rng.uniform(80.0, 400.0),
            "return_ms": rng.uniform(100.0, 500.0),
            "n_dips": rng.randint(1, 3),
            "vel_scale": 1.0,
        }
    if tech == "slide":
        return {
            "from_semitone": rng.uniform(1.0, 5.0),
            "slide_ms": rng.uniform(40.0, 160.0),
            "mode": rng.choice(["pb", "notes"]),
            "vel_scale": 1.0,
        }
    if tech == "hammer":
        return {
            "grace_ms": rng.uniform(20.0, 60.0),
            "grace_vel_scale": rng.uniform(0.4, 0.75),
            "overlap_ms": rng.uniform(0.0, 30.0),
            "interval": rng.randint(1, 3),
            "vel_scale": 1.0,
        }
    if tech == "pull":
        return {
            "from_semitone": rng.uniform(1.0, 3.0),
            "from_ms": rng.uniform(20.0, 70.0),
            "from_vel_scale": rng.uniform(0.5, 0.9),
            "dest_vel_scale": rng.uniform(0.8, 1.1),
            "vel_scale": 1.0,
        }
    if tech == "tap":
        return {
            "tap_vel_scale": rng.uniform(0.7, 1.15),
            "tap_len_ms": rng.uniform(40.0, 120.0),
            "pattern": rng.choice(["single", "run"]),
            "run_notes": rng.randint(2, 5),
            "vel_scale": 1.0,
        }
    if tech == "harmonic":
        return {"vel_scale": rng.uniform(0.7, 1.0)}
    return {"vel_scale": 1.0}


# Relative weights when choosing a non-normal tech for synthetic fill
TECH_WEIGHTS: dict[str, float] = {
    "vibrato": 1.4,
    "tremolo": 1.1,
    "ghost": 0.8,
    "mute": 1.0,
    "bend": 1.0,
    "choke_strong": 0.6,
    "prebend": 0.7,
    "bend_release": 0.7,
    "arming": 0.5,
    "slide": 0.9,
    "hammer": 0.9,
    "pull": 0.9,
    "tap": 0.5,
    "harmonic": 0.4,
}
