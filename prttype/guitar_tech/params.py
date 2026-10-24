"""Random parameter ranges for technique teacher synthesis."""

from __future__ import annotations

import random
from typing import Any

# Stable IDs used in labels / training
TECH_NAMES: tuple[str, ...] = (
    "none",
    "vibrato",
    "tremolo",
    "staccato",
    "ghost",
    "palm_mute",
    "choke",  # チョーキング
    "choke_strong",
    "prebend",
    "bend_release",
    "arming",
    "slide",
    "hammer",
    "pull",
    "trill",
    "tap",
)

TECH_TO_ID = {n: i for i, n in enumerate(TECH_NAMES)}

# ノートを複数に割る奏法（和音バッキングだと同時発音が爆発する）
NOTE_MULTIPLY_TECHS = frozenset(
    {"tremolo", "trill", "hammer", "pull", "tap"}
)

# 役割別の許可集合（分類結果をここで落とす。logit バイアスではない）
LEAD_ALLOWED_TECHS = frozenset(TECH_NAMES)
BACKING_ALLOWED_TECHS = frozenset(TECH_NAMES) - NOTE_MULTIPLY_TECHS


def allowed_techs_for_role(role: str) -> frozenset[str]:
    if role == "backing":
        return BACKING_ALLOWED_TECHS
    return LEAD_ALLOWED_TECHS


def sample_params(tech: str, rng: random.Random) -> dict[str, Any]:
    if tech == "none":
        return {}
    if tech == "vibrato":
        return {
            "depth": rng.uniform(0.12, 0.45),
            "rate_hz": rng.uniform(4.0, 7.5),
            "delay": rng.uniform(0.0, 0.35),
            "phase": rng.uniform(0.0, 6.28318),
            "fade_in_ms": rng.uniform(0.0, 150.0),
        }
    if tech == "tremolo":
        return {
            "note_len_ms": rng.uniform(25.0, 70.0),
            "gap_ms": rng.uniform(0.0, 25.0),
            "vel_mean": rng.uniform(70.0, 110.0),
            "vel_jitter": rng.uniform(0.0, 12.0),
            "accent_every": rng.choice([0, 2, 3, 4]),
        }
    if tech == "staccato":
        return {
            "len_ratio": rng.uniform(0.15, 0.45),
            "vel_scale": rng.uniform(0.85, 1.1),
            "min_ms": rng.uniform(20.0, 40.0),
            "max_ms": rng.uniform(80.0, 120.0),
        }
    if tech == "ghost":
        return {
            "len_ms": rng.uniform(8.0, 25.0),
            "vel": rng.uniform(25.0, 55.0),
            "pitch_mode": rng.choice(["keep", "drop12"]),
        }
    if tech == "palm_mute":
        return {
            "vel_scale": rng.uniform(0.35, 0.7),
            "len_ratio": rng.uniform(0.4, 0.85),
            "low_bias": rng.uniform(0.0, 0.3),
        }
    if tech == "choke":
        return {
            "target_semitone": rng.uniform(0.5, 2.0),
            "rise_ms": rng.uniform(60.0, 180.0),
            "hold_frac": rng.uniform(0.3, 0.7),
            "overshoot": rng.uniform(0.0, 0.15),
        }
    if tech == "choke_strong":
        return {
            "target_semitone": rng.uniform(1.5, 3.0),
            "rise_ms": rng.uniform(40.0, 120.0),
            "hold_frac": rng.uniform(0.4, 0.8),
            "overshoot": rng.uniform(0.0, 0.25),
        }
    if tech == "prebend":
        return {
            "pre_semitone": rng.uniform(0.5, 2.0),
            "pre_hold_ms": rng.uniform(40.0, 200.0),
            "release_ms": rng.choice([0.0, rng.uniform(80.0, 250.0)]),
        }
    if tech == "bend_release":
        return {
            "peak_semitone": rng.uniform(0.5, 2.0),
            "rise_ms": rng.uniform(40.0, 120.0),
            "release_ms": rng.uniform(80.0, 300.0),
        }
    if tech == "arming":
        return {
            "direction": rng.choice(["down", "dip"]),
            "depth": rng.uniform(0.3, 1.0),
            "dip_ms": rng.uniform(80.0, 400.0),
            "return_ms": rng.uniform(100.0, 500.0),
            "n_dips": rng.randint(1, 3),
        }
    if tech == "slide":
        return {
            "from_semitone": rng.uniform(1.0, 5.0),
            "slide_ms": rng.uniform(40.0, 160.0),
            "mode": rng.choice(["pb", "notes"]),
        }
    if tech == "hammer":
        return {
            "grace_ms": rng.uniform(20.0, 60.0),
            "grace_vel_scale": rng.uniform(0.4, 0.75),
            "overlap_ms": rng.uniform(0.0, 30.0),
            "interval": rng.randint(1, 3),
        }
    if tech == "pull":
        return {
            "from_semitone": rng.uniform(1.0, 3.0),
            "from_ms": rng.uniform(20.0, 70.0),
            "from_vel_scale": rng.uniform(0.5, 0.9),
            "dest_vel_scale": rng.uniform(0.8, 1.1),
        }
    if tech == "trill":
        return {
            "alt_semitone": rng.choice([1, 2]),
            "alt_ms": rng.uniform(40.0, 100.0),
            "n_alts": rng.randint(3, 10),
            "vel_jitter": rng.uniform(0.0, 10.0),
        }
    if tech == "tap":
        return {
            "tap_vel_scale": rng.uniform(0.7, 1.15),
            "tap_len_ms": rng.uniform(40.0, 120.0),
            "pattern": rng.choice(["single", "run"]),
            "run_notes": rng.randint(2, 5),
        }
    return {}


# Relative weights when choosing a non-none tech for an eligible note
TECH_WEIGHTS: dict[str, float] = {
    "vibrato": 1.4,
    "tremolo": 1.1,
    "staccato": 1.0,
    "ghost": 0.8,
    "palm_mute": 1.0,
    "choke": 1.0,
    "choke_strong": 0.6,
    "prebend": 0.7,
    "bend_release": 0.7,
    "arming": 0.5,
    "slide": 0.9,
    "hammer": 0.9,
    "pull": 0.9,
    "trill": 0.7,
    "tap": 0.5,
}
