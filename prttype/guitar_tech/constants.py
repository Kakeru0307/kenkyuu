from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TECH_ROOT = ROOT / "data" / "ギター"
DATASET_DIR = ROOT / "data" / "ギター_dataset"
APPLY_DATASET_DIR = ROOT / "data" / "ギター_dataset" / "apply"
CHECKPOINT_DIR = ROOT / "checkpoints" / "guitar_tech"
OUTPUT_DIR = ROOT / "midi" / "guitar_tech"

# 固定長フレーム表現（約4秒クリップ想定）— TechFrameAE 用
N_FRAMES = 64
# [pitch_norm, gate, pb_norm]
FRAME_FEATURE_DIM = 3
# 後方互換（フレーム AE）。ノート奏法特徴は features.NOTE_FEATURE_DIM
FEATURE_DIM = FRAME_FEATURE_DIM
PB_MAX = 8192.0
DEFAULT_PROGRAM = 29  # Overdriven Guitar
# Pitch-bend を使う奏法（チャンネル共有のため重なりノートではスキップ）
PB_TECHS = frozenset(
    {
        "vibrato",
        "choke",
        "choke_strong",
        "prebend",
        "bend_release",
        "arming",
        "slide",
    }
)
