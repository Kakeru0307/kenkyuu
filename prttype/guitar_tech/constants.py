from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # prttype
COLAB = ROOT.parent.parent / "colab_train"

TECH_ROOT = ROOT / "data" / "ギター"
DATASET_DIR = ROOT / "data" / "ギター_dataset"
APPLY_DATASET_DIR = ROOT / "data" / "ギター_dataset" / "apply"
CHECKPOINT_DIR = ROOT / "checkpoints" / "guitar_tech"
OUTPUT_DIR = ROOT / "midi" / "guitar_tech"

EGSOLO_ROOT = COLAB / "lead_data" / "_probe" / "egsolo_note"
GUITAR_TECHS_ROOT = COLAB / "data" / "raw" / "guitar-techs"
if not GUITAR_TECHS_ROOT.is_dir():
    GUITAR_TECHS_ROOT = ROOT / "data" / "raw" / "guitar-techs"

N_FRAMES = 64
FRAME_FEATURE_DIM = 3
FEATURE_DIM = FRAME_FEATURE_DIM
PB_MAX = 8192.0
DEFAULT_PROGRAM = 29

PB_TECHS = frozenset(
    {
        "vibrato",
        "bend",
        "choke_strong",
        "prebend",
        "bend_release",
        "arming",
        "slide",
    }
)

EGSOLO_PITCH_TO_TECH: dict[int, str] = {
    36: "normal",
    38: "slide",
    40: "bend",
    41: "vibrato",
    42: "mute",
    43: "pull",
    44: "harmonic",
    45: "hammer",
    47: "tap",
}
