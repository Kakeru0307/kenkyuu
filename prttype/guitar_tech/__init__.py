"""Guitar technique layer: apply (note→tech) + optional frame AE."""

from .constants import FRAME_FEATURE_DIM, N_FRAMES, TECH_ROOT
from .features import NOTE_FEATURE_DIM

# FEATURE_DIM はフレーム AE 互換。ノート特徴は NOTE_FEATURE_DIM。
FEATURE_DIM = FRAME_FEATURE_DIM

__all__ = [
    "TECH_ROOT",
    "N_FRAMES",
    "FRAME_FEATURE_DIM",
    "NOTE_FEATURE_DIM",
    "FEATURE_DIM",
]
