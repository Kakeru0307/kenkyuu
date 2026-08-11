"""Structure prior: WRIME + V/A → musical structure params.

Trained on gated prior_pairs accept rows. Inference for the final pipeline
(文 → WRIME → V/A → prior → backing/lead).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from emotion_va import wrime_to_va

WRIME_KEYS: tuple[str, ...] = (
    "joy",
    "sadness",
    "anticipation",
    "surprise",
    "anger",
    "fear",
    "disgust",
    "trust",
)

# WRIME 8 + valence + arousal
FEATURE_DIM = len(WRIME_KEYS) + 2

PROGRESSIONS: tuple[str, ...] = (
    "marusa",
    "komuro",
    "canon_short",
    "canon_full",
    "jpop_subdom",
    "classic_turnaround",
    "two_five_one",
    "doowop",
    "pop_axis",
    "cycle_1625",
    "subdom_start",
    "resolve_4516",
    "descending_bass",
    "deceptive",
    "plagal_ish",
    "simple_15",
    "simple_14",
    "simple_16",
    "simple_64",
    "simple_45",
    "minor_komuro",
    "minor_natural_loop",
    "minor_basic",
    "minor_rock",
    "minor_expand",
    "minor_marusa_like",
    "minor_simple_17",
    "minor_simple_16",
    "minor_simple_14",
    "rock_bVII",
    "rock_I_IV_bVII_IV",
    "rock_vi_walkdown",
    "blues_12bar_short",
)

KEYS: tuple[str, ...] = (
    "C",
    "Db",
    "D",
    "Eb",
    "E",
    "F",
    "Gb",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
)

MODES: tuple[str, ...] = ("major", "natural_minor")
ENERGIES: tuple[str, ...] = ("low", "mid", "high")
BARS_PER_CHORD: tuple[int, ...] = (1, 2)

BPM_LO = 60.0
BPM_HI = 150.0

_PROGRESSION_FAMILY: dict[str, str] = {
    **{
        n: "diatonic_major"
        for n in PROGRESSIONS
        if not n.startswith("minor_")
        and n not in ("rock_bVII", "rock_I_IV_bVII_IV", "rock_vi_walkdown", "blues_12bar_short")
    },
    **{n: "diatonic_minor" for n in PROGRESSIONS if n.startswith("minor_")},
    "rock_bVII": "borrowed",
    "rock_I_IV_bVII_IV": "borrowed",
    "rock_vi_walkdown": "borrowed",
    "blues_12bar_short": "blues",
}


@dataclass(frozen=True)
class StructurePriorOut:
    progression: str
    key: str
    bpm: float
    bars: int
    bars_per_chord: int
    mode: str
    family: str
    energy: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def wrime_to_vec(wrime: dict[str, float] | None) -> list[float]:
    w = wrime or {}
    return [float(w.get(k, 0.0)) for k in WRIME_KEYS]


def bpm_to_unit(bpm: float) -> float:
    return max(0.0, min(1.0, (float(bpm) - BPM_LO) / (BPM_HI - BPM_LO)))


def unit_to_bpm(u: float) -> float:
    return BPM_LO + float(u) * (BPM_HI - BPM_LO)


def feature_dim() -> int:
    return FEATURE_DIM


def _resolve_va(
    *,
    wrime: dict[str, float] | None,
    va: tuple[float, float] | None,
) -> tuple[float, float]:
    if va is not None:
        return float(va[0]), float(va[1])
    return wrime_to_va(wrime)


def encode_features(
    *,
    wrime: dict[str, float] | None,
    va: tuple[float, float] | None = None,
) -> list[float]:
    """WRIME 8スコア + Valence/Arousal → 10次元特徴。"""
    v, a = _resolve_va(wrime=wrime, va=va)
    return wrime_to_vec(wrime) + [v, a]


class StructurePriorNet(nn.Module):
    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        *,
        hidden: int = 64,
        n_prog: int = len(PROGRESSIONS),
        n_key: int = len(KEYS),
        n_mode: int = len(MODES),
        n_energy: int = len(ENERGIES),
        n_bpc: int = len(BARS_PER_CHORD),
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_bpm = nn.Linear(hidden, 1)
        self.head_energy = nn.Linear(hidden, n_energy)
        self.head_mode = nn.Linear(hidden, n_mode)
        self.head_key = nn.Linear(hidden, n_key)
        self.head_prog = nn.Linear(hidden, n_prog)
        self.head_bpc = nn.Linear(hidden, n_bpc)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "bpm": self.head_bpm(h).squeeze(-1),
            "energy": self.head_energy(h),
            "mode": self.head_mode(h),
            "key": self.head_key(h),
            "prog": self.head_prog(h),
            "bpc": self.head_bpc(h),
        }


def decode_outputs(
    out: dict[str, torch.Tensor],
    *,
    bars: int = 8,
    index: int = 0,
    sample: bool = False,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> StructurePriorOut:
    """Decode one sample from a batch logits dict (default: index 0).

    sample=False → argmax（決定的）
    sample=True  → categorical / BPM ノイズ（同じ入力でも seed で別テイク）
    """
    temp = max(1e-6, float(temperature))

    def _idx(logits: torch.Tensor) -> int:
        row = logits.reshape(-1, logits.shape[-1])[index].detach().float().cpu()
        if not sample:
            return int(row.argmax().item())
        probs = torch.softmax(row / temp, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator).item())

    bpm_mean = float(torch.sigmoid(out["bpm"].reshape(-1)[index]).item())
    if sample:
        noise_t = torch.randn((), generator=generator)
        noise = float(noise_t.item()) * (0.05 * temp)
        bpm_u = max(0.0, min(1.0, bpm_mean + noise))
    else:
        bpm_u = bpm_mean

    ei = _idx(out["energy"])
    mi = _idx(out["mode"])
    ki = _idx(out["key"])
    pi = _idx(out["prog"])
    bi = _idx(out["bpc"])

    prog = PROGRESSIONS[pi]
    return StructurePriorOut(
        progression=prog,
        key=KEYS[ki],
        bpm=round(unit_to_bpm(bpm_u), 1),
        bars=bars,
        bars_per_chord=BARS_PER_CHORD[bi],
        mode=MODES[mi],
        family=_PROGRESSION_FAMILY.get(prog, "diatonic_major"),
        energy=ENERGIES[ei],
    )


def load_prior(
    ckpt_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[StructurePriorNet, dict[str, Any]]:
    path = Path(ckpt_path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob.get("meta") or {}
    in_dim = int(meta.get("in_dim", FEATURE_DIM))
    if in_dim != FEATURE_DIM:
        raise RuntimeError(
            f"structure prior checkpoint の in_dim={in_dim} は非互換です "
            f"（期待値={FEATURE_DIM}: WRIME8+VA2）。"
            " train_structure_prior.py で再学習してください。"
        )
    hidden = int(meta.get("hidden", 64))
    model = StructurePriorNet(in_dim, hidden=hidden)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    if device is not None:
        model.to(device)
    return model, blob


@torch.inference_mode()
def predict_structure(
    model: StructurePriorNet,
    *,
    wrime: dict[str, float] | None,
    va: tuple[float, float] | None = None,
    bars: int = 8,
    device: str | torch.device | None = None,
    sample: bool = True,
    temperature: float = 1.0,
    seed: int | None = None,
) -> StructurePriorOut:
    feats = encode_features(wrime=wrime, va=va)
    x = torch.tensor([feats], dtype=torch.float32)
    if device is not None:
        x = x.to(device)
        model = model.to(device)
    gen: torch.Generator | None = None
    if sample and seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
    return decode_outputs(
        model(x),
        bars=bars,
        sample=sample,
        temperature=temperature,
        generator=gen,
    )
