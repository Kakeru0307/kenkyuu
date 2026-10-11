"""Structure prior: V/A → musical structure params.

Trained on gated prior_pairs accept rows + EMOPIA pairs.
Inference for the final pipeline (文 → VA encoder → V/A → prior → backing/lead).

v2: 入力は VA 2次元（感情エンコーダがテキストから直接回帰）。
v3: BPM は条件つきビン分布からサンプル。energy は BPM から導出（学習ヘッドなし）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# valence + arousal の 2次元のみ
FEATURE_DIM = 2

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

BPM_LO = 40.0
BPM_HI = 240.0
# 中央は約 10 BPM 刻み、端（〜50・200〜）は広め
BPM_BIN_EDGES: tuple[float, ...] = (
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
    110.0,
    120.0,
    130.0,
    140.0,
    150.0,
    160.0,
    170.0,
    180.0,
    190.0,
    200.0,
    220.0,
    240.0,
)
N_BPM_BINS = len(BPM_BIN_EDGES) - 1

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


def energy_from_bpm(bpm: float) -> str:
    b = float(bpm)
    if b < 90.0:
        return "low"
    if b < 120.0:
        return "mid"
    return "high"


def bpm_to_bin(bpm: float) -> int:
    """BPM → ビン index（[lo, hi) 、最後のビンは hi を含む）。"""
    b = max(BPM_LO, min(BPM_HI, float(bpm)))
    for i in range(N_BPM_BINS):
        lo = BPM_BIN_EDGES[i]
        hi = BPM_BIN_EDGES[i + 1]
        if i == N_BPM_BINS - 1:
            if lo <= b <= hi:
                return i
        elif lo <= b < hi:
            return i
    return N_BPM_BINS - 1


def bin_to_bpm(bin_i: int) -> float:
    """ビン代表値（区間中央）。"""
    i = max(0, min(N_BPM_BINS - 1, int(bin_i)))
    lo = BPM_BIN_EDGES[i]
    hi = BPM_BIN_EDGES[i + 1]
    return round((lo + hi) * 0.5, 1)


def bpm_to_unit(bpm: float) -> float:
    return max(0.0, min(1.0, (float(bpm) - BPM_LO) / (BPM_HI - BPM_LO)))


def unit_to_bpm(u: float) -> float:
    return BPM_LO + float(u) * (BPM_HI - BPM_LO)


def encode_features(
    *,
    va: tuple[float, float],
) -> list[float]:
    """Valence/Arousal → 2次元特徴。"""
    return [float(va[0]), float(va[1])]


class StructurePriorNet(nn.Module):
    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        *,
        hidden: int = 64,
        n_bpm: int = N_BPM_BINS,
        n_prog: int = len(PROGRESSIONS),
        n_key: int = len(KEYS),
        n_mode: int = len(MODES),
        n_bpc: int = len(BARS_PER_CHORD),
    ) -> None:
        super().__init__()
        self.n_bpm = int(n_bpm)
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_bpm = nn.Linear(hidden, self.n_bpm)
        self.head_mode = nn.Linear(hidden, n_mode)
        self.head_key = nn.Linear(hidden, n_key)
        self.head_prog = nn.Linear(hidden, n_prog)
        self.head_bpc = nn.Linear(hidden, n_bpc)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "bpm": self.head_bpm(h),
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
    sample=True  → categorical（同じ入力でも seed で別テイク）
    """
    temp = max(1e-6, float(temperature))

    def _idx(logits: torch.Tensor) -> int:
        row = logits.reshape(-1, logits.shape[-1])[index].detach().float().cpu()
        if not sample:
            return int(row.argmax().item())
        probs = torch.softmax(row / temp, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator).item())

    bpm_i = _idx(out["bpm"])
    bpm = bin_to_bpm(bpm_i)
    mi = _idx(out["mode"])
    ki = _idx(out["key"])
    pi = _idx(out["prog"])
    bi = _idx(out["bpc"])

    prog = PROGRESSIONS[pi]
    return StructurePriorOut(
        progression=prog,
        key=KEYS[ki],
        bpm=bpm,
        bars=bars,
        bars_per_chord=BARS_PER_CHORD[bi],
        mode=MODES[mi],
        family=_PROGRESSION_FAMILY.get(prog, "diatonic_major"),
        energy=energy_from_bpm(bpm),
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
            f"（期待値={FEATURE_DIM}: VA2）。"
            " train_structure_prior.py で再学習してください。"
        )
    state = blob["model_state_dict"]
    bpm_w = state.get("head_bpm.weight")
    if bpm_w is None:
        raise RuntimeError("structure prior checkpoint に head_bpm がありません。")
    n_bpm = int(bpm_w.shape[0])
    if n_bpm == 1:
        raise RuntimeError(
            "structure prior checkpoint は旧 BPM 回帰（1 次元）です。"
            " ビン分布版で train_structure_prior.py を再学習してください。"
        )
    if n_bpm != N_BPM_BINS:
        raise RuntimeError(
            f"structure prior checkpoint の BPM ビン数={n_bpm} は非互換です "
            f"（期待値={N_BPM_BINS}）。再学習してください。"
        )
    if "head_energy.weight" in state:
        raise RuntimeError(
            "structure prior checkpoint に旧 head_energy があります。"
            " ビン分布版で再学習してください。"
        )
    hidden = int(meta.get("hidden", 64))
    model = StructurePriorNet(in_dim, hidden=hidden, n_bpm=n_bpm)
    model.load_state_dict(state)
    model.eval()
    if device is not None:
        model.to(device)
    return model, blob


@torch.inference_mode()
def predict_structure(
    model: StructurePriorNet,
    *,
    va: tuple[float, float],
    bars: int = 8,
    device: str | torch.device | None = None,
    sample: bool = True,
    temperature: float = 1.0,
    seed: int | None = None,
) -> StructurePriorOut:
    resolved_va: tuple[float, float] = va if va is not None else (0.0, 0.0)
    feats = encode_features(va=resolved_va)
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
