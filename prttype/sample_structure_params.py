"""生成時の構造パラメータ（進行・キー・BPM・energy）。

最終形: 文 → VA encoder → structure prior。
prior チェックポイントが無い／文が無い場合のみ、カタログ乱択にフォールバックする。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from checkpoint_paths import resolve_structure_prior_checkpoint
from makeData.constants import BPM_RANGE, KEYS
from makeData.progressions import PROGRESSIONS, ProgressionSpec

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRIOR_CKPT = resolve_structure_prior_checkpoint()


@dataclass(frozen=True)
class StructureParams:
    progression: str
    key: str
    bpm: float
    bars: int = 8
    bars_per_chord: int = 1
    family: str = ""
    mode: str = ""
    energy: str = ""
    source: str = "catalog"  # catalog | prior
    va: tuple[float, float] = (0.0, 0.0)


def sample_structure_params(*, seed: int | None = None) -> StructureParams:
    """合成データ作成時と同系の候補から進行・キー・BPM をサンプルする（暫定フォールバック）。"""
    rng = random.Random(seed)
    spec: ProgressionSpec = rng.choice(list(PROGRESSIONS))
    if spec.mode == "natural_minor":
        key_pool = [k for k in KEYS if k in ("A", "E", "D", "G", "C", "B", "F")]
    else:
        key_pool = list(KEYS)
    key = rng.choice(key_pool)
    bpm = float(rng.randint(BPM_RANGE[0], BPM_RANGE[1]))
    bars_per_chord = rng.choice((1, 1, 1, 2))
    energy = "low" if bpm < 90 else ("high" if bpm >= 120 else "mid")
    return StructureParams(
        progression=spec.name,
        key=key,
        bpm=bpm,
        bars=8,
        bars_per_chord=bars_per_chord,
        family=spec.family,
        mode=spec.mode,
        energy=energy,
        source="catalog",
        va=(0.0, 0.0),
    )


def structure_from_prior(
    *,
    text: str | None = None,
    va: tuple[float, float] | None = None,
    prior_checkpoint: Path | None = DEFAULT_PRIOR_CKPT,
    bars: int = 8,
    device: str | None = None,
    sample: bool = True,
    temperature: float = 1.0,
    seed: int | None = None,
) -> StructureParams:
    """V/A から structure prior で構造を推論する。

    VA が渡されない場合はテキストから analyze_va() で取得する。
    既定は分布からサンプル（同じ文でも seed が違えば別テイク）。
    sample=False で argmax（決定的）。
    """
    from emotion_va import analyze_va
    from structure_prior import load_prior, predict_structure

    ckpt = prior_checkpoint or DEFAULT_PRIOR_CKPT
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"structure prior checkpoint がありません: {ckpt}")

    if va is not None:
        resolved_va = va
    elif text:
        resolved_va = analyze_va(text)
    else:
        raise ValueError("va または text が必要です")

    model, _blob = load_prior(ckpt, device=device)
    pred = predict_structure(
        model,
        va=resolved_va,
        bars=bars,
        device=device,
        sample=sample,
        temperature=temperature,
        seed=seed,
    )
    return StructureParams(
        progression=pred.progression,
        key=pred.key,
        bpm=pred.bpm,
        bars=pred.bars,
        bars_per_chord=pred.bars_per_chord,
        family=pred.family,
        mode=pred.mode,
        energy=pred.energy,
        source="prior",
        va=resolved_va,
    )


def resolve_structure_params(
    *,
    text: str | None = None,
    va: tuple[float, float] | None = None,
    prior_checkpoint: Path | None = DEFAULT_PRIOR_CKPT,
    seed: int | None = None,
    prefer_prior: bool = True,
    sample: bool = True,
    temperature: float = 1.0,
) -> StructureParams:
    """VA があれば prior、無ければカタログ乱択。prior 欠落時はカタログへ落とす。"""
    ckpt = Path(prior_checkpoint) if prior_checkpoint else DEFAULT_PRIOR_CKPT
    if prefer_prior and (text or va is not None) and ckpt.is_file():
        return structure_from_prior(
            text=text,
            va=va,
            prior_checkpoint=ckpt,
            sample=sample,
            temperature=temperature,
            seed=seed,
        )
    return sample_structure_params(seed=seed)
