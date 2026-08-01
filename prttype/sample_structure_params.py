"""生成時の構造パラメータ（進行・キー・BPM・energy）。

最終形: 文 → WRIME → structure prior。
prior チェックポイントが無い／文が無い／感情が na の場合のみ、カタログ乱択にフォールバックする。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from makeData.constants import BPM_RANGE, KEYS
from makeData.progressions import PROGRESSIONS, ProgressionSpec

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRIOR_CKPT = SCRIPT_DIR / "checkpoints" / "prior" / "prior_last.pt"


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
    emotion_target: str = ""


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
    )


def structure_from_prior(
    *,
    text: str | None = None,
    wrime: dict[str, float] | None = None,
    emotion_target: str | None = None,
    prior_checkpoint: Path | None = DEFAULT_PRIOR_CKPT,
    bars: int = 8,
    device: str | None = None,
    sample: bool = True,
    temperature: float = 1.0,
    seed: int | None = None,
) -> StructureParams:
    """WRIME（または文）から structure prior で構造を推論する。

    既定は分布からサンプル（同じ文でも seed が違えば別テイク）。
    sample=False で argmax（決定的）。
    emotion_target が na（判定不能）のときは ValueError（呼び出し側でカタログへ）。
    """
    from structure_prior import EMOTION_TARGETS, load_prior, predict_structure
    from wrime_emotion import analyze_emotion, wrime_to_music_emotion

    ckpt = prior_checkpoint or DEFAULT_PRIOR_CKPT
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"structure prior checkpoint がありません: {ckpt}")

    scores = wrime
    label = emotion_target
    if scores is None:
        if not text:
            raise ValueError("text または wrime が必要です")
        w = analyze_emotion(text)
        scores = dict(w.scores)

    if label is None:
        mapped = wrime_to_music_emotion(scores)
        if mapped == "na":
            raise ValueError("emotion_target unresolved (na)")
        label = mapped
    elif label == "na":
        raise ValueError("emotion_target unresolved (na)")
    elif label not in EMOTION_TARGETS:
        raise ValueError(f"unknown emotion_target: {label}")

    model, blob = load_prior(ckpt, device=device)
    meta = blob.get("meta") or {}
    use_et = bool(meta.get("use_emotion_target", True))
    pred = predict_structure(
        model,
        wrime=scores,
        emotion_target=label,
        use_emotion_target=use_et,
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
        emotion_target=label,
    )


def resolve_structure_params(
    *,
    text: str | None = None,
    emotion_target: str | None = None,
    wrime: dict[str, float] | None = None,
    prior_checkpoint: Path | None = DEFAULT_PRIOR_CKPT,
    seed: int | None = None,
    prefer_prior: bool = True,
    sample: bool = True,
    temperature: float = 1.0,
) -> StructureParams:
    """文があれば prior、無ければカタログ乱択。prior 欠落／感情 na 時はカタログへ落とす。"""
    ckpt = Path(prior_checkpoint) if prior_checkpoint else DEFAULT_PRIOR_CKPT
    if prefer_prior and (text or wrime) and ckpt.is_file():
        try:
            return structure_from_prior(
                text=text,
                wrime=wrime,
                emotion_target=emotion_target,
                prior_checkpoint=ckpt,
                sample=sample,
                temperature=temperature,
                seed=seed,
            )
        except ValueError as exc:
            if "unresolved (na)" in str(exc):
                print("[structure] emotion unresolved (na) → catalog fallback")
                return sample_structure_params(seed=seed)
            raise
    return sample_structure_params(seed=seed)
