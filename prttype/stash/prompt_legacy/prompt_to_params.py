"""【実験用】ふわっとした文 → 生成パラメータ。

本実装ではない。U-Net 本線入口は `generate_backing.py`。
ここでの progression/bpm/key は簡易ルール（既定: 感情→カタログ、
`use_manifest=True` 時のみ合成 manifest サンプリング）。
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Backend = Literal["wrime", "heuristic"]


@dataclass
class GenerationParams:
    progression: str
    key: str
    bpm: float
    bars: int = 8
    bars_per_chord: int = 1
    emotion: str = "na"
    rhythm_intensity: str = "na"
    genres: list[str] = field(default_factory=list)
    backend: str = "wrime"
    prompt: str = ""
    emotion_scores: dict[str, float] = field(default_factory=dict)
    selection: str = "manifest"  # manifest | catalog_fallback
    source_file: str = ""
    family: str = ""
    mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MusicEmotion = Literal["joy", "sadness", "tension", "calm", "na"]


def _make_rng(seed: int | None) -> random.Random:
    if seed is None:
        return random.Random()
    return random.Random(seed)


def wrime_to_music_emotion(scores: dict[str, float]) -> MusicEmotion:
    joy = scores.get("joy", 0.0)
    sadness = scores.get("sadness", 0.0)
    anger = scores.get("anger", 0.0)
    fear = scores.get("fear", 0.0)
    disgust = scores.get("disgust", 0.0)
    trust = scores.get("trust", 0.0)
    anticipation = scores.get("anticipation", 0.0)
    surprise = scores.get("surprise", 0.0)

    tension_score = max(anger, fear, disgust) * 0.7 + anticipation * 0.3
    calm_score = trust * 0.55 + max(0.0, 0.35 - tension_score) + max(0.0, 0.25 - joy)

    candidates = {
        "joy": joy,
        "sadness": sadness,
        "tension": tension_score,
        "calm": calm_score,
    }
    best = max(candidates, key=candidates.get)
    if candidates[best] < 0.18:
        if surprise >= 0.25 and joy >= sadness:
            return "joy"
        return "na"
    return best  # type: ignore[return-value]


def _tempo_hint_from_text(text: str) -> float | None:
    if re.search(r"遅|スロー|ゆったり|slow", text, re.I):
        return 78.0
    if re.search(r"速|アップテンポ|疾走|fast|rapid", text, re.I):
        return 140.0
    return None


def _genre_hints(text: str) -> list[str]:
    genres: list[str] = []
    if re.search(r"ジャズ|jazz", text, re.I):
        genres.append("jazz")
    if re.search(r"ブルース|blues", text, re.I):
        genres.append("blues")
    if re.search(r"ロック|rock", text, re.I):
        genres.append("pop_rock")
    if re.search(r"ポップ|Jポップ|jpop", text, re.I):
        genres.append("pop_rock")
    return genres


def _rhythm_for_emotion(emotion: str) -> str:
    return {
        "joy": "moderate",
        "sadness": "peaceful",
        "calm": "peaceful",
        "tension": "strong",
        "na": "moderate",
    }.get(emotion, "moderate")


def _heuristic_emotion(text: str) -> MusicEmotion:
    if re.search(r"悲し|暗|しんみり|切な|寂し|melanchol|gloom|sad", text, re.I):
        return "sadness"
    if re.search(r"静か|穏やか|癒し|リラックス|落ち着く|calm|peace|relax", text, re.I):
        return "calm"
    if re.search(r"激し|緊張|熱|ロック|アゲ|tense|intense|energy", text, re.I):
        return "tension"
    if re.search(r"明る|元気|うれ|嬉|楽しい|ハッピー|happy|joy|bright", text, re.I):
        return "joy"
    return "na"


def _catalog_fallback(
    *,
    prompt: str,
    emotion: str,
    rng: random.Random,
    backend: str,
    emotion_scores: dict[str, float] | None = None,
) -> GenerationParams:
    """manifest 失敗時のみ: 旧カタログ抽選。"""
    from makeData.progressions import PROGRESSIONS

    mode = "minor" if emotion in ("sadness", "calm", "tension") else "major"
    if emotion == "joy":
        mode = "major"
    pool = [
        p
        for p in PROGRESSIONS
        if (p.mode == "natural_minor") == (mode == "minor")
        or (mode == "minor" and p.mode == "natural_minor")
        or (mode == "major" and p.mode == "major")
    ]
    if not pool:
        pool = list(PROGRESSIONS)
    prefer = {
        "sadness": "diatonic_minor",
        "calm": "diatonic_minor",
        "joy": "diatonic_major",
        "tension": "borrowed",
    }.get(emotion)
    if prefer:
        filtered = [p for p in pool if p.family == prefer]
        if filtered:
            pool = filtered
    weights = [max(p.weight, 0.1) for p in pool]
    spec = rng.choices(pool, weights=weights, k=1)[0]
    key = rng.choice(
        ["A", "E", "D", "G", "C", "B"]
        if spec.mode == "natural_minor"
        else ["C", "G", "D", "A", "E", "F"]
    )
    bpm = {"sadness": 84.0, "calm": 88.0, "joy": 118.0, "tension": 128.0}.get(
        emotion, 112.0
    )
    hint = _tempo_hint_from_text(prompt)
    if hint is not None:
        bpm = hint
    return GenerationParams(
        progression=spec.name,
        key=key,
        bpm=float(bpm),
        bars=8,
        bars_per_chord=1,
        emotion=emotion,
        rhythm_intensity=_rhythm_for_emotion(emotion),
        genres=_genre_hints(prompt),
        backend=backend,
        prompt=prompt,
        emotion_scores=dict(emotion_scores or {}),
        selection="catalog_fallback",
        source_file="",
        family=spec.family,
        mode=spec.mode,
    )


def emotion_to_params(
    *,
    prompt: str,
    emotion: MusicEmotion,
    backend: str,
    seed: int | None = None,
    emotion_scores: dict[str, float] | None = None,
    manifest_path: Path | None = None,
    use_manifest: bool = False,
) -> GenerationParams:
    """感情から GenerationParams を作る（簡易ルール。本実装ではない）。

    既定はカタログ。use_manifest=True のときのみ合成 manifest をサンプリング。
    """
    rng = _make_rng(seed)
    if not use_manifest:
        return _catalog_fallback(
            prompt=prompt,
            emotion=emotion,
            rng=rng,
            backend=backend,
            emotion_scores=emotion_scores,
        )

    from learned_params import sample_from_manifest

    hint = _tempo_hint_from_text(prompt)
    try:
        sample = sample_from_manifest(
            emotion=emotion,
            rng=rng,
            manifest_path=manifest_path,
            bpm_hint=hint,
        )
        bars = 8 if sample.bars < 16 else 16
        return GenerationParams(
            progression=sample.progression,
            key=sample.key,
            bpm=round(sample.bpm, 1),
            bars=bars,
            bars_per_chord=sample.bars_per_chord,
            emotion=emotion,
            rhythm_intensity=_rhythm_for_emotion(emotion),
            genres=_genre_hints(prompt),
            backend=backend,
            prompt=prompt,
            emotion_scores=dict(emotion_scores or {}),
            selection="manifest",
            source_file=sample.source_file,
            family=sample.family,
            mode=sample.mode,
        )
    except Exception as exc:
        print(f"[warn] manifest 選択に失敗したため catalog にフォールバック: {exc}")
        return _catalog_fallback(
            prompt=prompt,
            emotion=emotion,
            rng=rng,
            backend=backend,
            emotion_scores=emotion_scores,
        )


def prompt_to_params(
    prompt: str,
    *,
    backend: Backend = "wrime",
    seed: int | None = None,
    fallback_heuristic: bool = True,
    manifest_path: Path | None = None,
    use_manifest: bool = False,
) -> GenerationParams:
    """文から GenerationParams を作る（簡易ルール。本実装ではない）。"""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt が空です")

    if backend == "heuristic":
        emotion = _heuristic_emotion(prompt)
        return emotion_to_params(
            prompt=prompt,
            emotion=emotion,
            backend="heuristic",
            seed=seed,
            manifest_path=manifest_path,
            use_manifest=use_manifest,
        )

    try:
        from wrime_emotion import analyze_emotion

        result = analyze_emotion(prompt)
        emotion = wrime_to_music_emotion(result.scores)
        hint = _heuristic_emotion(prompt)
        if hint in ("sadness", "calm") and emotion in ("na", "tension", "joy"):
            emotion = hint
        elif hint == "tension" and emotion in ("na", "calm"):
            emotion = hint
        elif hint == "joy" and emotion == "na":
            emotion = hint
        return emotion_to_params(
            prompt=prompt,
            emotion=emotion,
            backend="wrime",
            seed=seed,
            emotion_scores=result.scores,
            manifest_path=manifest_path,
            use_manifest=use_manifest,
        )
    except Exception as exc:
        if not fallback_heuristic:
            raise
        print(f"[warn] WRIME 解析に失敗したため heuristic にフォールバック: {exc}")
        emotion = _heuristic_emotion(prompt)
        return emotion_to_params(
            prompt=prompt,
            emotion=emotion,
            backend="heuristic_fallback",
            seed=seed,
            manifest_path=manifest_path,
            use_manifest=use_manifest,
        )
