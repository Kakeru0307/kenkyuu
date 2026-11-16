"""Jamendo 等の実タグ → 音楽条件軸の写像。

自作スコアや根拠の薄い軸は入れない。
軸名は prior / head_music で共有する。
"""

from __future__ import annotations

# 軸 ID（安定キー）と表示名
MUSIC_AXES: tuple[str, ...] = (
    "rock",
    "jazz",
    "electronic",
    "classical",
    "uptempo",
    "bgm",
    "ballad",
)

# MTG-Jamendo の genre / moodtheme タグ（プレフィックス無しの素名）→ 軸
# 根拠: タグそのものがその軸を指すものだけ。jp_pop は Jamendo に無いので置かない。
TAG_TO_AXIS: dict[str, str] = {
    # genre
    "rock": "rock",
    "metal": "rock",
    "punk": "rock",
    "jazz": "jazz",
    "electronic": "electronic",
    "techno": "electronic",
    "trance": "electronic",
    "house": "electronic",
    "ambient": "electronic",
    "classical": "classical",
    # mood / theme（Jamendo moodtheme 系）
    "energetic": "uptempo",
    "upbeat": "uptempo",
    "fast": "uptempo",
    "relaxing": "bgm",
    "relax": "bgm",
    "calm": "bgm",
    "chill": "bgm",
    "meditative": "bgm",
    "ballad": "ballad",
    "romantic": "ballad",
    "melancholic": "ballad",
    "sad": "ballad",
}

# Warriner / ANEW 照会用の音楽 mood 語（英）。VA は辞書ファイルから読む（ここで発明しない）
MUSIC_MOOD_LEMMAS: tuple[str, ...] = (
    "happy",
    "sad",
    "angry",
    "calm",
    "relaxed",
    "energetic",
    "peaceful",
    "aggressive",
    "melancholy",
    "romantic",
    "dark",
    "bright",
    "tense",
    "gentle",
    "powerful",
    "soft",
    "exciting",
    "mellow",
    "cheerful",
    "gloomy",
)


def axis_index(name: str) -> int:
    return MUSIC_AXES.index(name)


def empty_scores() -> dict[str, float]:
    return {a: 0.0 for a in MUSIC_AXES}


def tags_to_scores(tags: list[str] | set[str]) -> dict[str, float]:
    """複数タグから軸スコア（0/1、同一軸は max）。未知タグは無視。"""
    scores = empty_scores()
    for raw in tags:
        key = str(raw).strip().lower()
        # mood/theme---happy や genre---rock を素名に
        if "---" in key:
            key = key.split("---", 1)[-1]
        if "/" in key:
            key = key.split("/")[-1]
        axis = TAG_TO_AXIS.get(key)
        if axis is not None:
            scores[axis] = 1.0
    return scores
