"""WRIME 日本語感情強度モデルの薄いラッパ。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

HF_MODEL_ID = "MuneK/bert-large-japanese-v2-finetuned-wrime"
WRIME_LABELS = (
    "joy",
    "sadness",
    "anticipation",
    "surprise",
    "anger",
    "fear",
    "disgust",
    "trust",
)

# structure prior 学習ラベル（WRIME 8感情とは別。calm/tension は合成）
MusicEmotion = Literal["joy", "sadness", "tension", "calm", "na"]
PRIOR_EMOTION_TARGETS = ("joy", "sadness", "calm", "tension")


@dataclass
class WrimeScores:
    scores: dict[str, float]
    top_label: str
    top_score: float
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "top_label": self.top_label,
            "top_score": self.top_score,
            "text": self.text,
        }


class WrimeEmotionAnalyzer:
    """MuneK/bert-large-japanese-v2-finetuned-wrime で 8 感情強度を推定する。"""

    def __init__(
        self,
        model_id: str = HF_MODEL_ID,
        *,
        device: str | None = None,
    ) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "WRIME には transformers が必要です: pip install 'transformers>=4.36,<5'"
            ) from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        # id2label が LABEL_0 など汎用名のときは WRIME 公式順を使う
        id2label = getattr(self.model.config, "id2label", None) or {}
        self.labels = list(WRIME_LABELS)
        if id2label and len(id2label) == len(WRIME_LABELS):
            ordered = [str(id2label[i]) for i in range(len(id2label))]
            if not all(name.startswith("LABEL_") for name in ordered):
                self.labels = ordered

    @torch.inference_mode()
    def analyze(self, text: str) -> WrimeScores:
        text = text.strip()
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        logits = self.model(**encoded).logits.squeeze(0)
        # 回帰強度: 生値を [0,1] に収める（負値もありうる）
        values = logits.detach().cpu().float()
        if values.ndim == 0:
            values = values.unsqueeze(0)
        values = values.tolist()
        # 値が概ね 0–1 外なら sigmoid、0–3 程度の回帰なら /3 で正規化
        vmax = max(abs(float(v)) for v in values) if values else 0.0
        if vmax > 1.5:
            values = [max(0.0, min(1.0, float(v) / 3.0)) for v in values]
        elif min(float(v) for v in values) < 0:
            values = torch.sigmoid(torch.tensor(values)).tolist()
        else:
            values = [max(0.0, min(1.0, float(v))) for v in values]
        scores = {
            label: float(values[i]) if i < len(values) else 0.0
            for i, label in enumerate(self.labels)
        }
        top_label = max(scores, key=scores.get)
        return WrimeScores(
            scores=scores,
            top_label=top_label,
            top_score=scores[top_label],
            text=text,
        )


_CLF_PATH_DEFAULT = (
    Path(__file__).resolve().parent / "checkpoints" / "emotion" / "emotion_clf.pt"
)


class _EmotionClassifier:
    """学習済み WRIME→prior ラベル分類器のラッパ（遅延ロード）。"""

    def __init__(self, path: Path) -> None:
        import torch.nn as nn

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.mu: torch.Tensor = ckpt["mu"]
        self.sd: torch.Tensor = ckpt["sd"]
        self.labels: tuple[str, ...] = tuple(ckpt["labels"])
        self.min_confidence: float = float(ckpt.get("min_confidence", 0.50))
        hidden = int(ckpt.get("hidden", 16))
        n_in = len(WRIME_LABELS)
        n_out = len(self.labels)
        self.model = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, scores: dict[str, float]) -> MusicEmotion:
        x = torch.tensor([scores.get(k, 0.0) for k in WRIME_LABELS], dtype=torch.float32)
        z = (x - self.mu) / self.sd
        logits = self.model(z)
        probs = torch.softmax(logits, dim=0)
        idx = int(probs.argmax())
        if float(probs[idx]) < self.min_confidence:
            return "na"
        return self.labels[idx]  # type: ignore[return-value]


_emotion_clf: _EmotionClassifier | None = None


def _get_emotion_clf(path: Path | None = None) -> _EmotionClassifier:
    global _emotion_clf
    if _emotion_clf is None:
        _emotion_clf = _EmotionClassifier(path or _CLF_PATH_DEFAULT)
    return _emotion_clf


def wrime_to_music_emotion(
    scores: dict[str, float],
    *,
    clf_path: Path | None = None,
) -> MusicEmotion:
    """WRIME 8感情スコア → prior 用 4ラベル（または判定不能 na）。

    accept.jsonl で学習した小型分類器（8→16→ReLU→4）を使う。
    手書き式ではなく学習済み重みで判定するため、prior 学習と整合する。
    """
    p = Path(clf_path) if isinstance(clf_path, str) else (clf_path or _CLF_PATH_DEFAULT)
    if not p.exists():
        raise FileNotFoundError(
            f"emotion classifier not found: {p}\n"
            "Run: python prttype/scripts/train_emotion_classifier.py"
        )
    return _get_emotion_clf(p).predict(scores)


_analyzer: WrimeEmotionAnalyzer | None = None


def get_analyzer(**kwargs: Any) -> WrimeEmotionAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = WrimeEmotionAnalyzer(**kwargs)
    return _analyzer


def analyze_emotion(text: str, **kwargs: Any) -> WrimeScores:
    return get_analyzer(**kwargs).analyze(text)
