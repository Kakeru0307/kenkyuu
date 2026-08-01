"""WRIME 日本語感情強度モデルの薄いラッパ。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


_analyzer: WrimeEmotionAnalyzer | None = None


def get_analyzer(**kwargs: Any) -> WrimeEmotionAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = WrimeEmotionAnalyzer(**kwargs)
    return _analyzer


def analyze_emotion(text: str, **kwargs: Any) -> WrimeScores:
    return get_analyzer(**kwargs).analyze(text)
