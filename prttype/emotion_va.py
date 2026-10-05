"""テキスト感情エンコーダ: テキスト → Valence/Arousal。

本線 API: analyze_va(text) → (valence, arousal)

モデル: gmendes9/multilingual_va_prediction の XLM-RoBERTa-large
配置: checkpoints/va_encoder/xlm_roberta_large/（pytorch_model.bin 等）

ckpt が無い場合はエラー（WRIME 等へのフォールバックはしない）。
"""

from __future__ import annotations

from pathlib import Path

import torch

_VA_CKPT_DEFAULT = (
    Path(__file__).resolve().parent / "checkpoints" / "va_encoder" / "xlm_roberta_large"
)


def _resolve_va_ckpt(ckpt_path: Path | str) -> Path:
    """存在する VA ckpt パスを返す。無ければ FileNotFoundError。"""
    p = Path(ckpt_path)
    if p.is_dir() and (p / "pytorch_model.bin").is_file():
        return p
    if p.is_file():
        return p
    raise FileNotFoundError(
        f"VA encoder checkpoint がありません: {ckpt_path}\n"
        "gmendes9/multilingual_va_prediction の XLM-RoBERTa-large を\n"
        "checkpoints/va_encoder/xlm_roberta_large/ に配置してください。"
    )


def _hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """論文どおり 0..1 に潰す hard-sigmoid。"""
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


class XlmRobertaVaAnalyzer:
    """XLM-RoBERTa-large ベースの VA 回帰。

    HF 形式（roberta.* + classifier.dense / classifier.out_proj）を読み、
    出力を structure prior 向け [-1, +1] に変換する。
    """

    def __init__(
        self,
        ckpt_path: Path | str = _VA_CKPT_DEFAULT,
        *,
        device: str | None = None,
    ) -> None:
        try:
            from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel
        except ImportError as exc:
            raise ImportError(
                "VA encoder には transformers / sentencepiece / protobuf が必要です"
            ) from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        resolved = _resolve_va_ckpt(ckpt_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(resolved))
        config = XLMRobertaConfig.from_pretrained(str(resolved))
        state = torch.load(resolved / "pytorch_model.bin", map_location="cpu", weights_only=False)

        if not any(k.startswith("roberta.") for k in state):
            raise RuntimeError(
                f"想定外の state_dict です（roberta.* キー無し）: {resolved}"
            )

        self.backbone = XLMRobertaModel(config)
        hidden = int(config.hidden_size)
        self.dense = torch.nn.Linear(hidden, hidden)
        self.out_proj = torch.nn.Linear(hidden, 2)

        backbone_state = {
            k[len("roberta.") :]: v for k, v in state.items() if k.startswith("roberta.")
        }
        self.backbone.load_state_dict(backbone_state, strict=False)
        self.dense.load_state_dict(
            {
                "weight": state["classifier.dense.weight"],
                "bias": state["classifier.dense.bias"],
            }
        )
        self.out_proj.load_state_dict(
            {
                "weight": state["classifier.out_proj.weight"],
                "bias": state["classifier.out_proj.bias"],
            }
        )

        self.backbone.to(self.device).eval()
        self.dense.to(self.device).eval()
        self.out_proj.to(self.device).eval()

    @torch.inference_mode()
    def analyze(self, text: str) -> tuple[float, float]:
        """テキスト → (valence, arousal)。各値は -1〜+1。"""
        encoded = self.tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.backbone(**encoded)
        x = outputs.last_hidden_state[:, 0, :]
        x = torch.tanh(self.dense(x))
        logits = self.out_proj(x).squeeze(0)
        va01 = _hard_sigmoid(logits)
        va = va01 * 2.0 - 1.0
        return float(va[0].item()), float(va[1].item())


_va_analyzer: XlmRobertaVaAnalyzer | None = None
_va_analyzer_ckpt: Path | None = None


def get_va_analyzer(ckpt_path: Path | str = _VA_CKPT_DEFAULT) -> XlmRobertaVaAnalyzer:
    global _va_analyzer, _va_analyzer_ckpt
    p = _resolve_va_ckpt(ckpt_path)
    if _va_analyzer is None or _va_analyzer_ckpt != p:
        _va_analyzer = XlmRobertaVaAnalyzer(p)
        _va_analyzer_ckpt = p
    return _va_analyzer


def analyze_va(text: str, ckpt_path: Path | str = _VA_CKPT_DEFAULT) -> tuple[float, float]:
    """テキスト → (valence, arousal)。VA encoder 必須。"""
    return get_va_analyzer(ckpt_path).analyze(text)
