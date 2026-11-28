"""文条件エンコーダ: テキスト → VA + 音楽条件スコア。

API:
  analyze_condition(text) → {"va": (v, a), "music_scores": {axis: float}}
  analyze_va(text) は emotion_va 本線のまま（後方互換）。
  新 ckpt があるとき analyze_condition を使う。

モデル: 日本語 Transformer + head_va + head_music
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from data.music_condition.axis_tag_map import MUSIC_AXES, empty_scores

_DEFAULT_CKPT = (
    Path(__file__).resolve().parent / "checkpoints" / "va_encoder" / "jp_condition_last"
)


class ConditionEncoder(nn.Module):
    """共有 encoder + head_va(2) + head_music(n_axes)。"""

    def __init__(
        self,
        backbone_name: str = "cl-tohoku/bert-base-japanese-v3",
        n_axes: int = len(MUSIC_AXES),
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = int(self.backbone.config.hidden_size)
        self.head_va = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        self.head_music = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_axes),
        )
        self.n_axes = n_axes

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS
        cls = out.last_hidden_state[:, 0, :]
        va_logits = self.head_va(cls)
        music_logits = self.head_music(cls)
        return {
            "va": torch.tanh(va_logits),  # [-1,1]
            "music_logits": music_logits,
            "music_scores": torch.sigmoid(music_logits),  # [0,1]
        }


def save_condition_ckpt(
    model: ConditionEncoder,
    path: Path,
    *,
    tokenizer_name: str,
    meta: dict | None = None,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "pytorch_model.bin")
    payload = {
        "backbone_name": model.backbone_name,
        "n_axes": model.n_axes,
        "axes": list(MUSIC_AXES),
        "tokenizer_name": tokenizer_name,
        **(meta or {}),
    }
    (path / "condition_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_condition_encoder(
    ckpt_path: Path | str = _DEFAULT_CKPT,
    *,
    device: str | None = None,
) -> tuple[ConditionEncoder, object]:
    """(model, tokenizer)。"""
    from transformers import AutoTokenizer

    ckpt = Path(ckpt_path)
    meta_path = ckpt / "condition_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"condition_meta.json がありません: {ckpt}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = ConditionEncoder(
        backbone_name=meta.get("backbone_name", "cl-tohoku/bert-base-japanese-v3"),
        n_axes=int(meta.get("n_axes", len(MUSIC_AXES))),
    )
    state = torch.load(ckpt / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    tok_name = meta.get("tokenizer_name") or model.backbone_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer


_cond: ConditionEncoder | None = None
_cond_tok = None
_cond_path: Path | None = None


def get_condition_encoder(ckpt_path: Path | str = _DEFAULT_CKPT):
    global _cond, _cond_tok, _cond_path
    p = Path(ckpt_path)
    if _cond is None or _cond_path != p:
        _cond, _cond_tok = load_condition_encoder(p)
        _cond_path = p
    return _cond, _cond_tok


@torch.inference_mode()
def analyze_condition(
    text: str,
    ckpt_path: Path | str = _DEFAULT_CKPT,
) -> dict:
    """text → {va: (v,a), music_scores: {axis: float}}。"""
    model, tokenizer = get_condition_encoder(ckpt_path)
    device = next(model.parameters()).device
    enc = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(enc["input_ids"], enc["attention_mask"])
    va = out["va"].squeeze(0)
    scores_t = out["music_scores"].squeeze(0)
    axes = list(MUSIC_AXES)
    if scores_t.numel() != len(axes):
        # meta と不一致時は空スコア
        scores = empty_scores()
    else:
        scores = {axes[i]: float(scores_t[i].item()) for i in range(len(axes))}
    return {
        "va": (float(va[0].item()), float(va[1].item())),
        "music_scores": scores,
    }
