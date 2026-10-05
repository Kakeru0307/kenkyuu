# 感情エンコーダ（旧 WRIME → 現行 VA encoder）

> 最終更新: 2026-10-05

## 役割

日本語テキストを受け取り、Valence / Arousal を出力する。
出力は structure prior（曲調層）への入力になる。

## 使用モデル（本線）

`gmendes9/multilingual_va_prediction` の **XLM-RoBERTa-large**

- 配置: `prttype/checkpoints/va_encoder/xlm_roberta_large/`
- API: `emotion_va.analyze_va(text) → (valence, arousal)`
- WRIME フォールバックは廃止（ckpt 無ければエラー）

## 入口

```python
from emotion_va import analyze_va

v, a = analyze_va("悲しくて切ない曲を作って")
```

## 旧 WRIME について

`MuneK/bert-large-japanese-v2-finetuned-wrime` は本線から外した。
レガシー参照は `prttype/stash/prompt_legacy/wrime_emotion.py` のみ。

## 現状の問題

### 問題1: 音楽的な語彙への適合

- 多言語 VA 回帰だが、音楽特有の言い回しでの較正は未実施
- **対策**: 音楽レビュー等でのドメイン適応（必要なら）

## 関連ファイル

- `prttype/emotion_va.py` — XlmRobertaVaAnalyzer / analyze_va
- `prttype/sample_structure_params.py` — 文→VA→prior
- `prttype/generate_form.py` — 感情エンコーダ呼び出し
