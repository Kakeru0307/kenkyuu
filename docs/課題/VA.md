# V/A（Valence / Arousal）

> 最終更新: 2026-11-09

## 役割

テキストから得た感情を数値（V/A）に変換し、曲調層・構成層・注釈層に渡す。
V（感情価）は -1〜+1（ネガティブ〜ポジティブ）、A（覚醒度）は -1〜+1（静か〜激しい）。

現在の実装: `prttype/emotion_va.py` — XLM-RoBERTa-large（`gmendes9/multilingual_va_prediction`）

## 現状の問題

### ~~問題1: VA座標が固定マッピング~~ → **解決済み（2026-10 移行）**

- 旧: `_PLUTCHIK_VA` による 8感情ハードコード近似値
- **現在**: XLM-RoBERTa-large による回帰（データから学習済みの分布）
- `wrime_to_va` / WRIME ベースのパスは廃止。ckpt 無しはエラー

### 問題2: VAが各パートの生成に届いていない（未解決）

- **場所**: `prttype/generate_form.py` → `generate_backing_music` 等への呼び出し
- VAは構成層（`sample_song_form`）には渡っているが、各パート生成関数（backing/lead/drum/bass）には渡っていない
- 演奏の激しさは energy（low/mid/high）でしか調整されておらず、VAと直接対応していない
- **対策**: energy の算出を VA から動的に計算する実装に変更する

### ~~問題3: WRIMEスコアが低い場合にVAが不安定~~ → **廃止（WRIME パス削除）**

- `wrime_to_va` は `prttype/stash/prompt_legacy/` に移設。本線では使用しない

## 依存関係

- 入力: 日本語テキスト（XLM-RoBERTa-large で VA 回帰）
- 出力先: 曲調層（`sample_structure_params.py`）、構成層（`song_form.py`）、注釈層（未接続）

## 学習データ候補

| データセット | 用途 |
|---|---|
| EMOPIA | Q1〜Q4 → VA座標の学習 |
| DadaGP + Essentia | ギター曲の VA 推定（申請中） |

---

## VA encoder セットアップ（新規環境向け）

> 最終更新: 2026-11-09

### 現在の実装

`prttype/emotion_va.py` は `gmendes9/multilingual_va_prediction` の **XLM-RoBERTa-large** を使用。
日本語テキスト → Valence / Arousal（各 -1〜+1）をワンコールで返す。

```python
from prttype.emotion_va import analyze_va
v, a = analyze_va("激しくて速い曲")
```

### ckpt の場所

```
prttype/checkpoints/va_encoder/xlm_roberta_large/
  pytorch_model.bin       ← 約 2.1 GB（git 管理外）
  config.json
  sentencepiece.bpe.model
  special_tokens_map.json
  tokenizer_config.json
  training_args.bin
```

`pytorch_model.bin` のみ git 管理外（.gitignore に記載）。残りのファイルはリポジトリに含まれる。

### ダウンロード手順

#### 方法1: HuggingFace から直接ダウンロード

```powershell
# prttype/ 直下で実行
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='gmendes9/multilingual_va_prediction',
    filename='pytorch_model.bin',
    local_dir='checkpoints/va_encoder/xlm_roberta_large'
)
"
```

#### 方法2: 別 PC からコピー

既にダウンロード済みの PC から `pytorch_model.bin`（約 2.1 GB）を USB / ネットワーク経由でコピーする。

```
# コピー先
prttype/checkpoints/va_encoder/xlm_roberta_large/pytorch_model.bin
```

### 動作確認

```powershell
cd prttype
python -c "from emotion_va import analyze_va; print(analyze_va('楽しい曲'))"
# → (0.xx, 0.xx) のような数値が出れば OK
```

### サイズと代替案

| モデル | サイズ | 備考 |
|---|---|---|
| XLM-RoBERTa-large（現在） | 約 2.1 GB | 多言語・高精度 |
| XLM-RoBERTa-base | 約 1.1 GB | 精度はやや下がる |
| float16 量子化版 | 約 1.1 GB | `torch.save` で変換可能 |
