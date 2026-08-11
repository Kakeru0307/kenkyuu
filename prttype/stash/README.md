# stash — 主軸外の保管

バッキング／リード生成を本線にしたため、旧フロー・検証用・実験入口をここに移しました。
**削除ではなく保管**です。必要なら戻してください。

- 移設日（旧）: 2026-07-18
- 追記日: 2026-07-26 — ふわっと文・manifest 実験入口

## 構成

| フォルダ | 内容 |
|---|---|
| `scripts/` | 旧検証・可視化スクリプト / 廃止した `train_emotion_classifier.py` |
| `prompt_legacy/` | 文→簡易ルール／manifest 選択の実験入口（最終形ではない） |
| `midi_legacy/` | test1 系の入出力 MIDI（元 MIDI 改変時代） |
| `data_legacy/` | 往復変換結果・旧 patches・Colab 試生成 MIDI |
| `checkpoints_legacy/` | 旧 `guitar-techs` / `p3-music` 等の重み |

## prompt_legacy/（2026-07-26）

| ファイル | 当時の用途 |
|---|---|
| `generate_from_prompt.py` | ふわっと文 → MIDI の実験ラッパー |
| `prompt_to_params.py` | WRIME → catalog / manifest で進行・BPM選択 |
| `learned_params.py` | synthetic manifest から1件サンプリング |
| `wrime_emotion.py` | WRIME 感情解析 |

最終形（文→学習生成→ViTex注釈）とは別物。推論時の台帳ルックアップは本線にしない。
戻して使う場合は `prttype/` ルートに戻し、依存を確認すること。

## scripts/

| ファイル | 当時の用途 |
|---|---|
| `smoke_test_smp.py` | smp U-Net 動作確認 |
| `export_skeleton.py` | 既存 MIDI から骨格抽出（→ `generate_backing.py` に置換） |
| `midiToPic.py` | ViTex 風 RGB 可視化（学習入力には使わない） |
| `patch_to_image.py` / `.ipynb` | パッチ可視化 |
| `vitex_input.png` | 参考画像 |

## 現行マイルストーン（ルートに残しているもの）

- `generate_backing.py` / `generate_lead.py` / `generate_song.py` … 生成入口  
  CLI では進行・キー・BPMを受け取らない（`sample_structure_params.py` で分布サンプル）
- `sample_structure_params.py` … 構造パラメータの暫定サンプラ（条件 prior までの仮）
- `inference.py` … 既存 MIDI → U-Net 変換（補助）
- `skeleton.py` / `midi_to_patch.py` / `patch_to_midi.py` / `model.py`
- `train.py` / `dataset.py` / `prepare_*.py` / `makeData/`
- `progression_input.py` … 骨格生成（学習・生成の内部）
- `generate_lead_pairs.py` … リード学習ペア生成
- `checkpoints/backing/` … バッキング推論用
- `checkpoints/lead/` … リード用

正本: `.cursor/rules/final-generation-form.mdc`
