# V/A（Valence / Arousal）

> 最終更新: 2026-10-04

## 役割

テキストから得た感情を数値（V/A）に変換し、曲調層・構成層・注釈層に渡す。
V（感情価）は -1〜+1（ネガティブ〜ポジティブ）、A（覚醒度）は -1〜+1（静か〜激しい）。

## 現状の問題

### 問題1: VA座標が固定マッピング

- **場所**: `prttype/emotion_va.py` の `_PLUTCHIK_VA`
- 8感情 → VA の変換がハードコードされた近似値
- 音楽データから学習したものではないため、ギター音楽との対応が保証されていない
- **対策**: EMOPIA（Q1〜Q4ラベル付きMIDI）を学習データとして VA座標を推定し直す

### 問題2: VAが各パートの生成に届いていない

- **場所**: `prttype/generate_form.py` → `generate_backing_music` 等への呼び出し
- VAは構成層（`sample_song_form`）には渡っているが、各パート生成関数（backing/lead/drum/bass）には渡っていない
- 演奏の激しさは energy（low/mid/high）でしか調整されておらず、VAと直接対応していない
- **対策**: energy の算出を VA から動的に計算する実装に変更する

### 問題3: WRIMEスコアが低い場合にVAが不安定

- **場所**: `prttype/emotion_va.py` の `wrime_to_va`
- 感情スコアが全体的に小さいと total が小さくなり、VA が (0, 0) に引き寄せられる
- 音楽的に中性なテキストでも意図しない VA になる場合がある
- **対策**: スコアを正規化してから加重平均を取る

## 依存関係

- 入力: WRIME 8スコア（`emotion_va.py`）
- 出力先: 曲調層（`sample_structure_params.py`）、構成層（`song_form.py`）、注釈層（未接続）

## 学習データ候補

| データセット | 用途 |
|---|---|
| EMOPIA | Q1〜Q4 → VA座標の学習 |
| DadaGP + Essentia | ギター曲の VA 推定（申請中） |
