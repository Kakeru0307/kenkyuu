# DadaGP を使った structure prior 学習データ作成

## 目的

曲の感情と区間構造の対応を学習するための教師データを作る。
現状の structure prior は区間長が8小節固定で感情に基づいていない。
DadaGP の実際の曲から「この感情の曲はこういう構造になる」を学ぶ。

## 全体の流れ

```
DadaGP の GuitarPro ファイル
  ↓ PyGuitarPro で MIDI に変換
  ↓ FluidSynth + SoundFont で WAV に変換
  ↓ Essentia の VA 回帰モデルを通す
  → valence と arousal の数値が出る

同じ GuitarPro ファイルから
  ↓ セクション名を抽出
  ↓ 区間長・BPM・調を取り出す
  → 区間構造のパラメータが出る

2つを合わせる
  → (valence, arousal, ジャンル) と (区間長・BPM・調) のペアが揃う
  → structure prior の学習データとして使う
```

## 各ステップの詳細

### MIDI → WAV 変換

- ツール: FluidSynth
- 音源: この研究で使っているのと同じ SoundFont を使う
- SoundFont が合成音のため、Essentia の精度が実音源より下がる可能性がある

### Essentia による VA 推定

- モデル: Essentia の arousal/valence 回帰モデル
- 学習データ: DEAM、emoMusic、MuSe の3種類から選ぶ
- 精度: 激しさ r=0.81、明るさ r=0.67
- 出力: valence と arousal がそれぞれ 1〜9 の値

### 区間構造の抽出

- GuitarPro ファイルにはセクション名が付いている曲がある
- セクション名がある曲は verse / chorus / bridge などのラベルが使える
- ない曲は区間境界検出ツールで推定する

### 学習データの形式

| 入力 | 出力 |
|---|---|
| valence, arousal | テンプレート種類 |
| valence, arousal | 区間ごとの小節数 |
| valence, arousal | BPM の分布 |

## 懸念点

- DadaGP へのアクセスがメール申請制。まず申請が必要
- SoundFont 合成音では Essentia の精度が落ちる可能性がある
- セクション名が付いていない曲の区間検出は別途ツールが必要
- 26,181 曲全部を処理するのは時間がかかる。ロック・メタル系に絞れば数千曲に減らせる

## 優先順位

1. DadaGP のアクセス申請を送る
2. Essentia のインストールと動作確認
3. 小規模で試す。100曲程度で VA 推定と区間抽出が動くか確認
4. スケールアップして学習データを作る
