# stage1-20 chord-peak デコード + リリース隙間（release_gap_ticks=1）

- date: 2026-08-01
- checkpoint: prttype/checkpoints/backing/unet_last.pt（epochs=20, lr=0.001）
- command: `py -3 prttype/generate_backing.py --seed 42`

## データの内容

- PATTERN_WEIGHTS: progression_strum 100%
- ATTACKS_PER_BAR_WEIGHTS: {8: 1.0}（8分固定）
- PLACEMENT_WEIGHTS: {even: 1.0}（※このckptは even バグ修正前データで学習: tick 0,2,4,6,9,11,13,15）
- DEFAULT_SYNTHETIC_COUNT: 2000 / epochs: 20
- 変更点: chord-peak デコードに `release_gap_ticks=1` を追加（打ち直し直前を無音化）

## 生成結果

- midi: docs/生成結果/8-1-17-15/stage1-20_chordpeak_gap.mid
- ノート数: 152
- MusPy: M1=1.0 M2=0.0 M3=2.96 M4=1.0
- 同一ピッチの隣接関係: overlap=0 / touch=3 / gap=141
  （前回 8-1-17-09: overlap=0 / touch=132 / gap=9）
- 主観: 未回答

## 得られた結論

- 「音の切り替わりで被って聞こえる」原因は学習状態ではなくデコード。
  sustain を次の onset 直前まで埋めていたため切れ目ゼロ（touch=132）だった。
- 学習データ側は duration=1 + 2tick 間隔で 1tick の隙間があるため、
  デコードに `release_gap_ticks=1` を入れてデータの性質に揃えた。
- touch 132 → 3、gap 9 → 141。M1/M4=1.0 は維持。
- 次: even バグ修正済みデータで再学習し、onset 位置そのものの精度を上げる。
