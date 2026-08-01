# stage1-20 chord-peak デコード + release_gap_ticks=2

- date: 2026-08-01
- checkpoint: prttype/checkpoints/backing/unet_last.pt（epochs=20, lr=0.001）
- command: `py -3 prttype/generate_backing.py --seed 42`

## データの内容

- PATTERN_WEIGHTS: progression_strum 100%
- ATTACKS_PER_BAR_WEIGHTS: {8: 1.0}
- PLACEMENT_WEIGHTS: {even: 1.0}（※このckptは even バグ修正前データで学習）
- DEFAULT_SYNTHETIC_COUNT: 2000 / epochs: 20
- 変更点: `release_gap_ticks` 1 → **2**（8分=2tick 分の隙間）

## 生成結果

- midi: docs/生成結果/8-1-17-18/stage1-20_chordpeak_gap2.mid
- ノート数: 152
- MusPy: M1=1.0 M2=0.0 M3=3.08 M4=1.0
- 同一ピッチ: overlap=0 / touch=3 / gap=141
- duration 分布: 1tick×147, 2tick×5
- 主観: 未回答

## 得られた結論

- 既定のリリース隙間を 2tick（8分1つ分）に変更。
- onset 間隔が約 2tick の区間では sustain が入らずほぼ duration=1 になるため、
  gap=1 時と touch/gap カウントはほぼ同じ。広い onset 間隔では隙間が広がる。
- M3b polyphony_rate は gap1時の 0.48 → 0.32（短いノート寄り）。
