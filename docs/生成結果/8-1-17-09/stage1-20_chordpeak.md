# stage1-20 chord-peak デコード（B ピーク + C コード共通 onset）

- date: 2026-08-01
- checkpoint: prttype/checkpoints/backing/unet_last.pt（epochs=20, lr=0.001）
- command: `py -3 prttype/generate_backing.py --seed 42`

## データの内容

- PATTERN_WEIGHTS: progression_strum 100%
- ATTACKS_PER_BAR_WEIGHTS: {8: 1.0}（8分固定）
- PLACEMENT_WEIGHTS: {even: 1.0}（※このckptは even バグ修正前データで学習: tick 0,2,4,6,9,11,13,15）
- DEFAULT_SYNTHETIC_COUNT: 2000 / epochs: 20
- 変更点: デコードを chord-peak（既定ON）に置換

## 生成結果

- midi: docs/生成結果/8-1-17-09/stage1-20_chordpeak.mid
- ノート数: 149
- MusPy: M1=1.0 M2=0.0 M3=2.89 M4=1.0
- 主観: 未回答

デコード内容（`midi_to_patch.decode_chord_peak_pianoroll`）:
- B: envelope のピーク／立ち上がり + 活性帯の局所最大で onset 検出
- C: 複数 pitch で共通 onset を共有（和音の同時発音）
- ピーク不足時のみ従来の閾値量子化 + 2tick 分割へフォールバック
- OFF: `--no-chord-peak-decode`

## 得られた結論

- 実モデル出力（max≈0.63・平坦寄り）でも HIT し、149 ノートの和音ストロークを生成。
- M1/M4=1.0・M2=0.0 で妥当性は維持。前回 th=0.3（179 notes, 2tick分割）比で
  和音の同時性が改善する読み方に変更。
- 次: even バグ修正済みデータで再学習し、旧デコード（--no-chord-peak-decode）と比較。
