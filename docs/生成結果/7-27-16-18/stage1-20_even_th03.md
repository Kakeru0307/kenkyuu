# stage1-20 even placement + onset_th=0.3

- date: 2026-07-27
- checkpoint: prttype/checkpoints/backing/unet_last.pt（epochs=20, lr=0.001）
- command: `py -3 prttype/generate_backing.py --seed 42`

## データの内容

- PATTERN_WEIGHTS: progression_strum 100%
- ATTACKS_PER_BAR_WEIGHTS: {8: 1.0}（8分固定）
- PLACEMENT_WEIGHTS: {even: 1.0}（※バグ修正前: tick 0,2,4,6,9,11,13,15 で学習済み）
- DEFAULT_SYNTHETIC_COUNT: 2000
- epochs: 20

## 生成結果

- midi: docs/生成結果/7-27-16-18/stage1-20_even_th03.mid
- ノート数: 179
- MusPy: M1=1.0 M2=0.0 M3=2.77 M4=1.0
- 主観: かなり良いバッキングが出来ている（ユーザー評価）

変更点（前回比）:
- `inference.onset_th`: 0.5 → **0.3**（デフォルト変更）
- `patch_to_midi._extract_notes_from_channel`: all-onset blob を 2-tick 分割

## 得られた結論

- onset_th=0.3 + 2-tick blob 分割でバッキングとして聴感上良好な出力が得られた。
- M1/M4=1.0（スケール整合性完璧）、M2=0.0（空拍なし）、179 ノートで十分な密度。
- even placement のバグ（tick 9 ズレ）は学習済みチェックポイントに残っている。
  次回: バグ修正済みデータで Colab 再生成 → 再学習 → 比較。
