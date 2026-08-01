# Stage1 epochs=10（strum 100%・刻み8固定）の生成

- date: 2026-07-27
- checkpoint: `prttype/checkpoints/backing/unet_last.pt`（mtime 2026-07-27 15:15; epochs=10, lr=0.001, in_channels=12 → Stage1）
- command: `python generate_backing.py --text "元気なストローク" --seed 3 --output docs/生成結果/7-27-15-17/stage1-10_strum100_atk8.mid`

## データの内容

- **学習パターン**: `progression_strum: 1.0`（arp/scale 等 0）
- **刻み**: `ATTACKS_PER_BAR_WEIGHTS = {8: 1.0}`（8分固定実験）
- BPM 合成範囲: 60–150（未拡張）
- Stage / タスク: Stage1、`downbeat_chord`、epochs=10
- 推論: ルール後処理なし。骨格は小節頭コードのみ
- 比較: `7-27-14-12`（strum100・刻み多様・epochs=20）、`7-27-13-22`（strum/arp 混在）

## 生成結果

- midi: `docs/生成結果/7-27-15-17/stage1-10_strum100_atk8.mid`
- 構造サンプル: progression=deceptive, key=Eb, bpm=150, mode=major, bars=8（seed=3）
- ノート数 15 / unique onset 9。strokes/bar はほぼ **1回/小節**（多くが半拍目 tick8）。onset あたり 1〜3 音（平均 ≈1.67）→ 和音は一部出る
- 被覆: 各小節の後半に長い sustain が多く、8分刻みの切れ目は見えない
- MusPy: M1=1.0000 M2=0.4688 M3=1.7368 M4=1.0000（M3b=0.1102）
- 主観: 未回答

## 得られた結論

- **8分刻み（目標 8 onset/小節）には未達**。むしろ小節あたり 1 ストローク寄りで、前回の刻み多様・epochs=20（最大4/小節）より密度は下がった
- 和音（M3≈1.74）は混在期より良いが、空拍率 M2=0.47 と高くスカスカ
- epochs=10 は短い可能性。データが本当に attacks=8 で再生成されたかも要確認（manifest の N 分布）。次は epochs を増やすか、学習データ側の onset 配置（even 8分）を可視化確認
