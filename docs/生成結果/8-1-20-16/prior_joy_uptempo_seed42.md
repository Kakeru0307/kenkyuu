# 文→WRIME→prior→song（アップテンポ・seed42）

- date: 2026-08-01
- checkpoint: backing in=12 epochs=20 / lead in=13 epochs=20 / prior `prior_last.pt`
- command: `python generate_song.py --text "アップテンポな感じの曲" --seed 42 --eval`

## データの内容
- 入口: `generate_song` が `resolve_structure_params`（文→WRIME→4ラベル→prior）
- 解決結果: `source=prior` emotion=`joy` progression=`komuro` key=`Eb` bpm=`150` energy=`high`
- 演奏: backing 12ch + lead 13ch（power mask）

## 生成結果
- midi: docs/生成結果/8-1-20-16/prior_joy_uptempo_seed42.mid
- MusPy: M1=0.9950 M2=0.0000 M3=3.0400 M3b=0.4882 M4=0.9950
- ノート: backing 125 / lead 74 / 合計 199
- 主観: 未回答

## 得られた結論
- song 本線で prior 経由が動いた（乱択ではない）
- 「アップテンポ」→ joy / BPM150 / high は雰囲気として妥当
- 聴感は主観待ち
