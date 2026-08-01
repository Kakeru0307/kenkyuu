# Stage1 epochs=20（strum 100% 学習）の生成

- date: 2026-07-27
- checkpoint: `prttype/checkpoints/backing/unet_last.pt`（mtime 2026-07-27 13:47; epochs=20, lr=0.001, in_channels=12 → Stage1）
- command: `python generate_backing.py --text "元気なストローク" --seed 3 --output docs/生成結果/7-27-14-12/stage1-20_strum100.mid`

## データの内容

- **学習パターン**: リポジトリ現状どおり `PATTERN_WEIGHTS` = `progression_strum: 1.0`（arp/scale 等 0）。ユーザーが持ち込んだ学習成果物として記録
- Stage / タスク: Stage1、`downbeat_chord`
- 比較対象: `docs/生成結果/7-27-13-22/`（当時 strum≈45% / arp≈40% 学習の epochs=20）
- 推論: ルール後処理なし（`chord_align` / loop-raise / chords-per-bar 強制は削除済み）。骨格は小節頭コードのみ

## 生成結果

- midi: `docs/生成結果/7-27-14-12/stage1-20_strum100.mid`
- 構造サンプル: progression=deceptive, key=Eb, bpm=150, mode=major, bars=8（seed=3 で前回と同じ prior 出方）
- ノート数 18 / unique onset 12。strokes/bar は {0:1, 1:4, 2:1, 4:1, 5:4, 6:1}（bar3・7 は onset なし）。onset あたり音数 1〜3（平均 1.5）→ **和音の同時発音が一部出た**
- MusPy: M1=1.0000 M2=0.0000 M3=1.6480 M4=1.0000（M3b polyphony_rate=0.1890）
- 主観: 未回答

## 得られた結論

- 前回（単音・M3=1.0・empty_beat=0.20）より **和音寄り・空白拍減少**（M3↑、M2=0）で、strum 100% 学習の方向と整合的
- それでも **8分刻み級の密ストロークには未達**（小節あたり 1〜4 onset、空小節あり）
- 次の論点は密度（attacks/bar）とストローク間の切れ目。閾値・デコードは副次、データ／学習の刻み語彙が本筋
