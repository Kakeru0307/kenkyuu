# backing+lead（新ckpt・seed42）

- date: 2026-08-01
- checkpoint: `prttype/checkpoints/backing/unet_last.pt`（epochs=20, in=12） / `prttype/checkpoints/lead/unet_last.pt`（epochs=20, in=12）
- command: `python generate_song.py --seed 42 --eval`

## データの内容
- backing: Stage1 20 epoch、入力 12ch（tonal11+BPM）。power chord 70% 定義の再学習想定
- lead: 20 epoch、入力 **12ch**（power attack mask なし。想定していた 13ch ではない）
- 構造パラメータは分布サンプル（seed=42 → doowop / C / BPM95 / 8bars）

## 生成結果
- midi: docs/生成結果/8-1-19-37/power_b12_l12_seed42.mid
- MusPy: M1=1.0000 M2=0.0000 M3=2.7746 M3b=0.4331 M4=1.0000
- ノート: backing 159 / lead 58 / 合計 217
- 主観: 未回答

## 得られた結論
- 両 ckpt で統合生成は成功
- lead ckpt が 12ch のため、今回の推論では power-mask 条件・power dyad 許容は無効（`in_channels >= 13` 分岐に入らない）
- 客観指標はスケール内・無音ビートなしで問題なし。聴感は主観待ち
