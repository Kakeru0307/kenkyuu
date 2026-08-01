# backing+lead（lead 13ch・seed42）

- date: 2026-08-01
- checkpoint: `prttype/checkpoints/backing/unet_last.pt`（epochs=20, in=12） / `prttype/checkpoints/lead/unet_last.pt`（epochs=20, in=13）
- command: `python generate_song.py --seed 42 --eval`

## データの内容
- backing: Stage1 20 epoch、入力 12ch（tonal11+BPM）。power chord 70% 再学習
- lead: 20 epoch、入力 **13ch**（骨格11+BPM+power attack mask）。hash が backing と別
- 構造パラメータは分布サンプル（seed=42 → doowop / C / BPM95 / 8bars）

## 生成結果
- midi: docs/生成結果/8-1-20-02/power_b12_l13_seed42.mid
- MusPy: M1=1.0000 M2=0.0000 M3=3.1809 M3b=0.5984 M4=1.0000
- ノート: backing 159 / lead 76 / 合計 235
- ピッチ: backing 低域8音、lead は上位まで含む15音（onsetも一致しない）
- 主観: 未回答

## 得られた結論
- 正しい 13ch lead ckpt では backing と別演奏になる（先の同一ファイル誤配置は解消）
- 客観指標はスケール内・無音ビートなし。聴感は主観待ち
