# STROKE_GROOVE 再学習 ckpt クリップ確認

- date: 2026-11-23
- checkpoint: prttype/checkpoints/backing/backing_cvae_last.pt（epochs=20, lr=0.001, cvae_film, latent_dim=16）
- command: `venv\Scripts\python.exe prttype\generate_backing.py --seed 42 --out midis\stroke_ckpt_s42.mid`

## データの内容
- Colab で STROKE_GROOVE 入り synth 4000 本を再合成→scratch 学習した ckpt をローカルへ持参
- PATTERN_WEIGHTS: progression_strum 0.85 / progression_arpeggio 0.15
- STROKE_GROOVE_WEIGHTS: eight_even 主軸（他 six_teen_busy / backbeat_chop 等）
- BPM 合成範囲 60–200、pitch 38–88
- 生成時: catalog prior（doowop / B / 130 / major / 8 bars）、peak decode・2tick segment は現行デフォルト OFF

## 生成結果
- midi: docs/生成結果/11-23-07-34/stroke_ckpt_s42.mid
- ノート数: 34
- MusPy: M1=0.9706 M2=0.0000 M3=3.5312 M3b=0.9922 M4=0.9706
- 主観: 未回答

## 得られた結論
- 新 ckpt でローカル再生成は通る（保存パス・推論ともに問題なし）
- スケール整合は高いが、奏法・ストローク感の良し悪しは聴感確認が必要
