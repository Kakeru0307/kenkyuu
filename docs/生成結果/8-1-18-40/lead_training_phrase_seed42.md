# リード学習データ・フレーズ型 seed42

- date: 2026-08-01
- checkpoint: なし（再学習前の合成 target）
- command: `generate_progression_lead(doowop, C, BPM95, bars=8, seed=42)`

## データの内容

- 旧定義: 均一8分、ほぼ隣接音のランダムウォーク、表拍必須
- 新定義: 2小節モチーフ `A → A' → B → cadence`
- 音価: 1 / 2 / 4 / 6 tick を混在
- モチーフ末尾に休符を残し、終止音はコードトーンへ着地
- このMIDIは学習後のモデル出力ではなく、再生成予定の学習 target 例

## 生成結果

- midi: docs/生成結果/8-1-18-40/lead_training_phrase_seed42.mid
- ノート数: 19（旧モデル出力64音に対して疎）
- onset間隔: 3 / 6 / 8 / 11 / 12 tick（均一2tickではない）
- duration: 1 / 2 / 4 / 6 tick
- 最大 onset 間隔: 12 tick
- MusPy: M1=1.0000 M2=0.0645 M3=1.0000 M4=1.0000
- 主観: 未回答

## 得られた結論

- タイミング均一・全音同音価・休符なしの3問題は、合成target上では解消
- 実際のリード生成へ反映するには lead pairs の再生成とcheckpoint再学習が必要
