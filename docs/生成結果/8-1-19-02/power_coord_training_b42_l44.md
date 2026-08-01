# Backing／Lead power chord協調 target

- date: 2026-08-01
- checkpoint: なし（再学習前の合成target）
- command: `doowop / C / BPM95 / backing seed42 / lead seed44`

## データの内容

- backing: 8分attack 64回のうちpower chord 48回（75%、設定値70%）
- lead: 2小節モチーフ型、フレーズ40%でpower chord候補
- lead条件: backing power chord attack mask
- 衝突条件: power chord同士のattack時刻だけ分離。持続時間の重なりは許容
- lead学習入力: コード骨格11ch＋BPM 1ch＋mask 1ch（計13ch）
- 学習予定: 旧12ch checkpointからresumeせず20 epoch

## 生成結果

- midi: docs/生成結果/8-1-19-02/power_coord_training_b42_l44.mid
- backing power chord attack: 48
- lead power chord attack: tick 12 / 64
- 同時attack衝突: 0
- 500曲検証でlead power chordを含むフレーズ率: 30.05%（設定40%、空き枠不足を含め目標3〜5割内）
- MusPy: M1=1.0000 M2=0.0000 M3=2.1300 M4=1.0000
- 主観: 未回答

## 得られた結論

- backing 70% power chordとlead側の非衝突power chordを、学習target／条件maskとして表現できる
- 実モデル出力への反映にはbacking再生成・再学習とlead 13ch新規20 epoch学習が必要
