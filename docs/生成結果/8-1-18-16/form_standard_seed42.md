# 構成レイヤ v1 煙テスト（standard / seed42）

- date: 2026-08-01
- checkpoint: prttype/checkpoints/backing/unet_last.pt（epochs=20, lr=0.001）
- command: `python generate_form.py --seed 42 --output midi/form_smoke_seed42.mid`

## データの内容

- 構成: テンプレ `standard`（イントロ→Aメロ→Bメロ→Aメロ→サビ→アウトロ、48小節）
- 曲の色: catalog 乱択（seed=42）→ doowop / C / BPM 95 / major
- 区間盛り上がり: イントロ・アウトロ=疎デコード、サビ=密デコード、他=既定
- スケルトン MIDI 保存なし。最終出力は1本連結

## 生成結果

- midi: docs/生成結果/8-1-18-16/form_standard_seed42.mid
- 区間ノート数: イントロ111 / Aメロ168 / Bメロ168 / Aメロ168 / サビ185 / アウトロ111（合計911）
- MusPy: M1=0.9989 M2=0.0000 M3=2.8525 M4=0.9989
- 主観: 未回答（Q1–O1 を依頼中）

## 得られた結論

- 構成レイヤの通し生成（テンプレ抽選→区間デコード差→1本連結）は動作する
- サビがイントロより密（185 vs 111）で、盛り上がり配線は聴感確認前の定量目安としては効いている
- BメロとAメロが同ノート数（168）で、進行固定のもと区別が弱い既知弱点は残る
