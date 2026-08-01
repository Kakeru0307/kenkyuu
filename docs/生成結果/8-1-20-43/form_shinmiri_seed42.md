# form + lead 統合（しんみり / seed42）

- date: 2026-08-01
- checkpoint: `prttype/checkpoints/backing/unet_last.pt` + `prttype/checkpoints/lead/unet_last.pt`
- prior: `prttype/checkpoints/prior/prior_last.pt`
- emotion_clf: `prttype/checkpoints/emotion/emotion_clf.pt`
- command: `python generate_form.py --text "しんみりとした曲" --seed 42 --output midi/form_shinmiri_seed42.mid`

## データの内容
- 入口: 構成レイヤ `generate_form` + バッキング + リード（2トラック）
- 文→WRIME→学習済み emotion_clf → structure prior
- emotion_target=`sadness` / progression=`minor_simple_14` / key=`A` / BPM=`65.6` / energy=`low`
- テンプレ: `standard`（48bars: イントロ→Aメロ→Bメロ→Aメロ→サビ→アウトロ）
- リード onset_th はセクション energy に連動（low=0.40 / mid=0.30 / high=0.22）
- バッキング power-chord attack をリード側で blocking

## 生成結果
- midi: docs/生成結果/8-1-20-43/form_shinmiri_seed42.mid
- トラック: Backing, Lead
- 総ノート: 1157（区間別 backing/lead は生成ログ参照）
- MusPy: M1=0.9957 M2=0.0052 M3=2.7836 M4=0.9957
- 主観: 未回答

## 得られた結論
- しんみり文が sadness・低BPM・Aマイナーに落ち、emotion_clf 修正が効いている
- 構成レイヤにリードを載せても Backing/Lead が別トラックとして分離できる
- 客観指標は妥当。聴感は Q1–O1 待ち
