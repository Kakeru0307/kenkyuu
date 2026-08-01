# stage1-20 even placement + 2-tick blob segmentation

- date: 2026-07-27
- checkpoint: prttype/checkpoints/backing/unet_last.pt（epochs=20, lr=0.001）
- command: `py -3 prttype/generate_backing.py --seed 42`

## データの内容

- PATTERN_WEIGHTS: progression_strum 100%
- ATTACKS_PER_BAR_WEIGHTS: {8: 1.0}（8分固定）
- PLACEMENT_WEIGHTS: {even: 1.0}（偶数配置固定）
- DEFAULT_SYNTHETIC_COUNT: 2000
- epochs: 20

## 生成結果

- midi: docs/生成結果/7-27-16-03/stage1-20_even_segment.mid
- ノート数: 99（修正前: 25）
- MusPy: M1=1.0 M2=0.125 M3=2.25 M4=1.0
- 主観: 未回答

デコード修正内容:
- `patch_to_midi._extract_notes_from_channel` にサステイン値(2)を含まない
  連続オンセットブロブを 2-tick 単位で強制分割するロジックを追加。
- モデルが onset/sustain を学習できていない（max=0.63 で sustain_th=0.85 未達）
  ため、全活性セルが 1 に量子化され長音になっていた問題を回避。

## 得られた結論

- 25 → 99 ノートに改善。8分刻みのストローク感が出るようになった。
- モデルは「どのピッチ帯域をいつ鳴らすか」は学習済み（M1/M4 = 1.0）。
- 「onset と sustain の区別」はまだ学習できていない（max 出力 0.63 < sustain_th 0.85）。
- 今後: さらに学習を続ければ sustain 値が上がり、自然な再アタックが生まれる可能性あり。
  デコード側の 2-tick 分割はその暫定処置。
