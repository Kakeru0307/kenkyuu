# 生成結果ログ

学習データやチェックポイントを変えたときの **バッキング生成実験** を、1回あたり1フォルダで残す置き場。

各ランは `docs/生成結果/<M>-<D>-<HH>-<mm>/`（例: `7-27-13-33`）直下に **md と midi を同居**。各エントリの内容は次の3点のみ:

1. データの内容
2. 生成結果（代表 MIDI 1本）
3. 得られた結論

規約: `.cursor/rules/generation-results.mdc`  
手順（生成→評価→保存）: `.cursor/skills/generate-and-log/SKILL.md`  
評価軸: `.cursor/rules/post-generation-eval.mdc` / `docs/evaluation-criteria-slides.md`
