# 生成音楽の評価基準（スライド用）

> 発表・スライド作成用。1 見出し ≒ 1 スライドを想定。  
> 教授フィードバックへの回答：不安定な評価 → 既存研究に基づく「良い音楽」の操作的定義。  
> 関連: [project-summary-slides.md](./project-summary-slides.md)

---

## スライド 1 — タイトル

**生成音楽の評価基準の見直し**

- 課題：現状の評価が不安定で、学術的に弱い
- 方針：既存研究で定義された評価軸を採用する
- 本日：調査結果・採否・本研究での操作的定義

---

## スライド 2 — 候補論文① Chu et al. 2022（主観）

**論文**  
Chu et al., *An Empirical Study on How People Perceive AI-generated Music*, CIKM 2022  
DOI: [10.1145/3511808.3557235](https://doi.org/10.1145/3511808.3557235)

**定義されている主観軸（9）**

Overall / Creativity / Naturalness / Melodiousness / Richness / Rhythmicity / Correctness / Structureness / Coherence

**良い点**

- 軸ごとに定義文がある（「基準を決めよ」に直結）
- シンボリック生成向けで本研究の出力形式に近い
- 先行指標を整理した上での提案で出典として説明しやすい

**悪い点**

- 9軸は評定負担・ばらつきが増えやすい
- Coherence 原義は「参照曲との類似」寄りで、条件付き生成への適用に注意が必要
- 客観指標がない（単独では再現性が弱い）

---

## スライド 3 — 候補論文② Yang & Lerch / MusPy（客観）

**論文・ツール**

- Yang & Lerch, *On the evaluation of generative models in music*, Neural Computing and Applications, 2020  
  DOI: [10.1007/s00521-018-3849-7](https://doi.org/10.1007/s00521-018-3849-7)
- MusPy metrics（実装）: [muspy.readthedocs.io](https://muspy.readthedocs.io/en/stable/doc/metrics.html)

**測るもの（例）**

- Pitch / Rhythm 特徴分布の近さ（OA, KLD）
- `pitch_in_scale_rate`, `empty_beat_rate`, `scale_consistency`, `polyphony` 等

**良い点**

- 再現可能・自動化可能（「不安定」批判への直接回答）
- シンボリック評価の定番

**悪い点**

- 「美しいか」は測れない（分布の近さ ≠ 聴感品質）
- 単独では「良い音楽」の定義にならない

---

## スライド 4 — 候補論文③ SongEval / MusicEval（参考）

**SongEval (2025)** — [arXiv:2505.10793](https://arxiv.org/abs/2505.10793)

- 5軸: coherence / memorability / vocal naturalness / structure clarity / overall musicality
- **良い点**: 美的語彙が明確、専門家アノテーション
- **悪い点**: フル尺・ボーカル曲寄り。本研究の短尺 MIDI とはドメイン不一致 → **本採用しない**

**MusicEval (2025)** — [arXiv:2501.10811](https://arxiv.org/abs/2501.10811)

- 2軸: overall musical impression / text–music alignment
- **良い点**: 軸が少なく、専門家 MOS の型が明確
- **悪い点**: 「良さ」が Overall に圧縮され分解が粗い → 主観の本体としては Chu を優先

---

## スライド 5 — 実務的結論（採用セット）

| 層 | 出典 | 役割 |
|---|---|---|
| **A 音楽品質（主観）** | Chu 2022 から絞る | 主結果 |
| **C 客観** | Yang & Lerch / MusPy | 再現性・破綻チェック |

**使わないもの**

- SongEval 5軸のそのまま採用
- Chu の全9軸（負担過大）

---

## スライド 6 — 採用する主観軸（最終）

尺度: **5点リッカート**（定義文を画面に固定）

| ID | 軸 | 評価者への定義文 | 出典 |
|---|---|---|---|
| Q1 | Melodiousness | 旋律として音楽的・調和的に聞こえるか | Chu |
| Q2 | Rhythmicity | 拍・リズムにまとまりがあるか | Chu |
| Q3 | Structureness | 反復や展開など、構造的なまとまりがあるか | Chu |
| Q4 | Correctness | 突然の無音・不自然な飛びなど、技術的破綻が少ないか | Chu |
| O1 | Overall | 総合的に、この曲の聴感品質に満足できるか | Chu |

**初期は使わない（Chu から除外）**  
Creativity / Richness / Naturalness / Coherence

---

## スライド 7 — 採用する客観軸（最終）

| ID | 指標 | 見るもの | 位置づけ |
|---|---|---|---|
| M1 | `pitch_in_scale_rate` | 調内音の割合 | 調性破綻の少なさ |
| M2 | `empty_beat_rate` | 空拍の割合 | スカスカ／過密の異常 |
| M3 | `polyphony` 等 | 同時発音密度 | 学習分布からの逸脱 |
| M4 | `scale_consistency` | 調の一貫性 | 調のふらつき |
| M5 | Yang–Lerch OA/KLD | Pitch/Rhythm 分布近さ | データらしさ |

**論文に明記する約束**  
客観指標は美学的優劣の定義ではなく、**技術的妥当性（壊れていないか）** の検証に用いる。

---

## 参考リンク（スライド末尾用）

| 文献 | URL |
|---|---|
| Chu et al. CIKM 2022 | https://doi.org/10.1145/3511808.3557235 |
| Yang & Lerch 2020 | https://doi.org/10.1007/s00521-018-3849-7 |
| MusPy metrics | https://muspy.readthedocs.io/en/stable/doc/metrics.html |
| SongEval | https://arxiv.org/abs/2505.10793 |
| MusicEval | https://arxiv.org/abs/2501.10811 |
| Survey (metrics) | https://arxiv.org/abs/2509.00051 |
