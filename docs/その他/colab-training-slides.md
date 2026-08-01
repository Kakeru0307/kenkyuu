# Colab 学習 & 学習データ（スライド用）

> 発表・スライド作成用。1 見出し ≒ 1 スライドを想定。  
> 詳細ロードマップ: [implementation-roadmap.md](./implementation-roadmap.md) §11

---

## スライド 1 — タイトル

**U-Net ギター生成：Google Colab 学習とデータセット方針**

- 研究プロトタイプ：MIDI → パッチ → smp U-Net → MIDI
- 学習は **Google Colab（GPU）**、推論・確認は **ローカル**
- 確定方針：**二段階学習**（合成データ → 実データ）

---

## スライド 2 — 二段階学習（確定方針）

| 段階 | データ | 目的 | 出力 |
|---|---|---|---|
| **Stage 1** | **合成** MIDI | コード・音階・ストロークの **基本ルール** | `stage1/unet_last.pt` |
| **Stage 2** | **Guitar-TECHS** 等 | **ギターらしさ・音価・技法** | `stage2/unet_last.pt` ★推論用 |

Stage 2 は Stage 1 の重みを **読み込んでから** 学習（ファインチューニング）。

**採用しない:** 合成 90% + 実 10% を **1 回の学習で混ぜる**（実データの効きが弱い）

---

## スライド 3 — 学習ペア（input / target）

| | 内容 |
|---|---|
| **入力（骨格）** | 発音点のみ（`onset_to_full` … 暫定） |
| **正解（target）** | 音価・持続付きのパッチ |
| **U-Net の役割** | 骨格 → フレーズらしい音価へ変換 |

将来（§10）: 音階骨格・ストローク骨格など **入力の定義を拡張**予定。  
Stage 1 / 2 で **同じタスク定義**を使う。

---

## スライド 4 — データセット規模（目標 ≈ 1 万パッチ）

| ソース | 目標パッチ数 | 現状 | 用途 |
|---|---:|---:|---|
| **合成** | 6,000〜7,000 | 未作成 | Stage 1 |
| **Guitar-TECHS** | 3,000〜4,000 | **≈ 3,381** | Stage 2 |
| **合計** | **≈ 10,000** | 3,381 | — |

- カウント単位は **パッチ数**（MIDI 本数ではない）
- 合成は最初 **500〜1,500 MIDI** から試作 → 足りなければ増量

---

## スライド 5 — Stage 1：合成データの内容

**生成方法:** Python で自動生成（`generate_phrases.py` … 予定）

| 要素 | バリエーション例 |
|---|---|
| 和声 | 基本コード（C, Am, G7 …） |
| 音階 | 各キーのスケール |
| リズム | 4 分・8 分のシンプルストローク |
| BPM | 60〜150 |
| 楽器 | program **27**（エレキギター / category 3） |

---

## スライド 6 — Stage 2：実データ（Guitar-TECHS）

| カテゴリ | 内容 | MIDI 数（目安） |
|---|---|---:|
| P1/P2_chords | コード練習 | 56 |
| P1/P2_scales | 音階 | 24 |
| P1_techniques | ベンド・ハーモニクス等 | 5 |
| P3_music | 短い曲 | 12 |
| **合計** | 6 弦トラック・program 27 済 | **98** |

- ソース: [Guitar-TECHS (Zenodo)](https://zenodo.org/records/14963133)
- パッチ化後: **`data/pairs/guitar-techs/`**（Colab 上で生成）

---

## スライド 7 — `colab_train/` に含めるもの

| 含める | 含めない |
|---|---|
| 学習スクリプト（`train.py` 等） | `venv/` |
| raw MIDI（98 曲・0.1MB） | **`data/pairs/`（約 5GB）** |
| `colab_train.ipynb` | 推論用 `inference.py` |
| `checkpoints/`（Git push 用） | ViTex 本体 |

**pairs は Colab 上で `prepare_dataset.py` を実行して都度生成**

---

## スライド 8 — Colab 実行手順（Stage 1）

1. GitHub から `colab_train.ipynb` を開く
2. `REPO_URL` / `GITHUB_TOKEN`（PAT）を設定
3. `pip install -r requirements.txt`
4. 合成 MIDI 生成 → `remap_guitar_programs.py`
5. `prepare_dataset.py` → `data/pairs/synthetic/`
6. `train.py` → `checkpoints/stage1/unet_last.pt`

```bash
python train.py --pairs-dir data/pairs/synthetic \
  --epochs 20 --batch-size 16 \
  --checkpoint-dir checkpoints/stage1 --pos-weight 10
```

---

## スライド 9 — 比較：学習方式と音楽品質

| 方式 | 音楽品質 | 採用 |
|---|---|:---:|
| **二段階**（合成 → TECHS） | ◎ ギターらしさを出しやすい | ✅ |
| 1 万パッチ混ぜ（実 30%） | ○ 実データが増えれば有効 | △ |
| 合成 90% + 実 10% 一括 | △ 地味・機械的 | ❌ |
| TECHS のみ（Stage 2 だけ） | ○ 量は限られる | 試作可 |

---

## スライド 10 — 今後の課題（関連）

| # | 課題 |
|---|---|
| 1 | U-Net **骨格（入力）** の正式定義（§10） |
| 2 | `generate_phrases.py` 実装 |
| 3 | `train.py --resume` 実装 |
| 4 | Colab ノートブックを Stage 1 / 2 両対応に更新 |
| 5 | ViTex 軽量注釈（Phase 5） |

---

## スライド 11 — まとめ

1. **Colab** で GPU 学習 → **checkpoint だけ Git** → ローカルで推論
2. **二段階学習**: 合成（基本）→ Guitar-TECHS（仕上げ）
3. 目標 **≈ 1 万パッチ**（合成 6〜7k + 実 3〜4k）
4. 最終モデルは **Stage 2**
5. 著作権安全な合成 + 表現力のある実データの **組み合わせ**

---

## 付録 — スピーカーノート（任意）

<details>
<summary>補足説明用メモ</summary>

- **ViTex 形式:** RGB 色ルールは不要。0/1/2 pianoroll が標準。
- **program 27:** 方針 B。DAW でもギター音。category 3 に統一。
- **weighted MSE:** 無音ばかり予測する問題への対策（`pos-weight 10`）。
- **P3_music 試作:** Stage 2 のみの短時間試験に `p3-music`（87 パッチ）を使用済み。
- **Git サイズ:** pairs は push しない。raw 0.1MB + コード + checkpoint のみ。

</details>

---

## 関連ドキュメント

| ファイル | 内容 |
|---|---|
| [implementation-roadmap.md](./implementation-roadmap.md) | 全フェーズ・§10 骨格設計・§11 学習方針・§12 BPM |
| [project-summary-slides.md](./project-summary-slides.md) | プロジェクト全体サマリー |
| [../colab_train/README.md](../colab_train/README.md) | Colab リポジトリ操作手順 |
