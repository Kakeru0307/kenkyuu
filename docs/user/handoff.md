# Handoff: 研究プロジェクト統合（2026-07-20）

- saved_at: 2026-07-20 15:30
- workspace: c:\Users\kake0\github\研究
- mode: agent
- trigger: user（`docs/user/*_handoff.md` 8件を統合）

---

## Goal（現在地）

**ふわっとした日本語 → 感情 → 学習分布に沿った進行・BPM・キー → U-Net で backing/lead →（将来）長尺曲**

- 利用者は**進行を指定しない**（内部で manifest 等から選択）
- 刻み（何回・どこで弾くか）は**モデルが創造的に決める**（骨格に N を載せない）
- BPM は**条件チャンネル**で密度をバイアス（in=12: tonal11 + BPM1）
- 長尺（〜3分）・曲構成の賢さは **U-Net 外**（フォーム層。まずテンプレ A/B）

---

## やったこと（詳細）

### A. プロダクト・方針

| 項目 | 内容 |
|------|------|
| 製品前提 | ユーザーは進行を渡さない。学習データ／内部選択で決める |
| MuseCoco | 試作後 **本線から除外**（「しんみり」→ tension/BPM140 等の不一致） |
| 感情解析 | **WRIME** → **V/A**（Plutchik固定座標）。`emotion_va.py`。`emotion_clf`・4ラベルは廃止 |
| 進行・BPM 選択 | structure prior（入力 WRIME8+VA2=10次元）。旧 in=12 ckpt は非互換 |
| seed | テキストからの hash 廃止。`--seed` 任意、なしは真ランダム |
| 生成入口 | `generate_from_prompt.py` → `prompt_to_params.py` → `generate_song` |

### B. 合成データ・リズム（makeData）

| 項目 | 内容 |
|------|------|
| 規模 | **15,000 本**入れ替え（旧 6k 追記しない。`--force` 必須） |
| パターン | `progression_strum` / `progression_arpeggio` 各 45% |
| 刻み N | 1..16（中心 4–8）。`resolve_onset_ticks(N, placement)` |
| placement | even / front / back / offbeat / sparse_random |
| articulation | solid / staccato / mixed / sustained / rests（1曲=1ノリ） |
| **BPM↔N 相関** | `makeData/rhythm.py`：`sample_attacks_for_bpm(bpm)` で合成時に連動 |
| arp meta | `attacks_per_bar` を汚染しない → **`arp_notes_per_bar`** に分離 |
| front バグ | N>8 で後半に染み出す問題を修正 |
| CLI | 既存 MIDI ありで `--force`/`--append` なし → **エラー** |

### C. 推論パイプライン（prompt → params）

| 項目 | 内容 |
|------|------|
| WRIME → emotion | joy / sadness / calm / tension / na |
| manifest 選択 | 感情 → mode/family/BPM 帯 → 段階的緩和 → 1 本サンプル |
| **やめたこと** | manifest の N / placement で生成制御（音に効かないため） |
| フォールバック | manifest 失敗時のみ旧カタログ |
| heuristic | WRIME 失敗時。キーワードで sadness/calm 等を上書き |

### D. モデル・学習（BPM 条件）

| 項目 | 内容 |
|------|------|
| U-Net 入力 | **12ch**（tonal 11 + 正規化 BPM 1）→ 出力 11 |
| `density_cond.py` | `bpm_to_unit`, `make_bpm_cond_map`, `MODEL_IN_CHANNELS=12` |
| ペア | `prepare_dataset.py` が `*_cond.npy`（BPM unit スカラー）を保存 |
| Dataset | `PatchPairDataset` が tonal + cond を concat |
| 推論 | `predict_patches(..., bpm=...)` で条件 ch 付与 |
| ckpt | `in_channels=12` を save/load。**旧 in=11 ckpt は非互換** |
| colab 同期 | `makeData/`, `density_cond.py`, `dataset.py`, `prepare_dataset.py`, `model.py`, `inference.py`, `generate_backing.py`, notebook `SYNTHETIC_COUNT=15000` |

### E. 以前に完了していた基盤（引き続き有効）

| 項目 | 内容 |
|------|------|
| 進行つき合成 | 6k → 15k へ拡張。進行カタログ・骨格生成 |
| 生成 API | `generate_backing` / `generate_lead` / `generate_song`（2トラック統合） |
| 骨格 | `downbeat_chord`（小節頭コード）。`full` は精度低下 |
| 音価多様化（案1） | strum 5 articulation 型。1曲=1ノリ |
| CVAE（案A） | `prttype/model.py` に実装済み。拡散は使わない |
| `--resume` | `train.py` に実装 |
| colab 整理 | 推論スクリプト等を `prttype/` 側に集約 |
| handoff 運用 | load Rule / save Skill / auto|user|version 構成 |

### F. 調査・設計で確定したこと（実装前含む）

| 項目 | 内容 |
|------|------|
| 骨格に N 固定 | **ユーザー却下**（創造的に埋めたい） |
| 長尺 3 分 | 8 小節 U-Net を **フォーム層で連結**。U-Net 一発学習はしない |
| フォーム C（学習） | セクション注釈データが必要。まずテンプレ A/B で可 |
| U-Net の役割 | 8 小節の**局所演奏化**。曲構成の賢さは U-Net 外 |
| H2 案1 | onset(1) と sustain(2) を**別減点**する損失分離 → **今後やる**（合意済み） |
| BPM 幅 | 現状 60–150 は狭い → **拡張は今後**（再合成＋再学習） |

---

## できていないこと（詳細）

### 1. 学習・検証（最優先ブロッカー）

| 項目 | 状態 | 備考 |
|------|------|------|
| Colab 15k force 再生成 | ユーザー側で実施中／完了待ち | 旧 6k 混在禁止 |
| **Stage1 再学習（in=12）** | 完了待ち | 旧 `unet_last.pt`（in=11）は使えない |
| ckpt 配置・スモーク | 未 | BPM 60/100/140 で密度差を確認 |
| manifest → prttype 同期 | 未 | 生成後 `prttype/data/raw/synthetic/` へ |
| BPM 条件の効果検証 | 未 | 理論上は効くが未実測 |

### 2. 音質・損失（Phase 3: H2/H5/H8）

| ID | 課題 | 状態 |
|----|------|------|
| **H2** | onset(1) と sustain(2) が MSE 一括で混同。打ち直し過多／伸ばしすぎ | **未解決**。案1（損失分離）を次の再学習で入れる予定 |
| **H5** | 和声・偽 onset を見ない損失 | 未対応 |
| **H8** | キー／モードを条件にしていない | 未対応（BPM 条件のみ） |
| recall / スケール外音 | 伸びしろあり | 未対応 |

### 3. データ・パイプライン

| 項目 | 状態 |
|------|------|
| **patches:2 副作用** | 8 小節でも `end_time=128` で 2 パッチになり得る。次回データ用に修正予定 |
| **BPM 幅 60–150** | 遅バラード（〜50）・高速（160+）が分布外。拡張は再合成＋再学習 |
| Stage2 TECHS | 通常不使用（リズム崩れ）。意図的に回避 |
| `checkpoints/lead/meta.json` | 実態と不一致の可能性（古い記述） |

### 4. プロダクト機能

| 項目 | 状態 |
|------|------|
| **長尺（〜3 分）生成** | テンプレ A/B + `generate_form` で連結済み。品質は学習依存 |
| **ベース / ドラム** | 合成・ペア・推論・form 統合済み。ドラムは **12型カタログ + beat_type 条件（in=24）**。`FormSection.beat_type` は構成レイヤで選択（将来 prior 学習で差し替え可）。**Colab 再学習待ち**（旧 in=12 drum ckpt 非互換） |
| **フォーム学習 C** | 未。ただし v2 で区間進行ルール + `form_manifest` / `form_gate_server` による評価蓄積を開始。candidates が溜まったら prior 学習に差し替え |
| **H9 ギターソロ** | 未着手（定義のみ） |
| arpeggio / lead の音価多様化 | strum のみ多様化済み |
| CVAE + 多様データ再学習 | CVAE コードあり。多様化データでの再学習は未 |

### 5. ドキュメント・整合

| 項目 | 状態 |
|------|------|
| `issues-and-resolution-order.md` | 6k・MuseCoco 本線等、一部古い記述あり |
| `implementation-roadmap.md` | 製品前提（進行非指定・WRIME・manifest）との突合未 |

---

## Decisions（変えない前提）

1. ユーザーは進行を指定しない。内部で prior 等から選ぶ
2. 骨格は **downbeat_chord**（疎）。N・配置はモデルが創造
3. MuseCoco 本線外。感情は WRIME → V/A（固定座標）。emotion_clf / calm·tension 4ラベルは廃止
4. prior 条件は **WRIME 8 + valence/arousal**（in=10）。旧 emotion_target one-hot ckpt は無効
5. Stage2 通常不使用。拡散モデル不使用
6. 15k は **force 入れ替え**（追記で旧 8 分ノリ混在させない）
7. 長尺・曲構成は U-Net 外。3 分を U-Net 一発学習しない
8. H2 は **損失分離（案1）** が第一候補
9. BPM 拡張は定数だけでなく正規化・感情帯・N 帯・**再合成**セット
10. **学習効率**: 基本 15k + in=12 を **v1 土台 ckpt** として一度フル学習。以降の改善は **差分データ + fine-tune**（フル再学習を毎回しない）
11. U-Net 条件は **正規化 BPM 1ch**（tonal11+1→11）。旧 in=11 U-Net ckpt 無効
12. **ドラム型**: 12 型カタログ + one-hot 条件（in=24）。選定は構成レイヤ `FormSection.beat_type`（`ROLE_BEAT_CANDIDATES`）。将来 prior 学習で差し替え可。旧 energy 密度 drum / in=12 ckpt 非互換

---

## 学習効率化: v1 土台 + 差分 fine-tune

基本データ（15k・BPM 条件・刻み多様化・in=12）が定まってきたため、**毎回ゼロから Stage1 するのではなく、土台 ckpt から改善分だけ学習する**方針。

### 前提

| 項目 | 内容 |
|------|------|
| v1 土台 | 15k force 再生成 + Stage1 フル（例 20ep）→ `stage1_base.pt` として固定保存 |
| 土台の条件 | **in=12 + BPM cond + 現行 makeData 分布** と一致していること |
| 旧 in=11 ckpt | 土台にならない（別アーキテクチャ） |

### 改善サイクル（v2 以降）

```text
v1_base:  15k + in=12 → stage1_base.pt（一度だけフル学習）
v2_delta: 追加分（例: BPM 50–180 帯 / H2 損失変更後の再ペア）→ fine-tune 3–5ep
v3_delta: さらなる差分 → 前版 ckpt から fine-tune
```

| やり方 | 目安 |
|--------|------|
| 15k フル再学習 20ep | 毎回フル時間 |
| 土台 ckpt + 追加分 3–5ep | だいたい **2〜5 倍短い** ことが多い |

### fine-tune 時のルール

1. **新 pairs だけにしない** — 旧 pairs を **20〜30% 混ぜる**（分布忘却防止）
2. **lr を下げる** — 例: 1e-3 → **5e-4 〜 1e-4**
3. **epoch は短く** — 3〜5ep から。過学習・忘却を見て調整
4. **`--resume`** — `train.py` の checkpoint 再開で土台から続ける
5. **データ版を明示** — `synthetic_v1` / `v2_bpm_ext` 等。黙って混在させない

### 改善タイプ別の扱い

| 改善 | 土台 ckpt 使える？ | データ |
|------|-------------------|--------|
| H2 損失分離（案1） | ✅ 使える | 同じ pairs で loss だけ変えて fine-tune |
| BPM 幅拡張 | ✅ 使える | 新 BPM 帯の追加分 pairs + 旧 20% |
| makeData 分布大変更 | ⚠️ 要検討 | 混ぜ方 or 部分フル再学習 |
| in_ch / 骨格方式変更 | ❌ 不可 | v1 からやり直し |

### 合成データの追記

- MIDI 追記: `makeData/generate.py --append`（**版が同じ**ときのみ）
- 分布が変わる版（v2）: **force または別ディレクトリ**で混在回避
- pairs: `prepare_all.py --force-regenerate` 時は旧 pairs 削除済み。差分のみなら追記生成 + 新 pairs のみパッチ化も可

### ckpt 命名（推奨）

| ファイル | 意味 |
|----------|------|
| `checkpoints/backing/stage1_base.pt` | v1 土台（15k + in=12 フル） |
| `checkpoints/backing/stage1_v2_h2loss.pt` | H2 損失分離後 fine-tune |
| `checkpoints/backing/stage1_v2_bpm180.pt` | BPM 拡張後 fine-tune |

---

## Next actions（優先順）

1. **Stage1 完了** → **`stage1_base.pt` として保存** → BPM スモーク
2. manifest を prttype に同期 → `generate_from_prompt` 確認
3. **H2 onset 重み強化 fine-tune**（実装済み）→ 下記「onset 重み強化 fine-tune 手順」参照
4. **BPM 幅拡張**: 追加分合成 → 旧 20% 混ぜて fine-tune
5. **長尺**: テンプレ連結オーケストレータ（backing 土台 ckpt はそのまま利用可）
6. 次回データ再生成時: **patches:2** の `end_time` 本修正（暫定回避として `first_patch_only=True` を導入済み）
7. （余裕）H9 ソロ / H8 モード条件 / lead meta 更新 / docs 突合

---

## Key files

| パス | 役割 |
|------|------|
| `prttype/generate_from_prompt.py` | テキスト → MIDI 入口 |
| `prttype/emotion_va.py` | WRIME → V/A |
| `prttype/sample_structure_params.py` | 文 → prior / カタログ |
| `prttype/structure_prior.py` | prior ネット（in=10） |
| `prttype/makeData/rhythm.py` | 合成 BPM↔N |
| `prttype/makeData/patterns.py` | N×placement×articulation |
| `prttype/model.py` / `train.py` / `inference.py` | U-Net in=12 |
| `prttype/prepare_dataset.py` | `*_cond.npy` 付きペア |
| `colab_train/colab_train.ipynb` | 15k + Stage1 |
| `docs/issues-and-resolution-order.md` | 課題 H1–H9 正本 |

---

## onset 重み強化 fine-tune 手順（H2 相当）

```bash
# prttype/ で実行
python train.py \
  --pairs-dir data/pairs/synthetic \
  --checkpoint-dir checkpoints/backing \
  --resume checkpoints/backing/stage1_base.pt \
  --epochs 5 \
  --lr 2e-5 \
  --pos-weight 10.0 \
  --onset-weight 2.0 \
  --midbar-onset-bonus 3.0
# 出力: checkpoints/backing/unet_last.pt → 手動で stage1_v2_h2loss.pt にリネーム
```

**効果測定（fine-tune 前後で比較）**:
```bash
# ベースライン（ローカル: prttype/ で実行）
python scripts/analyze_bar_rhythm.py \
  --pairs-dir data/pairs/synthetic --n-samples 200 --first-patch-only

# 生成後
generate_from_prompt.py "しんみりしたバッキング" --no-lead
python scripts/analyze_bar_rhythm.py --midi midi/しんみりしたバッキング_*.mid
```

**注意**: `analyze_bar_rhythm.py --pairs-dir` / `--npy-dir` は **target npy を読むだけ**（推論しない）
ため、fine-tune 前後で同じ値になる。モデルの効果を見るには `--midi`（実際に生成した MIDI）を使うか、
Colab では下記の Stage1.5 セルのように **実際にモデルへ推論させた出力**を評価すること。

**Colab でも実行可能**: `colab_train.ipynb` の「Stage1.5」セル群（Stage1 push の後）に
ベースライン計測 → fine-tune → 実推論での効果測定 → push まで一式が入っている
（`checkpoints/stage1_h2/unet_last.pt`）。土台は Stage2 ではなく **Stage1**（Stage2 の
curated TECHS fine-tune はリズムを崩す教訓があるため）。

**Colab: backing/lead を個別 or 同時に実行**: ノートブック冒頭の「学習設定」セルに
`TRAIN_BACKING` / `TRAIN_LEAD`（と backing 内の `RUN_STAGE2` / `RUN_STAGE1_5`）トグルがあり、
これで on/off した上で「すべてのセルを実行」するだけで目的の学習だけが走る（データ生成・
チェックポイントとも backing/lead は完全に独立しているため片方だけでも安全に実行可能）。

**効果が薄い場合**: `--midbar-onset-bonus 6.0` に上げて追加 2ep。

**精度低下の検出**: `onset_precision_vs_skeleton`（Colabでは `precision_vs_chord` 表示）が
fine-tune 前から 10 pt 以上落ちたら `onset_weight` を 1.5 に下げる。

**bar_end_pattern_entropy の注意**: `analyze_bar_rhythm.py` は全パッチ・全小節をプールしてから
1 回だけエントロピーを計算する（パッチ単位で計算して平均すると n=8/パッチしかなく飽和するため、
2026-07-20 に修正済み）。0.0（常に同じ終わり方）〜4.0bit（16通りに均一分布）のスケール。

---

## Do not redo / pitfalls

- 骨格に N（onset 列）を載せる案（ユーザー却下）
- manifest N で生成制御（効かない）
- 学習中に BPM_RANGE / 損失 / in_ch / makeData 分布を変える
- 旧 in=11 ckpt を新パイプラインに使う
- 6k に追記（旧 8 分ノリ混在）
- Stage2 を backing 本線に戻す
- 音価多様化（案1）を未実施扱いでやり直す
- データ 4 分のみのまま CVAE に過度な期待
- `weighted_mse_loss` を BCE 化する（model 出力は raw 非有界値のため非互換）
- 小節末を一律で埋める後処理の追加（ユーザー却下）

---

## 用語メモ

- **onset (1)**: 発音の瞬間（弾き始め）
- **sustain (2)**: 伸ばしている途中
- **H2 案（onset 重み強化）**: MSE のまま onset セル・中拍 onset セルの重みを増やす方式。BCE 化は採用しない（model 出力が raw 非有界値のため）
- **first_patch_only**: `midi_to_patches` の patches:2 バグ（8 小節ちょうどで重複パッチが出る）の暫定回避。`prepare_all.py` で自動適用済み

---

## User quotes（代表）

> ユーザーは進行を指定するわけではない…学習データから自動的にmidを生成する  
> 創造的に埋めるしその何回どこで弾くかも考えたい  
> やめるものはやめる方針でその方針で生成してください  
> それいいかもね今後やることとしてhandoffにまとめておいて（H2 案1）  
> あとやることとして現状だとBPMの幅が狭くないですか？  
> 基本データは定まった → 学習済み ckpt を土台に追加データで改善（fine-tune）

---

## 統合元（アーカイブ）

以下 8 ファイルの内容を本書に統合済み（2026-07-20）:

- `2026-07-20_0621_handoff.md`
- `2026-07-20_0730_handoff.md`
- `2026-07-20_0731_handoff.md`
- `2026-07-20_0742_handoff.md`
- `2026-07-20_0819_handoff.md`
- `2026-07-20_0900_handoff.md`
- `2026-07-20_1523_handoff.md`
- `2026-07-20_1526_handoff.md`
