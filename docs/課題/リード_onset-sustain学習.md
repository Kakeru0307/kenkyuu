# リード onset/sustain 学習案（1→2 本線）

- 更新: 2026-11-23
- 目的: 教師の **1→0.5 / 2→1.0** を推論でも分け、デコード本線の **1→2** を復活させる
- 起動: この md を読んで Step 1 から実装・Colab 学習へ進む

## なぜ今の ckpt ではダメか

データ比の問題ではない。現状 lead ckpt は:

| 事実 | 意味 |
|---|---|
| 出力 max ≈ **0.65** | `sustain_th≈0.75` に届かない → **class2 がほぼ出ない** |
| 教師は 0.5 / 1.0 | モデルが中間値に潰れている |
| デコード本線は 1→2 | 2 が無いので cap / protect 等の推論ハックに依存していた |

聴感だけなら短音っぽく使えるが、**onset/sustain 学習としては不合格**。th だけで本線化したことにしない。

## データ事実（lead pairs zip・800 パッチ標本）

`colab_train/lead_data/lead_pairs_st5000_gaps.zip` の `target/*_tonal.npy`:

| 指標 | 値 |
|---|---|
| 正例セル onset : sustain | **41% : 59%**（sus/onset ≈ **1.4**） |
| 音符長 tick | median **1** / mean **2.4**（約 63% が長さ1＝sustain セル無し） |
| 無音/正例 | ≈ 1400 倍（極端に疎） |

→ pairs 作り直しは原則不要。不足は **0.5 と 1.0 の出し分け**。

## 採用ハイパラ（初手）

| 引数 | 値 | 理由 |
|---|---|---|
| `--pos-weight` | **20** | 現状維持。全逆頻度は不要 |
| `--onset-weight` | **1.0** | lead に backing の 4 を載せない |
| `--sustain-weight` | **2.0** | 頻度均衡 ≈1.4 の少し上。実効 onset20 / sus40 |

### 振る幅（学習後ヒストで）

| 観察 | 次 |
|---|---|
| 0.9〜1.0 が薄い | sustain **2.5 → 3.0** |
| 1.0 塗り・onset 消滅 | sustain **1.5** |
| **4.0 以上** | この比率では原則やらない → 3-class CE 検討 |

強すぎるとどちらか一方に潰れる（onset 過多で 2 不出 / sustain 塗りで棒）。

## Step（実装順）

### 1. 損失（`colab_train/train.py`）

`weighted_mse_loss` に sustain 加重を追加:

```text
sustain_mask = (targets == 1.0)   # 正規化後の値2
weights[sustain_mask] *= sustain_weight
```

- CLI: `--sustain-weight`（default 1.0 = 現行と同一）
- notebook: `CVAE_SUSTAIN_WEIGHT_LEAD = 2.0` を lead コマンドだけに渡す
- lead コマンドに明示: `--onset-weight 1.0 --sustain-weight 2.0`
- FiLM / power / mask-ratio 0.5 / 40 epoch / scratch は現状どおり

### 2. Colab 学習

- `GEN_LEAD=True`、scratch、AMP なし（現行）
- pairs: 既存 lead（GAPS boost 済み可）。再生成は「教師に 2 が無い」と分かったときだけ

### 3. 合格線

| 指標 | 目安 |
|---|---|
| 出力ヒスト | 0.5 付近と **0.9〜1.0** に山（または sustain 域が明確） |
| `sustain_th≈0.75` で class2 | 0% ではない（教師オーダー） |
| 生成 | 保護なしでも 100tick 級の偽長音が減る／消える |
| 聴感 | 伸ばしっぱなし事故↓、かつ全部短い刻みだけにも戻らない |

### 4. 推論合わせ（学習のあと）

- th を学習分布に合わせる（訓練 denorm の **0.25 / 0.75** を起点）
- **2 由来の cap 保護は廃止**
- デコードは既存 1→2 が本線（固定分割・4tick 等は戻さない）

### 5. それでもダメなら

出力を 3 クラス Softmax + CE（補助 MSE）。初手は sustain 加重 MSE。

## やらない

- いま sustain「種類」を増やす
- th / cap / protect だけで 1→2 本線扱い
- lead に `--onset-weight 4`（backing 用）をそのまま載せる
- prior だけ・manifest 1 曲選択などの本線外ショートカット

## 関連

- 本線: `generate_form.py`（文→VA→prior→song_form→4パート→リード奏法 PB）
- 損失現状: `colab_train/train.py` `weighted_mse_loss`（onset のみ追加倍率あり、sustain 未実装）
- notebook: `colab_train/colab_train.ipynb` lead は `--pos-weight 20` のみ（onset/sustain 未指定）
- 量子化: `prttype/util/midi_to_patch.py` `normalize` 0/0.5/1.0、訓練 denorm 0.25/0.75
