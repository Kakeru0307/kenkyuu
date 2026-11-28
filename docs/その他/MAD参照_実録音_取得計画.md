# MAD 主参照実録音 — 楽に取る取得計画

更新: 2026-11-25  
正本の母数・入れないもの: `docs/その他/MAD参照_実録音セット.md`  
指標の主副: `docs/その他/MAD評価方針.md`  
計算: `testMad/scripts/compute_mad.py`（`.wav` のみ）

実行開始: 2026-11-25（ローカル `testMad/ref_real/`）。  
失敗時の代替は「エラーではじかれたとき」を正とする。母数の入れ替え（pop・歌入り等）では埋めない。

### 実行ログ（作業メモ）

| 段階 | 状態 |
|---|---|
| 0 メタ | FMA `fma_metadata.zip`・MTG `raw_30s_cleantags.tsv` 取得済 |
| 1 ID | MTG 585（voice除外）・FMA 1,432 の TSV を `manifests/` に作成済 |
| 2 MTG | **完了** Jamendo直で 585/585・fail 0（`ref_real/wav/mtg_*.wav` 約2.9GB） |
| 3 FMA | `fma_large.zip` DL 中（容量的に個別より zip→パス抽出）。完了後 `extract_fma_ref_from_zip.py` |
| 4 | 未（WAV 揃い次第 Drive / Colab） |

## 目標

約 **1,500曲** のギター前景・歌なし実録音 WAV を、フル配布（FMA 93GB 全展開・MTG raw_30s 全tar 約508GB）を避けて手元／Drive に置く。

| 入れる | 概数 |
|---|---|
| FMA ロック系 ∩ インスト、30秒〜6.5分 | **1,432**（メタ集計済み） |
| MTG-Jamendo `instrumentalrock`（voice タグ除外） | **約585**（598−13） |

副参照 `testMad/ref/{midi,wav}` の101本とは**混ぜない**。置き場案: `testMad/ref_real/{mp3,wav,manifests}`（git 外。Drive へ）。

## 方針（楽さ優先）

メタで ID を切ってから、必要なファイルだけ HTTP / パス指定抽出する。

```text
メタ取得 → IDリスト → MTG個別DL（先） → MP3→WAV → 数十本目視
                ↓
           FMA必要分（後） → MP3→WAV → Drive → Colab MAD
```

**先に MTG だけ**で主 MAD の初回計算ができる。FMA は容量・取得経路が重いので後段。

### MP3 と WAV

配布は MP3（または 30秒クリップ）だが、`compute_mad.py` は **`.wav` しか数えない**。  
MP3→WAV はデコードして PCM を包むだけで、評価入力としては通例どおり問題ない。情報は増えない。長さ揃え（MTG は 30秒、FMA も長すぎれば切る）と欠落スキップが本筋の注意点。

## 段階0: メタだけ

1. FMA: `https://os.unil.cloud.switch.ch/fma/fma_metadata.zip`（約342MB）→ `tracks.csv` / `genres.csv`
2. MTG: [MTG/mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset) の `raw_30s_cleantags.tsv`（手元にあればそれを使う）

## 段階1: IDリスト

- FMA: ロック系 ∩ インスト、長さ 30秒〜6.5分 → 約1,432 ID。パス規則 `fma_large/{tid//1000:03d}/{tid:06d}.mp3`
- MTG: タグ `instrumentalrock`、かつ voice 系タグなし
- 一覧は `testMad/ref_real/manifests/` に TSV。git に入れるのは ID・タグだけ（音は入れない）

## 段階2: MTG を先に取る

全tar ではなく CDN の個別 MP3:

`https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/{folder}/{track_id}.mp3`

（folder は track_id 末尾2桁など。TSV の PATH 列を正とする）

- 並列は控えめ（切れ対策）
- 取れない ID は一覧に残してスキップ
- 成功分を ffmpeg で 30秒 WAV
- 無作為 **30本** だけ聴く。ギター前景でないものが多ければタグ条件を締める（全件目視はしない）

これで約500〜600本の主参照ができる。生成100本との MAD 初回はここで打ってよい。

## 段階3: FMA を必要分だけ

93GB を全部展開しない。次のどちらか。

**A（推奨・帯域が楽）**: Free Music Archive / mdeff/fma 系の track 単位取得。API キー等が要る場合あり。約1,432本分のみ。

**B（キー無し・確実寄り）**: `fma_large.zip` を Drive / 外付けに1回落とし、7-Zip で **IDリストのパスだけ抽出**（全展開しない）。壊れた MP3 は飛ばす。WAV 化は段階2と同じ。

## 段階4: まとめと MAD

- `ref_real/wav` に FMA + MTG を同居（ファイル名に `fma_` / `mtg_` 接頭辞）
- Colab は Drive のこのディレクトリを参照。参照差し替え後は `CLEAR_REF_ONLY` で埋め込みだけ作り直す
- 生成側は当面 `testMad/gen/wav` 100本。主参照1,500に対して生成が細いと感じたら数百本化は別チケット

## エラーではじかれたとき

原則: **同じ母数のまま経路を一段落とす**。欠落 ID は `manifests/failed_*.tsv` に残し、成功分だけで先に進む。足りない本数を `instrumentalpop`・歌入り rock・MusicCaps で埋めない。

共通ルール:

1. HTTP 4xx/5xx・タイムアウト・切断 → その ID を failed に書き、次へ（ループ無限リトライ禁止）
2. 同一ホストで連続失敗が目立つ → 並列を 1 にし、待ちを長くしてから再試行（1 周だけ）
3. 再試行後もだめ → 下表の代替経路へ切替
4. 破損 MP3・ffmpeg 失敗 → スキップ（再エンコードで無理に救わない）
5. 最終本数が計画より減っても、フィルタ条件は緩めない。減少数はメモして MAD 注記に残す

### 段階0（メタ）

| はじき | 次にやること |
|---|---|
| FMA `fma_metadata.zip` が落ちない / 壊れる | 公式ミラー `https://os.unil.cloud.switch.ch/fma/` を再試行。だめなら mdeff/fma の案内先メタを使う。メタ無しで音だけ取らない |
| MTG `raw_30s_cleantags.tsv` が無い | [MTG/mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset) から TSV のみ取得。手元コピーがあればそれを正とする |
| ジャンル ID の突合が合わない | `genres.csv` の親子を再確認。推測でロック系を広げない |

### 段階2（MTG 個別 CDN）

優先順（上がだめなら次）:

```text
個別 CDN（並列控えめ）
  → 失敗 ID だけ再試行（並列1・間隔長め）
  → 公式推奨の raw_30s tar を「必要 shard だけ」落とす
       （instrumentalrock の PATH が属する tar のみ。全 508GB は取らない）
  → tar 内の該当 PATH だけ抽出 → MP3→WAV
```

| はじき | 次にやること |
|---|---|
| 404 / ファイル無し | PATH 列の folder を再確認。それでも無ければ欠落としてスキップ |
| 429・接続リセット・極端な遅延 | 並列停止 → 待機 → 失敗一覧だけ再 DL。繰り返すなら tar 経路へ |
| CDN 全体が長時間不通 | その時点で成功分だけで WAV 化し MAD 初回。残りは tar 経路を後段 |
| tar の shard が特定できない / 取得が重い | MTG は成功分だけで主 MAD 初回を打つ。FMA 側で本数を補う（母数の種類は変えない） |

### 段階3（FMA）

優先順:

```text
A: track 単位 HTTP / API
  → キー・認証・大量 4xx で詰まる
  → B: fma_large.zip を1回取得 → 7-Zip で ID パスだけ抽出
  → zip 内に無い / 壊れている ID はスキップ
```

| はじき | 次にやること |
|---|---|
| A の API キー無し・鍵無効・エンドポイント変更 | **即 B**（zip＋パス抽出）。A に固執しない |
| A で大量 404 | パス規則 `fma_large/{tid//1000:03d}/{tid:06d}.mp3` を確認。直らなければ B |
| `fma_large.zip` の DL 切れ | レジューム可能なクライアントで再開。全展開はしない |
| 7-Zip のパス指定抽出が失敗 | パス区切り（`/`）と 6 桁ゼロ埋めを確認。ダメなら該当フォルダ単位だけ抽出（全展開は最後の手段で、できれば避ける） |
| zip 内 MP3 破損・デコード失敗 | スキップ。同一曲を別ソースで補完しない |

### WAV 化・Colab

| はじき | 次にやること |
|---|---|
| ffmpeg が一部だけ失敗 | 失敗 ID を failed に残し、成功 WAV だけで MAD |
| Drive 同期切れ・パス違い | ローカル `ref_real/wav` を正とし、再アップロード。埋め込みは `CLEAR_REF_ONLY` |
| 参照差し替え後に旧埋め込みが残る | 必ず `CLEAR_REF_ONLY`。混在した埋め込みで MAD を出さない |

### 本数が大きく欠けたとき

- MTG だけでも約500〜600 → **主 MAD 初回は打ってよい**
- FMA が半分以下しか取れない → 取れた分＋MTG で計算し、計画1,500との差を結果メモに書く
- それでもギター前景の標本目視で質が悪い → タグ条件を**締める**（緩めて本数稼ぎはしない）
- AudioSet / Music4All / チャート名曲だけで埋めない（`MAD参照_実録音セット.md` の「やらない」に従う）

## 実行可能性の根拠と未検証

### 根拠がある

| 項目 | 根拠 |
|---|---|
| 本数・フィルタ | `MAD参照_実録音セット.md` のメタ集計（2026-11-23） |
| MTG 個別 URL | CDN に `raw_30s/audio/{folder}/*.mp3` が公開されている（ディレクトリ一覧で確認可） |
| WAV 入力 | `compute_mad.py` の `.wav` のみカウント |
| FMA パス抽出 | 公式パス規則 + 7-Zip のパス指定抽出は一般に可能 |

### まだ保証できない（支障ゼロではない）

| 項目 | 内容 |
|---|---|
| MTG 連続 DL | 数百本の切れ・遅延・レート制限。公式 README は tar 推奨 |
| FMA API | 個別取得の現状安定性は未確認。確実寄りは zip＋パス抽出 |
| 欠落・破損 | 壊れた MP3・404 はスキップ前提 |
| ギター前景比率 | フィルタ後の質は数十本目視で締める（全件未聴） |
| 手元メタ | 取得作業時点で TSV / `tracks.csv` が無ければ段階0から |

## やらない

- `fma_large` / MTG raw_30s の全展開を前提にする
- `instrumentalpop`・歌入り rock・FMA-Pop・MusicCaps を母数に足す
- 副参照101本と実録音を同じフォルダに混ぜる
- 全曲の目視採用
- 主 MAD と副 MAD を注記なしに同じ表へ入れる

## 関連

- `docs/その他/MAD参照_実録音セット.md`
- `docs/その他/MAD評価方針.md`
- `docs/その他/MAD参照_ギターバンド基準と候補曲.md`
- `testMad/README.md`
