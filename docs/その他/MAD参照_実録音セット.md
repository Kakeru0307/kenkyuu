# MAD 主参照 — 実録音セット（結論）

更新: 2026-11-27  
正本の役割: 実録音参照を何で何本にするか。指標の主副は `docs/その他/MAD評価方針.md`。  
配置: **`testMad/ref/wav`**（`ref_real` は廃止・統合済み）。マニフェストは `testMad/ref/manifests/`。

## 結論

主参照は **ギター前景・歌なしの実録音** のまま。規模は **約1,500曲** でよい。

5,000曲は論文（FMA-Pop 4,230、MusicCaps 5,521、社内セット 7,846）の汎用ポップ規模であり、ギターインストの公開実録音では届かない。薄い曲を足して5,000にしない。

| 入れる | 入れない |
|---|---|
| FMA のロック系 ∩ インストジャンル（30秒〜6.5分で **1,432**） | FMA-Pop / MusicCaps を主参照にする |
| MTG-Jamendo の `instrumentalrock`（**598**。歌タグは13） | `instrumentalpop` 1,477 を母数に足す |
| 既存101曲のスタジオ版は見本（数十曲）。母数にはしない | 歌入りロック（MTG `rock` 6,865）で本数を稼ぐ |

FMA と MTG は別カタログなので、単純和でおおよそ **2,000曲**、長さと歌タグを落とすと **1,500前後**。重複の実数は未突合。

## なぜこの本数でよいか

MAUVE は両側がだいたい **1,000本超** だと距離として読める。論文の5,000本はその先の頭打ち。1,500は101本より安定し、4,000本よりばらつきが大きい、という位置。

論文の MAD ≈ 2.45 は MusicCaps **5,521本同士**。こちらの主 MAD とその数字を、本数が違うまま並べない。北極星の目安（MAD ≈ 2.3）は `MAD評価方針.md` のまま。

生成側が100本のままだと、参照を1,500にしても生成分布が細い。参照1,500に対して生成は **数百本** あるとバランスが取れる。副参照（SF揃え・101本）の診断用100本とは別枠。

## 数えた実数（2026-11-23、メタデータのみ）

### FMA（`tracks.csv` / `genres.csv`、106,574曲）

| フィルタ | 曲数 |
|---|---|
| ロック系 ∩ インストジャンル | **1,789** |
| そのうち 30秒〜6.5分 | **1,432** |
| インストからアンビエント・サントラ・ニューエイジを除く | 2,804 |
| そのうちロック系 | 513 |
| ポストロック全体（歌入り含む） | 1,952 |

1,789曲はすべて `fma_large` 側。`fma_medium`（22GB）には入っていない。音源は `fma_large.zip`（30秒 MP3、93GB）から抜く。公式配布は `https://os.unil.cloud.switch.ch/fma/`。展開は 7-Zip。壊れた MP3 は飛ばす。

### MTG-Jamendo（`raw_30s_cleantags.tsv`、CC、約5.5万曲）

| タグ | 曲数 | voice タグ | ギターかつドラムかつ voice なし |
|---|---|---|---|
| rock | 6,865 | （歌入りが本体） | — |
| instrumentalpop | 1,477 | **4** | **79** |
| instrumentalrock | **598** | **13** | **108** |
| postrock | 377 | 36 | 70 |

`instrumentalpop` は歌入りポップではない。外す理由は、ギターかつドラムと分かる曲が79しかなく、残りがピアノやシンセのインストになりやすいこと。入れると MAD は「ギターバンドに近いか」ではなく「そのインストポップに近いか」になる。

歌入りロックは本数は足りるが、MERT では歌声が目立つ。編成が良くなっても距離が縮みにくい。主参照にしない。

フル音源は約508GB。取るのは `instrumentalrock` だけ。

## 取得状況（2026-11-27）

| ソース | WAV | 場所・命名 |
|---|---|---|
| FMA ロック∩インスト | **1430**（マニフェスト1432のうち壊れた MP3 2本スキップ: 91206 / 145056） | `testMad/ref/wav/fma_XXXXXX.wav` |
| MTG `instrumentalrock` | **585**（既存） | `testMad/ref/wav/mtg_XXXXXXX.wav` |
| **合計** | **2015** | `testMad/ref/wav/` |

形式: PCM s16le / 44.1 kHz / stereo。FMA は `Downloads/fma_large.zip` から変換。マニフェスト: `fma_rock_instrumental.tsv` / `mtg_instrumentalrock.tsv` / `fma_convert_failed.tsv`。

`testMad/ref/midi`（旧 MIDI→SF 101本）は残置。副参照用 WAV は別置きしない（実録音を `ref/wav` に一本化）。

### 歌検出一次仕分け（2026-11-27・実行中）

ユーザー許可: ルール7の例外として **Demucs 歌ステム比で一次仕分け**。最終採用は目視。

| 項目 | 内容 |
|---|---|
| スクリプト | `testMad/scripts/triage_ref_vocals.py` |
| ログ | `testMad/cache/vocal_triage.log` |
| 結果 TSV | `testMad/ref/manifests/vocal_triage.tsv` |
| 手法 | `htdemucs`、中央15秒。`rms` / `p90` / `frac` |
| REJECT | rms≥0.12 or p90≥0.40 or frac≥0.25 → `ref/wav_reject_vocal/`（`--apply` 後） |
| KEEP | rms&lt;0.05 and p90&lt;0.15 and frac&lt;0.10 → `ref/wav` に残す |
| UNCERTAIN | 上記以外 → `ref/wav_uncertain_vocal/`（`--apply` 後・目視） |

全本スコア完了後に `--apply` で移動。uncertain は耳で最終確認。

ライセンスは曲ごとの CC。ローカル評価に使い、再配布しない。音源は git 外。

## やらない

- 主参照を FMA-Pop、MusicCaps、歌入りロック、インストポップで埋める
- チャート名曲40曲だけで主参照にする（見本としては残せる）
- AudioSet（10秒 YouTube。ギター注釈は多いが、インスト×ギター×ドラムの積は未公開で欠落も多い）
- Music4All（109,269曲の30秒はあるが、歌詞がある曲だけ。申請制）
- 主 MAD と副 MAD（MIDI→同一 SF、101本）を注記なしに同じ表へ入れる

## 関連

- 方針: `docs/その他/MAD評価方針.md`
- 副参照の曲一覧: `docs/その他/MAD参照_ギターバンド基準と候補曲.md`
- 計算: `testMad/scripts/compute_mad.py`
