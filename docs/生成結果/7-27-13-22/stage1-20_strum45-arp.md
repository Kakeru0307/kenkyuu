# Stage1 epochs=20（strum≈45% / arp≈40% 学習）の生成スナップショット

- date: 2026-07-27
- checkpoint: `prttype/checkpoints/backing/unet_last.pt`（epochs=20, lr=0.001, in_channels=12 → Stage1）
- command: `python generate_backing.py --text "元気なストローク" --seed 3 --output midi/backing_log_seed3.mid`

## データの内容

- **この ckpt の学習データ混在（ユーザー報告・当時の合成方針）**: ストローク系 ≈45%、アルペジオ系 ≈40%（残りは旧型 chord_strum / 単体 arp / scale 等の薄い尾）
- **当時の `PATTERN_WEIGHTS` 想定**: `progression_strum` 0.45 / `progression_arpeggio` 0.45 前後（同比重に近い）
- **注意（コード現状とのズレ）**: リポジトリの `makeData/constants.py` は既に `progression_strum: 1.0`（他 0）に変更済み。**本 ckpt はその再生成・再学習前**の成果物
- Stage / タスク: Stage1、`downbeat_chord`（小節頭コード骨格 → フル）
- 合成まわり（学習時の一般設定）: bars=8 固定寄り、strum は articulation・attacks_per_bar 多様化あり

### 生成規律（コード現状・推論）

- CLI: 進行・キー・BPM をユーザー指定しない。`--text` → WRIME → structure prior（無ければカタログ乱択）。`--seed` で prior 再現
- 骨格: `downbeat_chord`（小節頭コードトーン）
- モデル入力: 骨格 + BPM 条件チャンネル（in_channels=12）
- ※本ログ時点は `chord_align` / loop-raise 等の推論ノブあり。現コードでは削除しモデル出力のみ
- 最終形ルール上、本線は「文→prior→（未実装 form）→backing」。本ログは **バッキング単体マイルストーン** の記録

## 生成結果

- midi: `docs/生成結果/7-27-13-22/stage1-20_strum45-arp.mid`
- 構造サンプル: progression=deceptive, key=Eb, bpm=150, mode=major, bars=8
- ノート数 14 / ユニーク onset 14。小節あたりストローク 1〜3 回（8分刻みではない）。ほぼ単音（同時発音なし）
- MusPy: M1=1.0000 M2=0.2000 M3=1.0000 M4=1.0000（M3b polyphony_rate=0）
- 主観: 未回答

## 得られた結論

- 調性指標（M1/M4）は問題ないが、**和音ストローク・小節内の密な刻みは出ていない**（M3≈1、strokes/bar 1–3）
- strum/arp ほぼ半々の学習混在と整合的で、単音・疎な発音に寄りやすい
- 次の比較基準として、strum 100% 再生成＋再学習後の同コマンド（`--text` / `--seed 3`）ログを取るのがよい
