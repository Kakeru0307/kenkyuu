# Auto handoff archives

`prttype/` / `colab_train/` 編集前の auto-save 先。

- 12時間未満: 枠内の最古 `*_handoff.md` を上書き
- 12時間以上: 現行 auto ファイルを `docs/version/` へ移し、ここに新規作成
