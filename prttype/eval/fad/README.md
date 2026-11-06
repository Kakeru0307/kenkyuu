# FAD 評価用ディレクトリ

```text
.\venv\Scripts\python.exe prttype\scripts\compute_fad.py prttype\eval\fad\ref_wav prttype\eval\fad\gen_wav
```

| フォルダ | 中身 |
|---|---|
| `ref_wav/` | 参照 WAV（テスト用: EMOPIA 100本。本番はロック・メタル等） |
| `gen_wav/` | このプロジェクトの生成 WAV（`generate_form` 1本） |

- 埋め込み: `clap-laion-music`（`fadtk`）
- MIDI→WAV: `prttype/scripts/midi_to_wav.py`（同じ SoundFont）
- 本番スコアは各フォルダ 100本以上を想定。数値確認は後でよい
