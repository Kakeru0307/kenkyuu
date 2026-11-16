# condition_encoder 教師 JSONL

巨大な生データは git に入れない。Drive / ローカルで生成する。

| ファイル | 生成スクリプト | 内容 |
|---|---|---|
| `wrime_va.jsonl` | `scripts/build_wrime_va_pairs.py` | 一般日本語 → VA |
| `theta_va.jsonl` | `scripts/build_theta_va_pairs.py` | 音楽キャプション JA → 人手 VA |
| `warriner_mood_va.jsonl` | `scripts/build_warriner_mood_va_pairs.py` | mood 語 → Warriner VA |
| `music_axes.jsonl` | `scripts/build_music_condition_pairs.py` | Jamendo 実タグ → 軸スコア |

自作の設計 VA 文は使わない。
