---
name: generate-and-log
description: >-
  Generate backing MIDI, run MusPy eval, and save md+midi together under
  docs/生成結果/<M>-<D>-<HH>-<mm>/ (data / result / conclusion).
  Use when the user asks to generate and record, log an experiment, write
  生成結果, or after a new backing checkpoint / data change should be logged.
---

# Generate and log (backing experiment)

Follow `.cursor/rules/generation-results.mdc` for entry shape.  
Follow `.cursor/rules/post-generation-eval.mdc` for M1–M4 and Chu Q1–O1 (do not invent scores).

## Procedure (fixed order)

1. **Fix data conditions** (unknown → `未確認`):
   - checkpoint path + `epochs` / `lr` / stage from the `.pt` (not stale `meta.json` alone)
   - `PATTERN_WEIGHTS` (or other data mix that mattered)
   - seed, `--text` / prior params if used
2. **Generate** with `prttype/generate_backing.py` (venv python). Prefer a fixed `--seed` when comparing runs.
3. **Create run folder** `docs/生成結果/<M>-<D>-<HH>-<mm>/` (local time; use `-` not `:`). Collision → append `-<SS>`.
4. **Pick one** output MIDI → copy into that folder (same stem as the md when practical).
5. **Objective eval** on the copied MIDI:
   ```text
   py -3 prttype/evaluate_midi.py <run-folder>/<file>.mid --key <KEY> --mode <MODE>
   ```
   Record M1–M4. Skip M5 unless a reference corpus is requested.
6. **Ask the user** for Q1–Q4 + O1 (1–5). If they defer, write the entry without subjective scores and note `主観: 未回答`.
7. **Write** `<run-folder>/<短いラベル>.md` using the template below (md and midi must sit in the same folder).
8. Reply with a short summary and the run folder path.

## Entry template (use verbatim structure)

```markdown
# <短いタイトル>

- date: YYYY-MM-DD
- checkpoint: ...
- command: ...

## データの内容
- ...

## 生成結果
- midi: docs/生成結果/<M>-<D>-<HH>-<mm>/<短いラベル>.mid
- MusPy: M1=... M2=... M3=... M4=...
- （任意）主観 Q1–O1

## 得られた結論
- ...
```

## Pitfalls

- Do not put midi under a separate `midi/` subfolder; keep md+midi as siblings.
- Do not log more than one MIDI per entry.
- Code-only edits with no generation → do not create an entry unless the user asks.
- Past runs are not backfilled unless the user asks.
