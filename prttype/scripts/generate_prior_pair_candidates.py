"""条件 prior 用の短い対応データ候補を生成する。

機械生成は gate=pending の候補。学習には人手で accept したものだけを使う。
出力: data/prior_pairs/manifests/candidates.jsonl + midi/*.mid

--append で既存を消さず追記。ID は既存最大の次から。
--split で accept.jsonl / reject.jsonl に振り分け（candidates は残す）。

文候補は雰囲気グループから出し、構造は WRIME→V/A から決める。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from emotion_va import wrime_to_va
from makeData.constants import KEYS
from makeData.patterns import generate_progression_strum
from makeData.progressions import PROGRESSIONS, ProgressionSpec
from patch_to_midi import save_music

OUT_ROOT = SCRIPT_DIR / "data" / "prior_pairs"
MIDI_DIR = OUT_ROOT / "midi"
MANIFEST_DIR = OUT_ROOT / "manifests"
CANDIDATES_PATH = MANIFEST_DIR / "candidates.jsonl"

# 文の多様性用プロンプト群（構造決定には使わない。V/A が決める）
_PROMPT_GROUPS: dict[str, list[str]] = {
    "bright": [
        "明るいギターバッキング",
        "楽しい午後の伴奏",
        "元気が出るストローク",
        "晴れやかなコード進行",
        "弾むようなバッキング",
        "明るい朝のギター",
        "テンポよく進む伴奏",
        "明るいポップなバッキング",
    ],
    "sad": [
        "しんみりした夜のバッキング",
        "悲しい気持ちの伴奏",
        "泣きそうなギター",
        "物悲しいコード進行",
        "暗い部屋のバッキング",
        "寂しい夜の伴奏",
        "沈んだ気持ちのギター",
    ],
    "calm": [
        "落ち着いたギターバッキング",
        "穏やかな夕方の伴奏",
        "ゆるやかなストローク",
        "静かなコード進行",
        "ゆったりしたバッキング",
        "静かで穏やかな伴奏",
        "ゆっくり流れるギター",
        "落ち着いた夜のコード",
    ],
    "tense": [
        "緊迫したギターバッキング",
        "焦りのある伴奏",
        "張りつめたストローク",
        "不穏なコード進行",
        "緊張感のあるバッキング",
        "せわしないギター伴奏",
        "追い立てられるようなバッキング",
    ],
}


def _load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _next_id(rows: list[dict]) -> int:
    best = 0
    for row in rows:
        m = re.match(r"pp_(\d+)$", str(row.get("id", "")))
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def split_gates() -> None:
    rows = _load_rows(CANDIDATES_PATH)
    if not rows:
        raise SystemExit(f"no candidates: {CANDIDATES_PATH}")
    accept = [r for r in rows if (r.get("gate") or {}).get("status") == "accept"]
    reject = [r for r in rows if (r.get("gate") or {}).get("status") == "reject"]
    pending = [r for r in rows if (r.get("gate") or {}).get("status") == "pending"]
    _write_rows(MANIFEST_DIR / "accept.jsonl", accept)
    _write_rows(MANIFEST_DIR / "reject.jsonl", reject)
    _write_rows(MANIFEST_DIR / "pending.jsonl", pending)
    print(
        f"split: accept={len(accept)} reject={len(reject)} pending={len(pending)} "
        f"→ {MANIFEST_DIR}"
    )


def _va_to_struct_range(v: float, a: float) -> dict:
    """V/A から構造サンプル用のモード・BPM帯・energy 候補を決める。"""
    if a > 0.3:
        bpm = (110, 150)
        energy = ("mid", "high")
    elif a < -0.2:
        bpm = (60, 90)
        energy = ("low", "mid")
    else:
        bpm = (85, 120)
        energy = ("low", "mid", "high")

    if v < -0.2:
        modes = ("natural_minor",)
    elif v > 0.2:
        modes = ("major",)
    else:
        modes = ("major", "natural_minor")

    return {"modes": modes, "bpm": bpm, "energy": energy}


def _energy_for_bpm(bpm: float, choices: tuple[str, ...], rng: random.Random) -> str:
    if bpm < 90:
        preferred = "low"
    elif bpm < 115:
        preferred = "mid"
    else:
        preferred = "high"
    if preferred in choices:
        return preferred
    return rng.choice(list(choices))


def _sample_structure_for_va(v: float, a: float, rng: random.Random) -> dict:
    pref = _va_to_struct_range(v, a)
    modes = pref["modes"]
    pool = [p for p in PROGRESSIONS if p.mode in modes] or list(PROGRESSIONS)
    if a > 0.3 and v < -0.1:
        boosted = [p for p in pool if p.family in ("borrowed", "blues", "diatonic_minor")]
        if boosted:
            pool = boosted + pool
    spec: ProgressionSpec = rng.choice(pool)
    if spec.mode == "natural_minor":
        key_pool = [k for k in KEYS if k in ("A", "E", "D", "G", "C", "B", "F")]
    else:
        key_pool = list(KEYS)
    key = rng.choice(key_pool)
    lo, hi = pref["bpm"]
    bpm = float(rng.randint(int(lo), int(hi)))
    energy = _energy_for_bpm(bpm, pref["energy"], rng)
    bars_per_chord = rng.choice((1, 1, 1, 2))
    return {
        "progression": spec.name,
        "key": key,
        "bpm": bpm,
        "bars": 8,
        "bars_per_chord": bars_per_chord,
        "mode": spec.mode,
        "family": spec.family,
        "energy": energy,
    }


def _try_wrime(prompt: str) -> tuple[dict[str, float], str]:
    try:
        from emotion_va import analyze_emotion

        result = analyze_emotion(prompt)
        return dict(result.scores), str(result.top_label)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] WRIME スキップ: {exc}")
        return {}, ""


def generate_candidates(
    *,
    counts: dict[str, int],
    seed: int,
    skip_wrime: bool,
    append: bool,
) -> Path:
    rng = random.Random(seed)
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    existing = _load_rows(CANDIDATES_PATH) if append else []
    next_idx = _next_id(existing) if append else 1
    records: list[dict] = []

    for group, n in counts.items():
        if n <= 0:
            continue
        if group not in _PROMPT_GROUPS:
            raise ValueError(f"unknown prompt group: {group}")
        prompts = _PROMPT_GROUPS[group]
        for i in range(n):
            prompt = prompts[i % len(prompts)]
            if i >= len(prompts):
                prompt = f"{prompt}（追加{i // len(prompts) + 1}）"

            if skip_wrime:
                raise SystemExit(
                    "--skip-wrime は廃止しました。候補生成には WRIME→V/A が必須です。"
                )
            wrime_scores, wrime_label = _try_wrime(prompt)
            if not wrime_scores:
                raise SystemExit(f"WRIME 失敗のため中断: prompt={prompt!r}")
            v, a = wrime_to_va(wrime_scores)
            structure = _sample_structure_for_va(v, a, rng)
            spec = next(p for p in PROGRESSIONS if p.name == structure["progression"])
            music = generate_progression_strum(
                spec=spec,
                key=structure["key"],
                bpm=int(structure["bpm"]),
                bars=int(structure["bars"]),
                bars_per_chord=int(structure["bars_per_chord"]),
                rng=random.Random(rng.randint(0, 10**9)),
            )
            pair_id = f"pp_{next_idx:06d}"
            next_idx += 1
            rel_midi = f"midi/{pair_id}.mid"
            save_music(music, OUT_ROOT / rel_midi)

            records.append(
                {
                    "id": pair_id,
                    "prompt": prompt,
                    "prompt_group": group,
                    "emotion_wrime": wrime_scores,
                    "emotion_label": wrime_label,
                    "va": {"valence": v, "arousal": a},
                    "structure": structure,
                    "midi_path": rel_midi.replace("\\", "/"),
                    "gate": {
                        "status": "pending",
                        "rater": None,
                        "score_fit": None,
                        "note": "",
                    },
                }
            )
            if len(records) % 10 == 0:
                print(f"  generated {len(records)} new ...")

    all_rows = existing + records if append else records
    _write_rows(CANDIDATES_PATH, all_rows)
    pending = [r for r in all_rows if (r.get("gate") or {}).get("status") == "pending"]
    _write_rows(MANIFEST_DIR / "pending.jsonl", pending)

    print(f"new={len(records)} total={len(all_rows)} pending={len(pending)}")
    print(f"wrote → {CANDIDATES_PATH}")
    return CANDIDATES_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="prior 用短い対応データ候補を生成")
    parser.add_argument(
        "--split",
        action="store_true",
        help="既存 candidates を accept/reject/pending に振り分けて終了",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="既存を消さず追記（ID 継続）",
    )
    parser.add_argument("--per-group", type=int, default=None, help="全プロンプト群に同数")
    parser.add_argument("--bright", type=int, default=None)
    parser.add_argument("--sad", type=int, default=None)
    parser.add_argument("--calm", type=int, default=None)
    parser.add_argument("--tense", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.split:
        split_gates()
        return

    if args.per_group is not None:
        counts = {g: args.per_group for g in _PROMPT_GROUPS}
    else:
        counts = {
            "bright": 40 if args.bright is None else args.bright,
            "calm": 40 if args.calm is None else args.calm,
            "sad": 15 if args.sad is None else args.sad,
            "tense": 15 if args.tense is None else args.tense,
        }
        for key in ("bright", "sad", "calm", "tense"):
            val = getattr(args, key)
            if val is not None:
                counts[key] = val

    if not any(counts.values()):
        raise SystemExit("生成件数が 0 です")

    generate_candidates(
        counts=counts,
        seed=args.seed,
        skip_wrime=False,
        append=args.append,
    )


if __name__ == "__main__":
    main()
