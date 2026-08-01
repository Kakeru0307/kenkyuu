"""条件 prior 用の短い対応データ候補を生成する。

機械生成は gate=pending の候補。学習には人手で accept したものだけを使う。
出力: data/prior_pairs/manifests/candidates.jsonl + midi/*.mid

--append で既存を消さず追記。ID は既存最大の次から。
--split で accept.jsonl / reject.jsonl に振り分け（candidates は残す）。
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

from makeData.constants import KEYS
from makeData.patterns import generate_progression_strum
from makeData.progressions import PROGRESSIONS, ProgressionSpec
from patch_to_midi import save_music

OUT_ROOT = SCRIPT_DIR / "data" / "prior_pairs"
MIDI_DIR = OUT_ROOT / "midi"
MANIFEST_DIR = OUT_ROOT / "manifests"
CANDIDATES_PATH = MANIFEST_DIR / "candidates.jsonl"

_EMOTION_PROMPTS: dict[str, list[str]] = {
    "joy": [
        "明るいギターバッキング",
        "楽しい午後の伴奏",
        "元気が出るストローク",
        "晴れやかなコード進行",
        "弾むようなバッキング",
        "明るい朝のギター",
        "テンポよく進む伴奏",
        "明るいポップなバッキング",
    ],
    "sadness": [
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
    "tension": [
        "緊迫したギターバッキング",
        "焦りのある伴奏",
        "張りつめたストローク",
        "不穏なコード進行",
        "緊張感のあるバッキング",
        "せわしないギター伴奏",
        "追い立てられるようなバッキング",
    ],
}

_EMOTION_STRUCT: dict[str, dict] = {
    "joy": {
        "modes": ("major",),
        "bpm": (108, 140),
        "energy": ("mid", "high"),
    },
    "sadness": {
        "modes": ("natural_minor",),
        "bpm": (60, 95),
        "energy": ("low", "mid"),
    },
    "calm": {
        "modes": ("natural_minor", "major"),
        "bpm": (70, 105),
        "energy": ("low", "mid"),
    },
    "tension": {
        "modes": ("natural_minor", "major"),
        "bpm": (112, 150),
        "energy": ("mid", "high"),
    },
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


def _sample_structure_for_emotion(emotion: str, rng: random.Random) -> dict:
    pref = _EMOTION_STRUCT[emotion]
    modes = pref["modes"]
    pool = [p for p in PROGRESSIONS if p.mode in modes] or list(PROGRESSIONS)
    if emotion == "tension":
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
        from wrime_emotion import analyze_emotion

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

    for emotion, n in counts.items():
        if n <= 0:
            continue
        if emotion not in _EMOTION_PROMPTS:
            raise ValueError(f"unknown emotion: {emotion}")
        prompts = _EMOTION_PROMPTS[emotion]
        for i in range(n):
            prompt = prompts[i % len(prompts)]
            if i >= len(prompts):
                prompt = f"{prompt}（追加{i // len(prompts) + 1}）"
            structure = _sample_structure_for_emotion(emotion, rng)
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

            if skip_wrime:
                wrime_scores, wrime_label = {}, ""
            else:
                wrime_scores, wrime_label = _try_wrime(prompt)

            records.append(
                {
                    "id": pair_id,
                    "prompt": prompt,
                    "emotion_target": emotion,
                    "emotion_wrime": wrime_scores,
                    "emotion_label": wrime_label or emotion,
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
    # pending 一覧も更新（ゲートしやすく）
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
    parser.add_argument("--per-emotion", type=int, default=None, help="全感情に同数")
    parser.add_argument("--joy", type=int, default=None)
    parser.add_argument("--sadness", type=int, default=None)
    parser.add_argument("--calm", type=int, default=None)
    parser.add_argument("--tension", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-wrime", action="store_true")
    args = parser.parse_args()

    if args.split:
        split_gates()
        return

    if args.per_emotion is not None:
        counts = {e: args.per_emotion for e in _EMOTION_PROMPTS}
    else:
        # 追加生成の既定: joy/calm を厚く
        counts = {
            "joy": 40 if args.joy is None else args.joy,
            "calm": 40 if args.calm is None else args.calm,
            "sadness": 15 if args.sadness is None else args.sadness,
            "tension": 15 if args.tension is None else args.tension,
        }
        # 個別指定があれば上書き（None 以外）
        for key in ("joy", "sadness", "calm", "tension"):
            val = getattr(args, key)
            if val is not None:
                counts[key] = val

    # 全部 None で per-emotion も無しのとき、明示カウントが 0 ばかりならエラー回避
    if not any(counts.values()):
        raise SystemExit("生成件数が 0 です")

    generate_candidates(
        counts=counts,
        seed=args.seed,
        skip_wrime=args.skip_wrime,
        append=args.append,
    )


if __name__ == "__main__":
    main()
