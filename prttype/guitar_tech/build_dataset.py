"""Seed technique MIDIs → augmented frame dataset + dictionary."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .constants import DATASET_DIR, N_FRAMES, TECH_ROOT
from .frames import midi_to_frames


def list_seed_midis(root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    if not root.is_dir():
        raise FileNotFoundError(f"technique root missing: {root}")
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mids = sorted(d.glob("*.mid")) + sorted(d.glob("*.midi"))
        for mid in mids:
            items.append((d.name, mid))
    if not items:
        raise RuntimeError(f"no MIDI under {root}")
    return items


def build(
    *,
    seed_root: Path = TECH_ROOT,
    out_dir: Path = DATASET_DIR,
    per_seed: int = 24,
    seed: int = 0,
) -> Path:
    rng = random.Random(seed)
    seeds = list_seed_midis(seed_root)
    tech_names = sorted({name for name, _ in seeds})
    tech_to_id = {n: i for i, n in enumerate(tech_names)}

    xs: list[np.ndarray] = []
    ys: list[int] = []
    meta_rows: list[dict] = []

    for tech, path in seeds:
        tid = tech_to_id[tech]
        for k in range(per_seed):
            transpose = rng.randint(-5, 7)
            pb_scale = rng.uniform(0.85, 1.15)
            frames = midi_to_frames(
                path, n_frames=N_FRAMES, transpose=transpose, pb_scale=pb_scale
            )
            xs.append(frames)
            ys.append(tid)
            meta_rows.append(
                {
                    "tech": tech,
                    "tech_id": tid,
                    "source": str(path.relative_to(seed_root.parent)).replace("\\", "/"),
                    "transpose": transpose,
                    "pb_scale": round(pb_scale, 3),
                    "aug_index": k,
                }
            )

    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "frames.npz", x=x, y=y)

    dictionary = {
        "tech_names": tech_names,
        "tech_to_id": tech_to_id,
        "n_frames": N_FRAMES,
        "n_samples": int(x.shape[0]),
        "per_seed": per_seed,
        "seeds": [
            {"tech": t, "path": str(p.relative_to(seed_root.parent)).replace("\\", "/")}
            for t, p in seeds
        ],
    }
    (out_dir / "dictionary.json").write_text(
        json.dumps(dictionary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in meta_rows) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {out_dir}: samples={x.shape[0]} techs={len(tech_names)} "
        f"shape={tuple(x.shape)}"
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build guitar technique frame dataset")
    parser.add_argument("--seed-root", type=Path, default=TECH_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--per-seed", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(
        seed_root=args.seed_root,
        out_dir=args.out_dir,
        per_seed=args.per_seed,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
