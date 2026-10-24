"""Sample technique MIDIs from a trained TechFrameAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .constants import CHECKPOINT_DIR, DATASET_DIR, OUTPUT_DIR
from .frames import frames_to_midi
from .model import TechFrameAE


def load_model(ckpt_path: Path, device: torch.device) -> tuple[TechFrameAE, dict]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = blob.get("meta") or {}
    model = TechFrameAE(
        int(meta["n_tech"]),
        hidden=int(meta.get("hidden", 128)),
        z_dim=int(meta.get("z_dim", 16)),
    )
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    model.eval()
    return model, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate guitar technique MIDI")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "guitar_tech_last.pt",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tech", type=str, default=None, help="奏法名。省略で全種")
    parser.add_argument("--per-tech", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.15,
        help="学習サンプルの z に足すノイズ（大きいほど多様）",
    )
    parser.add_argument("--duration", type=float, default=4.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.checkpoint, device)
    names: list[str] = list(meta["tech_names"])
    name_to_id = {n: i for i, n in enumerate(names)}
    if args.tech is not None:
        if args.tech not in name_to_id:
            raise SystemExit(f"unknown tech {args.tech!r}. candidates: {names}")
        targets = [args.tech]
    else:
        targets = names

    ds = np.load(args.dataset_dir / "frames.npz")
    x_all = torch.from_numpy(ds["x"]).float()
    y_all = torch.from_numpy(ds["y"]).long()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    written: list[str] = []
    for tech in targets:
        tid = name_to_id[tech]
        idxs = torch.nonzero(y_all == tid, as_tuple=False).flatten().tolist()
        if not idxs:
            print(f"[skip] no dataset rows for {tech}")
            continue
        for i in range(args.per_tech):
            src_i = int(idxs[int(rng.integers(0, len(idxs)))])
            xb = x_all[src_i : src_i + 1].to(device)
            yb = torch.tensor([tid], device=device)
            with torch.no_grad():
                z = model.encode(xb)
                if args.noise > 0:
                    z = z + args.noise * torch.randn_like(z)
                frames = model.decode(yb, z)
                frames[..., 0] = frames[..., 0].clamp(0.0, 1.0)
                frames[..., 1] = frames[..., 1].clamp(0.0, 1.0)
                frames[..., 2] = frames[..., 2].clamp(-1.0, 1.0)
            midi = frames_to_midi(
                frames[0].detach().cpu().numpy(),
                duration_sec=args.duration,
            )
            path = out_dir / f"{tech}_{args.seed}_{i:02d}.mid"
            midi.write(str(path))
            written.append(str(path))
            print(f"wrote {path}")

    (out_dir / "last_generate.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "files": written,
                "techs": targets,
                "noise": args.noise,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
