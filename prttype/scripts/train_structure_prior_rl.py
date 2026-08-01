"""Offline RL for structure prior using gated accept/reject as reward.

Logged (WRIME, structure, reward) from candidates.jsonl — no new listening.

Objective (conservative / offline-safe):
  L = E[ -A * log π(a|s) ] + β * KL(π || π_ref)
where π_ref is a frozen copy of the supervised init, and
  A = reward - batch_mean(reward), reward ∈ {+1 accept, -1 reject}.

  python scripts/train_structure_prior_rl.py \\
    --jsonl data/prior_pairs/manifests/candidates.jsonl \\
    --init-checkpoint checkpoints/prior/prior_supervised_last.pt \\
    --checkpoint-dir checkpoints/prior \\
    --rounds 10
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structure_prior import (  # noqa: E402
    BARS_PER_CHORD,
    BPM_HI,
    BPM_LO,
    ENERGIES,
    KEYS,
    MODES,
    PROGRESSIONS,
    StructurePriorNet,
    bpm_to_unit,
    encode_features,
    feature_dim,
    load_prior,
    predict_structure,
)

BPM_STD = 0.08


def gate_reward(row: dict[str, Any]) -> float | None:
    gate = row.get("gate") or {}
    status = gate.get("status")
    if status == "accept":
        sf = gate.get("score_fit")
        if sf is None:
            return 1.0
        return max(0.2, min(1.0, float(sf) / 5.0))
    if status == "reject":
        return -1.0
    return None


def load_gated_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        r = gate_reward(row)
        if r is None:
            continue
        st = row.get("structure") or {}
        if not st.get("progression") or st.get("bpm") is None or not st.get("key"):
            continue
        if st["progression"] not in PROGRESSIONS or st["key"] not in KEYS:
            continue
        row = dict(row)
        row["_reward"] = float(r)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"ゲート済み行がありません: {path}")
    return rows


class OfflineRLDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], *, use_emotion_target: bool = True) -> None:
        self.rows = rows
        self.use_emotion_target = use_emotion_target
        self.prog_i = {n: i for i, n in enumerate(PROGRESSIONS)}
        self.key_i = {n: i for i, n in enumerate(KEYS)}
        self.mode_i = {n: i for i, n in enumerate(MODES)}
        self.energy_i = {n: i for i, n in enumerate(ENERGIES)}
        self.bpc_i = {n: i for i, n in enumerate(BARS_PER_CHORD)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        st = row["structure"]
        energy = st.get("energy")
        if energy not in self.energy_i:
            bpm = float(st["bpm"])
            energy = "low" if bpm < 90 else ("high" if bpm >= 120 else "mid")
        mode = st.get("mode") or "major"
        if mode not in self.mode_i:
            mode = "major"
        bpc = int(st.get("bars_per_chord") or 1)
        if bpc not in self.bpc_i:
            bpc = 1 if bpc < 2 else 2
        x = encode_features(
            wrime=row.get("emotion_wrime"),
            emotion_target=row.get("emotion_target") or row.get("emotion_label"),
            use_emotion_target=self.use_emotion_target,
        )
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "bpm": torch.tensor(bpm_to_unit(float(st["bpm"])), dtype=torch.float32),
            "energy": torch.tensor(self.energy_i[energy], dtype=torch.long),
            "mode": torch.tensor(self.mode_i[mode], dtype=torch.long),
            "key": torch.tensor(self.key_i[st["key"]], dtype=torch.long),
            "prog": torch.tensor(self.prog_i[st["progression"]], dtype=torch.long),
            "bpc": torch.tensor(self.bpc_i[bpc], dtype=torch.long),
            "reward": torch.tensor(float(row["_reward"]), dtype=torch.float32),
        }


def action_log_prob(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    bpm_mean = torch.sigmoid(out["bpm"])
    log_bpm = -0.5 * ((batch["bpm"] - bpm_mean) / BPM_STD) ** 2 - float(
        torch.log(torch.tensor(BPM_STD * (2.0 * 3.1415926535) ** 0.5))
    )
    log_e = F.log_softmax(out["energy"], dim=-1).gather(1, batch["energy"].unsqueeze(1)).squeeze(1)
    log_m = F.log_softmax(out["mode"], dim=-1).gather(1, batch["mode"].unsqueeze(1)).squeeze(1)
    log_k = F.log_softmax(out["key"], dim=-1).gather(1, batch["key"].unsqueeze(1)).squeeze(1)
    log_p = F.log_softmax(out["prog"], dim=-1).gather(1, batch["prog"].unsqueeze(1)).squeeze(1)
    log_b = F.log_softmax(out["bpc"], dim=-1).gather(1, batch["bpc"].unsqueeze(1)).squeeze(1)
    return log_bpm + log_e + log_m + 0.8 * log_k + 0.5 * log_p + 0.5 * log_b


def kl_to_ref(out: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean KL(π || π_ref) over discrete heads (+ BPM MSE proxy)."""
    kl = 0.0
    for name in ("energy", "mode", "key", "prog", "bpc"):
        log_p = F.log_softmax(out[name], dim=-1)
        log_q = F.log_softmax(ref[name], dim=-1)
        p = log_p.exp()
        kl = kl + (p * (log_p - log_q)).sum(-1).mean()
    bpm_p = torch.sigmoid(out["bpm"])
    bpm_q = torch.sigmoid(ref["bpm"])
    kl = kl + F.mse_loss(bpm_p, bpm_q)
    return kl


@torch.inference_mode()
def eval_policy_alignment(
    model: StructurePriorNet,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    n_pos = n_neg = 0
    hit_pos = hit_neg = 0
    logp_pos = logp_neg = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["x"])
        lp = action_log_prob(out, batch)
        ok = (out["energy"].argmax(-1) == batch["energy"]) & (out["mode"].argmax(-1) == batch["mode"])
        for i in range(batch["x"].shape[0]):
            r = float(batch["reward"][i].item())
            if r > 0:
                n_pos += 1
                hit_pos += int(ok[i].item())
                logp_pos += float(lp[i].item())
            else:
                n_neg += 1
                hit_neg += int(ok[i].item())
                logp_neg += float(lp[i].item())
    return {
        "accept_energy_mode_acc": hit_pos / max(1, n_pos),
        "reject_energy_mode_acc": hit_neg / max(1, n_neg),
        "accept_mean_logp": logp_pos / max(1, n_pos),
        "reject_mean_logp": logp_neg / max(1, n_neg),
        "gap_logp": (logp_pos / max(1, n_pos)) - (logp_neg / max(1, n_neg)),
        "n_accept": float(n_pos),
        "n_reject": float(n_neg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RL for structure prior (KL-regularized)")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "data" / "prior_pairs" / "manifests" / "candidates.jsonl",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "prior" / "prior_supervised_last.pt",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "prior")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--epochs-per-round", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-beta", type=float, default=0.5, help="KL(π||π_ref) weight")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-emotion-target", action="store_true")
    args = parser.parse_args()

    use_et = not args.no_emotion_target
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    jsonl = args.jsonl if args.jsonl.is_absolute() else (ROOT / args.jsonl)
    init_ckpt = args.init_checkpoint if args.init_checkpoint.is_absolute() else (ROOT / args.init_checkpoint)
    if not init_ckpt.is_file():
        alt = ROOT / "checkpoints" / "prior" / "prior_last.pt"
        if alt.is_file():
            init_ckpt = alt
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else (ROOT / args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows = load_gated_rows(jsonl)
    print(f"rows={len(rows)} from {jsonl}")
    print("reward", dict(Counter(round(x, 2) for x in (r["_reward"] for r in rows))))
    print("status", dict(Counter((r.get("gate") or {}).get("status") for r in rows)))

    ds = OfflineRLDataset(rows, use_emotion_target=use_et)
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True)
    eval_loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not init_ckpt.is_file():
        raise FileNotFoundError(f"init checkpoint がありません: {init_ckpt}")

    model, blob = load_prior(init_ckpt, device=device)
    meta = blob.get("meta") or {}
    use_et = bool(meta.get("use_emotion_target", use_et))
    print(f"init from {init_ckpt}")

    # frozen reference policy (supervised)
    ref_model, _ = load_prior(init_ckpt, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history: list[dict[str, Any]] = []
    base = eval_policy_alignment(model, eval_loader, device=device)
    print(
        f"round 000  accept_acc={base['accept_energy_mode_acc']:.3f}  "
        f"reject_acc={base['reject_energy_mode_acc']:.3f}  "
        f"accept_logp={base['accept_mean_logp']:.2f}  "
        f"reject_logp={base['reject_mean_logp']:.2f}  "
        f"gap={base['gap_logp']:.2f}"
    )
    history.append({"round": 0, **base})

    best_gap = base["gap_logp"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for rnd in range(1, args.rounds + 1):
        model.train()
        ep_loss = ep_pg = ep_kl = 0.0
        ep_steps = 0
        for _ in range(args.epochs_per_round):
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["x"])
                with torch.no_grad():
                    ref_out = ref_model(batch["x"])
                logp = action_log_prob(out, batch)
                reward = batch["reward"]
                advantage = reward - reward.mean()
                pg_loss = -(advantage.detach() * logp).mean()
                kl = kl_to_ref(out, ref_out)
                loss = pg_loss + args.kl_beta * kl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item())
                ep_pg += float(pg_loss.item())
                ep_kl += float(kl.item())
                ep_steps += 1

        model.eval()
        metrics = eval_policy_alignment(model, eval_loader, device=device)
        n = max(1, ep_steps)
        print(
            f"round {rnd:03d}  loss={ep_loss/n:.4f}  pg={ep_pg/n:.4f}  kl={ep_kl/n:.4f}  "
            f"accept_acc={metrics['accept_energy_mode_acc']:.3f}  "
            f"reject_acc={metrics['reject_energy_mode_acc']:.3f}  "
            f"accept_logp={metrics['accept_mean_logp']:.2f}  "
            f"reject_logp={metrics['reject_mean_logp']:.2f}  "
            f"gap={metrics['gap_logp']:.2f}"
        )
        history.append(
            {
                "round": rnd,
                "loss": ep_loss / n,
                "pg": ep_pg / n,
                "kl": ep_kl / n,
                **metrics,
            }
        )

        out_path = ckpt_dir / f"prior_rl_round{rnd:02d}.pt"
        blob_out = {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "meta": {
                **meta,
                "in_dim": int(meta.get("in_dim", feature_dim(use_emotion_target=use_et))),
                "hidden": int(meta.get("hidden", 64)),
                "use_emotion_target": use_et,
                "train_mode": "offline_rl_reinforce_kl",
                "round": rnd,
                "rounds": args.rounds,
                "epochs_per_round": args.epochs_per_round,
                "lr": args.lr,
                "kl_beta": args.kl_beta,
                "jsonl": str(jsonl),
                "init_checkpoint": str(init_ckpt),
                "bpm_range": [BPM_LO, BPM_HI],
                "progressions": list(PROGRESSIONS),
                "keys": list(KEYS),
                "modes": list(MODES),
                "energies": list(ENERGIES),
            },
        }
        torch.save(blob_out, out_path)

        # prefer larger gap (accept logp >> reject logp) without tanking accept_acc
        if metrics["gap_logp"] > best_gap and metrics["accept_energy_mode_acc"] >= base["accept_energy_mode_acc"] - 0.05:
            best_gap = metrics["gap_logp"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    final_path = ckpt_dir / "prior_rl_last.pt"
    final_blob = {
        "model_state_dict": best_state,
        "meta": {
            **meta,
            "in_dim": int(meta.get("in_dim", feature_dim(use_emotion_target=use_et))),
            "hidden": int(meta.get("hidden", 64)),
            "use_emotion_target": use_et,
            "train_mode": "offline_rl_reinforce_kl",
            "selected": "best_gap_with_acc_guard",
            "best_gap_logp": best_gap,
            "rounds": args.rounds,
            "kl_beta": args.kl_beta,
            "lr": args.lr,
            "jsonl": str(jsonl),
            "init_checkpoint": str(init_ckpt),
            "bpm_range": [BPM_LO, BPM_HI],
            "progressions": list(PROGRESSIONS),
            "keys": list(KEYS),
            "modes": list(MODES),
            "energies": list(ENERGIES),
        },
    }
    torch.save(final_blob, final_path)

    supervised = ckpt_dir / "prior_supervised_last.pt"
    prior_last = ckpt_dir / "prior_last.pt"
    if prior_last.is_file() and not supervised.is_file():
        shutil.copy2(prior_last, supervised)
    shutil.copy2(final_path, prior_last)

    final_m = eval_policy_alignment(model, eval_loader, device=device)
    print(
        f"selected  accept_acc={final_m['accept_energy_mode_acc']:.3f}  "
        f"reject_acc={final_m['reject_energy_mode_acc']:.3f}  "
        f"gap={final_m['gap_logp']:.2f}"
    )
    print(f"updated {prior_last} ← {final_path}")

    (ckpt_dir / "rl_offline_summary.json").write_text(
        json.dumps({"history": history, "final": final_m, "best_gap": best_gap}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    model.eval()
    shown: set[str] = set()
    for row in rows:
        if row["_reward"] <= 0:
            continue
        lab = row.get("emotion_target") or "?"
        if lab in shown:
            continue
        shown.add(lab)
        pred = predict_structure(
            model,
            wrime=row.get("emotion_wrime"),
            emotion_target=row.get("emotion_target"),
            use_emotion_target=use_et,
            device=device,
        )
        gold = row["structure"]
        print(
            f"sample[{lab}] pred bpm={pred.bpm} energy={pred.energy} mode={pred.mode} "
            f"prog={pred.progression} | gold bpm={gold['bpm']} energy={gold.get('energy')} "
            f"prog={gold['progression']}"
        )


if __name__ == "__main__":
    main()
