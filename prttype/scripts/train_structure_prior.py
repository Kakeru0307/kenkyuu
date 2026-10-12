"""Train VA → structure prior on accept.jsonl (+ optional extra pairs).

Usage:
  python scripts/train_structure_prior.py \\
    --jsonl data/prior_pairs/manifests/accept.jsonl \\
    --extra-jsonl \\
      data/vgmidi_pairs/structure_pairs.jsonl \\
      data/emopia_pairs/structure_pairs.jsonl \\
      data/pmemo_pairs/structure_pairs.jsonl \\
      data/fme24_pairs/structure_pairs.jsonl \\
    --checkpoint-dir checkpoints/structure_prior

BPM はビン分類（soft label）。energy は学習しない（推論時に BPM から導出）。
E3: 進行を正として mode を正規化（borrowed/blues は両 mode 可）。
C1: prog ヘッドは mode one-hot 条件。学習は gold mode で teacher forcing。
extra 行で欠けた項目（BPM・mode・key・prog・bpc）は損失から除外する。
ソース均等サンプリング: accept / vgmidi / emopia / pmemo / deam / fme24 / bal_*
（high_bpm は分布を歪めるため均等ソースに含めない・推奨しない）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from makeData.progressions import e3_canonical_mode  # noqa: E402
from structure_prior import (  # noqa: E402
    BARS_PER_CHORD,
    BPM_BIN_EDGES,
    BPM_HI,
    BPM_LO,
    FEATURE_DIM,
    KEYS,
    MODES,
    N_BPM_BINS,
    PROGRESSIONS,
    StructurePriorNet,
    bin_to_bpm,
    bpm_to_bin,
    encode_features,
    energy_from_bpm,
    predict_structure,
)

# 均等サンプリング対象（high_bpm は意図的に除外）
EQUAL_SOURCES: frozenset[str] = frozenset(
    (
        "accept",
        "vgmidi",
        "emopia",
        "pmemo",
        "deam",
        "fme24",
        "bal_high_fast",
        "bal_highV_lowA",
        "bal_lowV_highA",
        "bal_lowV_lowA",
    )
)
MASKABLE_HEADS: tuple[str, ...] = ("bpm", "mode", "key", "prog", "bpc")
BPM_SOFT_NEIGHBOR = 0.15
VA_SUM_HIGH = 0.4
VA_SUM_LOW = -0.4
BPM_SLOW = 90.0
BPM_VERY_FAST = 150.0


def row_va(row: dict[str, Any]) -> tuple[float, float]:
    va = row.get("va")
    if isinstance(va, dict):
        return float(va.get("valence", 0.0)), float(va.get("arousal", 0.0))
    if isinstance(va, (list, tuple)) and len(va) >= 2:
        return float(va[0]), float(va[1])
    raise ValueError("va がありません")


def va_quadrant(v: float, a: float) -> str:
    hv = "pos" if v >= 0 else "neg"
    ha = "high" if a >= 0 else "low"
    return f"{hv}_{ha}"


def va_bpm_mismatch(v: float, a: float, bpm: float) -> bool:
    """明らかな矛盾だけ外す。低V高A×速いのは残す。"""
    s = v + a
    if s >= VA_SUM_HIGH and bpm < BPM_SLOW:
        return True
    # 低V低A×超速のみ落とす（低V高Aの緊張×速さは学習に残す）
    if v < 0.0 and a < 0.0 and bpm >= BPM_VERY_FAST:
        return True
    return False


def filter_va_bpm_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        src = str(row.get("source") or "").lower()
        if src in ("va_boost_high", "va_boost_low"):
            kept.append(row)
            continue
        st = row.get("structure") or {}
        bpm = st.get("bpm")
        if bpm is None:
            kept.append(row)
            continue
        v, a = row_va(row)
        if va_bpm_mismatch(v, a, float(bpm)):
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def apply_e3_mode_normalize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """E3: 進行を正として mode を正規化。書き換え件数を返す。"""
    fixed = 0
    out: list[dict[str, Any]] = []
    for row in rows:
        st = dict(row.get("structure") or {})
        prog = st.get("progression")
        mask = dict(row.get("_label_mask") or {})
        has_prog = prog in PROGRESSIONS and mask.get("prog", True)
        has_mode_label = mask.get("mode", True) if mask else (st.get("mode") in MODES)
        # accept 行は mask 無し＝両ラベルありとみなす
        if not mask:
            has_prog = prog in PROGRESSIONS
            has_mode_label = True
        if has_prog and prog in PROGRESSIONS:
            old = st.get("mode")
            new = e3_canonical_mode(str(prog), old if old in MODES else None)
            if old != new:
                fixed += 1
                st["mode"] = new
            elif old not in MODES:
                st["mode"] = new
                if has_mode_label:
                    fixed += 1
            row = {**row, "structure": st}
        out.append(row)
    return out, fixed


def load_accept_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gate = (row.get("gate") or {}).get("status")
        if gate not in (None, "accept"):
            continue
        if gate is None and path.name != "accept.jsonl":
            continue
        st = row.get("structure") or {}
        if not st.get("progression") or st.get("bpm") is None or not st.get("key"):
            continue
        if st["progression"] not in PROGRESSIONS:
            continue
        if st["key"] not in KEYS:
            continue
        row["source"] = "accept"
        rows.append(row)
    if not rows:
        raise RuntimeError(f"学習可能な accept 行がありません: {path}")
    return rows


def source_of(row: dict[str, Any]) -> str:
    src = str(row.get("source") or "unknown").strip().lower()
    if src in EQUAL_SOURCES:
        return src
    # high_bpm 等は均等バケットに入れず、自然頻度に近い unknown 扱い
    return "unknown"


def make_source_weights(rows: list[dict[str, Any]], indices: list[int]) -> torch.Tensor:
    """source ごとの件数の逆数を重みにする。期待サンプリング比率をソース間で揃える。"""
    counts: Counter[str] = Counter(source_of(rows[i]) for i in indices)
    weights = []
    for i in indices:
        src = source_of(rows[i])
        n = max(1, counts[src])
        weights.append(1.0 / float(n))
    return torch.tensor(weights, dtype=torch.double)


def load_extra_rows(path: Path) -> list[dict[str, Any]]:
    """VGMIDI / EMOPIA 等の追加 JSONL を読む。

    無い項目は損失から除外する（_label_mask）。
    VA が無い行は捨てる。BPM が無い行（EMOPIA）は bpm をマスク。
    """
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        st = dict(row.get("structure") or {})
        if row.get("va") is None:
            continue
        has_bpm = st.get("bpm") is not None
        mask = {
            "bpm": has_bpm,
            "mode": st.get("mode") in MODES,
            "prog": st.get("progression") in PROGRESSIONS,
            "key": st.get("key") in KEYS,
            "bpc": st.get("bars_per_chord") in BARS_PER_CHORD,
        }
        if not has_bpm:
            st["bpm"] = 120.0  # プレースホルダ（損失ではマスク）
        if not mask["mode"]:
            st["mode"] = MODES[0]
        if not mask["prog"]:
            st["progression"] = PROGRESSIONS[0]
        if not mask["key"]:
            st["key"] = KEYS[0]
        if not mask["bpc"]:
            st["bars_per_chord"] = BARS_PER_CHORD[0]
        row["structure"] = st
        row["_label_mask"] = mask
        rows.append(row)
    return rows


def bpm_soft_targets(bin_idx: torch.Tensor, n_bins: int, neighbor: float = BPM_SOFT_NEIGHBOR) -> torch.Tensor:
    """正解ビン＋隣に質量を薄く置く soft target（行ごとに正規化）。"""
    b = bin_idx.long()
    batch = b.shape[0]
    t = torch.zeros(batch, n_bins, device=b.device, dtype=torch.float32)
    center = 1.0 - 2.0 * float(neighbor)
    t.scatter_(1, b.unsqueeze(1), center)
    left = (b - 1).clamp(min=0)
    right = (b + 1).clamp(max=n_bins - 1)
    ones = torch.ones(batch, 1, device=b.device, dtype=torch.float32) * float(neighbor)
    t.scatter_add_(1, left.unsqueeze(1), ones)
    t.scatter_add_(1, right.unsqueeze(1), ones)
    return t / t.sum(dim=1, keepdim=True).clamp(min=1e-8)


class PriorPairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.prog_i = {n: i for i, n in enumerate(PROGRESSIONS)}
        self.key_i = {n: i for i, n in enumerate(KEYS)}
        self.mode_i = {n: i for i, n in enumerate(MODES)}
        self.bpc_i = {n: i for i, n in enumerate(BARS_PER_CHORD)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        st = row["structure"]
        mode = st.get("mode") or "major"
        if mode not in self.mode_i:
            mode = "major"
        bpc = int(st.get("bars_per_chord") or 1)
        if bpc not in self.bpc_i:
            bpc = 1 if bpc < 2 else 2

        x = encode_features(va=row_va(row))
        label_mask = row.get("_label_mask") or {}
        item = {
            "x": torch.tensor(x, dtype=torch.float32),
            "bpm": torch.tensor(bpm_to_bin(float(st["bpm"])), dtype=torch.long),
            "bpm_value": torch.tensor(float(st["bpm"]), dtype=torch.float32),
            "mode": torch.tensor(self.mode_i[mode], dtype=torch.long),
            "key": torch.tensor(self.key_i[st["key"]], dtype=torch.long),
            "prog": torch.tensor(self.prog_i[st["progression"]], dtype=torch.long),
            "bpc": torch.tensor(self.bpc_i[bpc], dtype=torch.long),
        }
        for head in MASKABLE_HEADS:
            item[f"{head}_mask"] = torch.tensor(
                1.0 if label_mask.get(head, True) else 0.0, dtype=torch.float32
            )
        return item


def stratified_split(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        v, a = row_va(r)
        lab = va_quadrant(v, a)
        by_label.setdefault(lab, []).append(i)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for _lab, idxs in by_label.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio))) if len(idxs) >= 5 else max(0, len(idxs) // 5)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    if not train_idx:
        train_idx, val_idx = val_idx[:-1] or val_idx, val_idx[-1:] if val_idx else []
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def batch_loss(
    model: StructurePriorNet,
    batch: dict[str, torch.Tensor],
    *,
    ce: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    # C1: prog は gold mode で teacher forcing
    out = model(batch["x"], mode_index=batch["mode"])

    def masked_ce(head: str) -> torch.Tensor:
        per = ce(out[head], batch[head])
        mask = batch[f"{head}_mask"]
        return (per * mask).sum() / mask.sum().clamp(min=1.0)

    soft = bpm_soft_targets(batch["bpm"], N_BPM_BINS)
    log_p = F.log_softmax(out["bpm"], dim=-1)
    bpm_per = -(soft * log_p).sum(dim=-1)
    bpm_mask = batch["bpm_mask"]
    bpm_loss = (bpm_per * bpm_mask).sum() / bpm_mask.sum().clamp(min=1.0)

    losses = {
        "bpm": bpm_loss,
        "mode": masked_ce("mode"),
        "key": masked_ce("key"),
        "prog": masked_ce("prog"),
        "bpc": masked_ce("bpc"),
    }
    total = (
        2.5 * losses["bpm"]
        + 1.0 * losses["mode"]
        + 0.8 * losses["key"]
        + 0.6 * losses["prog"]
        + 0.5 * losses["bpc"]
    )
    stats = {k: float(v.detach().item()) for k, v in losses.items()}
    stats["total"] = float(total.detach().item())
    return total, stats


@torch.inference_mode()
def evaluate(
    model: StructurePriorNet,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    if len(loader.dataset) == 0:
        return {}
    ce = nn.CrossEntropyLoss(reduction="none")
    totals: Counter[str] = Counter()
    n = 0
    correct: Counter[str] = Counter()
    counted: Counter[str] = Counter()
    bpm_abs = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["x"], mode_index=batch["mode"])
        _, stats = batch_loss(model, batch, ce=ce)
        for k, v in stats.items():
            totals[k] += v
        pred_bin = out["bpm"].argmax(-1)
        bpm_mask = batch["bpm_mask"]
        pred_bpm = torch.tensor(
            [bin_to_bpm(int(i)) for i in pred_bin.detach().cpu().tolist()],
            device=device,
            dtype=torch.float32,
        )
        bpm_abs += float(((pred_bpm - batch["bpm_value"]).abs() * bpm_mask).sum().item())
        counted["bpm"] += float(bpm_mask.sum().item())
        hit_bpm = (pred_bin == batch["bpm"]).float()
        correct["bpm"] += float((hit_bpm * bpm_mask).sum().item())
        for name in ("mode", "key", "prog", "bpc"):
            hit = (out[name].argmax(-1) == batch[name]).float()
            mask = batch.get(f"{name}_mask", torch.ones_like(hit))
            correct[name] += float((hit * mask).sum().item())
            counted[name] += float(mask.sum().item())
        n += batch["x"].shape[0]
    if n == 0:
        return {}
    nb = max(1, len(loader))
    metrics = {f"loss_{k}": totals[k] / nb for k in totals}
    metrics["mae_bpm"] = bpm_abs / max(1.0, counted["bpm"])
    metrics["acc_bpm"] = correct["bpm"] / max(1.0, counted["bpm"])
    for name in ("mode", "key", "prog", "bpc"):
        metrics[f"acc_{name}"] = correct[name] / max(1.0, counted[name])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VA→structure prior (BPM bins)")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "data" / "prior_pairs" / "manifests" / "accept.jsonl",
    )
    parser.add_argument(
        "--extra-jsonl",
        type=Path,
        default=None,
        help="EMOPIA 等の追加学習データ JSONL（複数指定可）",
        nargs="*",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "structure_prior")
    parser.add_argument("--ckpt-name", type=str, default="structure_prior_last.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-filter-va-bpm-align",
        action="store_true",
        help="高VA×遅BPM / 低VA×速BPM の除外をしない",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    jsonl = args.jsonl if args.jsonl.is_absolute() else (ROOT / args.jsonl)
    if not jsonl.is_file():
        alt = ROOT / "data" / "prior_pairs" / "accept.jsonl"
        if alt.is_file():
            jsonl = alt
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else (ROOT / args.checkpoint_dir)
    rows = load_accept_rows(jsonl)
    print(f"rows={len(rows)} from {jsonl}")
    if args.extra_jsonl:
        for extra_path in args.extra_jsonl:
            extra_p = extra_path if extra_path.is_absolute() else (ROOT / extra_path)
            if "high_bpm" in extra_p.as_posix().lower():
                print(f"[警告] high_bpm は分布学習では推奨しません（スキップ）: {extra_p}")
                continue
            if extra_p.is_file():
                extra = load_extra_rows(extra_p)
                print(f"extra rows={len(extra)} from {extra_p}")
                rows = rows + extra
            else:
                print(f"[警告] extra-jsonl が見つかりません: {extra_p}")
    if not args.no_filter_va_bpm_align:
        before = len(rows)
        rows, dropped = filter_va_bpm_rows(rows)
        if dropped:
            print(f"va_bpm_align: dropped {dropped} mismatch rows ({before} -> {len(rows)})")
    rows, e3_fixed = apply_e3_mode_normalize(rows)
    if e3_fixed:
        print(f"e3_mode_normalize: rewrote mode on {e3_fixed} rows")
    print(f"合計 rows={len(rows)}")
    print("va_quadrant", dict(Counter(va_quadrant(*row_va(r)) for r in rows)))
    print(f"bpm_bins={N_BPM_BINS} edges={list(BPM_BIN_EDGES)}")

    train_idx, val_idx = stratified_split(rows, val_ratio=args.val_ratio, seed=args.seed)
    ds = PriorPairDataset(rows)
    train_source_counts = Counter(source_of(rows[i]) for i in train_idx)
    print(f"split train={len(train_idx)} val={len(val_idx)}")
    print(f"train_by_source {dict(train_source_counts)}")
    train_weights = make_source_weights(rows, train_idx)
    weight_sum_by_source: Counter[str] = Counter()
    for i, w in zip(train_idx, train_weights.tolist()):
        weight_sum_by_source[source_of(rows[i])] += float(w)
    print(f"train_weight_sum_by_source {dict(weight_sum_by_source)}")
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_idx),
        replacement=True,
    )
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=min(args.batch_size, max(1, len(train_idx))),
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=min(args.batch_size, max(1, len(val_idx) or 1)),
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = FEATURE_DIM
    model = StructurePriorNet(in_dim, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = batch_loss(model, batch, ce=ce)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            steps += 1
        model.eval()
        val_m = evaluate(model, val_loader, device=device)
        train_loss = running / max(1, steps)
        val_loss = val_m.get("loss_total", train_loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                f"acc_bpm={val_m.get('acc_bpm', 0):.2f}  "
                f"acc_mode={val_m.get('acc_mode', 0):.2f}  "
                f"mae_bpm={val_m.get('mae_bpm', 0):.1f}  "
                f"acc_prog={val_m.get('acc_prog', 0):.2f}"
            )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = ckpt_dir / args.ckpt_name
    blob = {
        "model_state_dict": model.state_dict(),
        "meta": {
            "in_dim": in_dim,
            "hidden": args.hidden,
            "feature": "va2",
            "bpm_head": "bins",
            "n_bpm_bins": N_BPM_BINS,
            "bpm_bin_edges": list(BPM_BIN_EDGES),
            "energy": "from_bpm",
            "prog_cond": "mode_oh",
            "e3_mode_normalize": True,
            "extra_jsonl": [str(p) for p in (args.extra_jsonl or [])],
            "epochs": args.epochs,
            "lr": args.lr,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "best_val_loss": best_val,
            "jsonl": str(jsonl),
            "progressions": list(PROGRESSIONS),
            "keys": list(KEYS),
            "modes": list(MODES),
            "bpm_range": [BPM_LO, BPM_HI],
        },
    }
    torch.save(blob, ckpt_path)
    print(f"wrote {ckpt_path}")

    model.eval()
    shown: set[str] = set()
    for row in rows:
        v, a = row_va(row)
        lab = va_quadrant(v, a)
        if lab in shown:
            continue
        shown.add(lab)
        pred = predict_structure(
            model,
            va=(v, a),
            device=device,
            sample=False,
        )
        gold = row["structure"]
        print(
            f"sample[{lab} V={v:+.2f} A={a:+.2f}] pred bpm={pred.bpm} energy={pred.energy} "
            f"mode={pred.mode} prog={pred.progression} | gold bpm={gold['bpm']} "
            f"energy={energy_from_bpm(float(gold['bpm']))} mode={gold.get('mode')} "
            f"prog={gold['progression']}"
        )


if __name__ == "__main__":
    main()
