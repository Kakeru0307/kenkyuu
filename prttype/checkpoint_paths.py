"""チェックポイントファイル名の統一ルール。

命名:
  {role}_unet_last.pt   … 通常 U-Net
  {role}_cvae_last.pt   … CVAE
  structure_prior_last.pt
  template_prior_last.pt

旧名（unet_last.pt / prior_last.pt / last.pt）も読み込み時にフォールバックする。
"""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINTS = SCRIPT_DIR / "checkpoints"

STRUCTURE_PRIOR_NAME = "structure_prior_last.pt"
TEMPLATE_PRIOR_NAME = "template_prior_last.pt"


def part_ckpt_name(role: str, *, cvae: bool) -> str:
    kind = "cvae" if cvae else "unet"
    return f"{role}_{kind}_last.pt"


def part_ckpt_path(role: str, checkpoint_dir: Path, *, cvae: bool) -> Path:
    return Path(checkpoint_dir) / part_ckpt_name(role, cvae=cvae)


def resolve_part_checkpoint(
    role: str,
    checkpoint_dir: Path | None = None,
    *,
    prefer_cvae: bool = True,
) -> Path:
    """役割フォルダから最新の演奏モデル ckpt を解決する。

    優先順: cvae → unet → 旧 unet_last.pt
    どれも無ければ期待パス（unet）を返す（呼び出し側で FileNotFoundError になる）。
    """
    root = Path(checkpoint_dir) if checkpoint_dir is not None else (CHECKPOINTS / role)
    cvae_path = root / part_ckpt_name(role, cvae=True)
    unet_path = root / part_ckpt_name(role, cvae=False)
    legacy = root / "unet_last.pt"
    if prefer_cvae and cvae_path.is_file():
        return cvae_path
    if unet_path.is_file():
        return unet_path
    if legacy.is_file():
        return legacy
    if prefer_cvae and cvae_path.is_file():
        return cvae_path
    return unet_path


def resolve_structure_prior_checkpoint(checkpoint_dir: Path | None = None) -> Path:
    root = Path(checkpoint_dir) if checkpoint_dir is not None else (CHECKPOINTS / "structure_prior")
    primary = root / STRUCTURE_PRIOR_NAME
    legacy = root / "prior_last.pt"
    legacy_va = root / "v2_va.pt"
    if primary.is_file():
        return primary
    if legacy.is_file():
        return legacy
    if legacy_va.is_file():
        return legacy_va
    return primary


def resolve_template_prior_checkpoint(checkpoint_dir: Path | None = None) -> Path:
    root = Path(checkpoint_dir) if checkpoint_dir is not None else (CHECKPOINTS / "template_prior")
    primary = root / TEMPLATE_PRIOR_NAME
    legacy = root / "last.pt"
    if primary.is_file():
        return primary
    if legacy.is_file():
        return legacy
    return primary


def role_from_checkpoint_dir(checkpoint_dir: Path) -> str:
    """checkpoints/backing → backing。stage1 は backing 扱い。"""
    name = Path(checkpoint_dir).name
    if name in {"stage1", "synthetic"}:
        return "backing"
    return name
