"""合成ギター MIDI 生成（Stage 1 用テストデータ）。"""

from __future__ import annotations

from typing import Any

__all__ = ["generate_dataset", "main"]


def __getattr__(name: str) -> Any:
    if name in ("generate_dataset", "main"):
        from .generate import generate_dataset, main

        return generate_dataset if name == "generate_dataset" else main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
