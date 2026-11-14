"""Atomic file writes: readers see the old file or the complete new one, never a partial one."""

from __future__ import annotations

import os
from pathlib import Path


def write_text(path: Path, text: str) -> None:
    """Atomically replace path with text (temporary sibling, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
