"""Run the offline-first ingestion and hybrid-retrieval demonstration."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast


def main() -> int:
    """Load and run the shared Phase 5 example."""
    examples_directory = Path(__file__).resolve().parent / "examples"
    sys.path.insert(0, str(examples_directory))
    try:
        namespace: dict[str, Any] = runpy.run_path(
            str(examples_directory / "_phase5.py"),
            run_name="phase5_demo",
        )
    finally:
        sys.path.remove(str(examples_directory))
    entrypoint = cast(Callable[[], int], namespace["main"])
    return entrypoint()


if __name__ == "__main__":
    raise SystemExit(main())
