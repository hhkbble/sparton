"""Compatibility wrapper for the M5 moderate-shape naive baseline."""

from __future__ import annotations

import sys

from bench_sparton_baseline import main


if __name__ == "__main__":
    main(
        [
            "--batch-sizes",
            "4",
            "--seq-lens",
            "64",
            "--dim",
            "64",
            "--vocab",
            "4096",
            "--dtype",
            "fp16",
            "--naive-policy",
            "on",
        ]
        + sys.argv[1:]
    )
