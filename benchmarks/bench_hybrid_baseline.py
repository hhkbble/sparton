"""Compatibility wrapper for the old hybrid SPLADE dev-shape baseline."""

from __future__ import annotations

import sys

from bench_sparton_baseline import main


if __name__ == "__main__":
    main(
        [
            "--batch-sizes",
            "32",
            "--seq-lens",
            "128",
            "--dim",
            "768",
            "--vocab",
            "30522",
            "--dtype",
            "fp16",
            "--naive-policy",
            "off",
        ]
        + sys.argv[1:]
    )
