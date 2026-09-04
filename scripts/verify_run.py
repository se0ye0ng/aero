"""Re-execute a run from its manifest and diff the metrics."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tolerance", type=float, default=0.002)
    args = ap.parse_args()
    raise SystemExit(f"TODO: verify {args.run} within {args.tolerance}")


if __name__ == "__main__":
    main()
