"""Command line entry point.

Everything is driven by Hydra configs. No experiment is defined in a script, so the run grid
in ``configs/experiment/`` is the single source of truth for what was run.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aero", description="AERO experiment driver")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a training experiment")
    p_run.add_argument("overrides", nargs="*", help="hydra overrides, e.g. experiment=e2_...")

    sub.add_parser("generate", help="run the generator over the configured source split")
    sub.add_parser("audit-labels", help="run the label-transfer audit (F4)")
    sub.add_parser("rfs", help="compute the RFS report for a generated set")

    p_dep = sub.add_parser("deploy", help="export, quantise and benchmark")
    p_dep.add_argument("--export", default="onnx")
    p_dep.add_argument("--quantize", default=None)
    p_dep.add_argument("--bench", action="store_true")

    args = parser.parse_args(argv)
    raise SystemExit(f"TODO: dispatch {args.command!r} - see docs/experiment_protocol.md")


if __name__ == "__main__":
    main()
