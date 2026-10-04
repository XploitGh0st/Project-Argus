"""Command line: ``python -m argus_dfg out/ast.pkl --out out/``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from argus_dfg import __version__, build_dfg


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="argus-dfg", description=__doc__)
    ap.add_argument("ast", help="path to ast.pkl")
    ap.add_argument("--top", default=None, help="top module (default: from the pickle)")
    ap.add_argument("--out", default="out", help="output directory (default: out)")
    ap.add_argument(
        "--resolve-constants", action="store_true", help="fold parameters/constants first"
    )
    ap.add_argument("--version", action="version", version=f"argus-dfg {__version__}")
    args = ap.parse_args(argv)

    try:
        result = build_dfg(args.ast, args.top, resolve_constants=args.resolve_constants)
    except (ValueError, TypeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    path = result.save(args.out)
    print(f"top module : {result.top_module}")
    print(f"signals    : {len(result.signals())}")
    print(f"nodes      : {len(result.nodes)}")
    print(f"edges      : {len(result.edges)}")
    print(f"written    : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
