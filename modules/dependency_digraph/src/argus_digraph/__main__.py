"""Command line: ``python -m argus_digraph cfg.json dfg.json --out out/``."""

import argparse
import sys
from collections.abc import Sequence

from argus_digraph import __version__, merge_graphs
from argus_digraph.inputs import CFG_KIND, DFG_KIND, load_json


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="argus-digraph", description=__doc__)
    ap.add_argument("cfg", help="path to cfg.json")
    ap.add_argument("dfg", help="path to dfg.json")
    ap.add_argument("--out", default="out", help="output directory (default: out)")
    ap.add_argument("--version", action="version", version=f"argus-digraph {__version__}")
    args = ap.parse_args(argv)

    try:
        cfg_doc = load_json(args.cfg, CFG_KIND)
        dfg_doc = load_json(args.dfg, DFG_KIND)
    except (ValueError, OSError) as exc:
        print(f"error loading inputs: {exc}", file=sys.stderr)
        return 2

    if cfg_doc["top_module"] != dfg_doc["top_module"]:
        print(f"error: top_module mismatch (cfg: {cfg_doc['top_module']}, dfg: {dfg_doc['top_module']})", file=sys.stderr)
        return 2

    try:
        result = merge_graphs(cfg_doc, dfg_doc)
    except Exception as exc:
        print(f"error merging graphs: {exc}", file=sys.stderr)
        return 2

    path = result.save(args.out)
    print(f"top module : {result.top_module}")
    print(f"nodes      : {len(result.nodes)}")
    print(f"edges      : {len(result.edges)}")
    print(f"written    : {path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
