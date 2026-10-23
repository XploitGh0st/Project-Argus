#!/usr/bin/env python3
"""
scripts/scan.py - Hardware Trojan Scanner CLI.

Usage:
  python scripts/scan.py --target <folder-or-file> [--top <top_module>] [--format text|json] [--threshold float]

Runs:
  normalize -> parse -> cfg/dfg -> merge -> node features -> predict -> prints suspicious nodes.
  Exit 0: clean
  Exit 1: trojan-suspect (any node above threshold)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modules" / "verilog_ast_parser" / "src"))
sys.path.insert(0, str(ROOT / "modules" / "ast_to_cfg" / "src"))
sys.path.insert(0, str(ROOT / "modules" / "ast_to_dfg" / "src"))
sys.path.insert(0, str(ROOT / "modules" / "dependency_digraph" / "src"))
sys.path.insert(0, str(ROOT / "modules" / "argus_ml" / "src"))

TB_RE = re.compile(r"(^test|^tb|_tb$|_test$)", re.IGNORECASE)
DEFAULT_MODEL_PATH = ROOT / "models" / "best" / "model.joblib"
DEFAULT_META_PATH = ROOT / "models" / "best" / "metadata.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Argus RTL Hardware Trojan Scanner CLI")
    parser.add_argument("--target", required=True, help="Path to RTL Verilog file or directory")
    parser.add_argument("--top", default=None, help="Top module name (optional; autodetected if omitted)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format (text or json)")
    parser.add_argument("--threshold", type=float, default=None, help="Suspicion probability threshold")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to trained model.joblib")
    parser.add_argument("--metadata", default=str(DEFAULT_META_PATH), help="Path to metadata.json")
    return parser.parse_args()


def load_target_files(target_path: Path) -> dict[str, str]:
    if not target_path.exists():
        raise FileNotFoundError(f"Target path does not exist: {target_path}")

    text_by_file = {}
    if target_path.is_file():
        text_by_file[str(target_path)] = target_path.read_text(encoding="utf-8", errors="replace")
    else:
        for f in sorted(target_path.rglob("*.v")):
            if not TB_RE.search(f.stem):
                text_by_file[str(f)] = f.read_text(encoding="utf-8", errors="replace")

    if not text_by_file:
        raise ValueError(f"No non-testbench Verilog files found in {target_path}")
    return text_by_file


def main():
    args = parse_args()
    target_path = Path(args.target).resolve()
    model_path = Path(args.model).resolve()
    meta_path = Path(args.metadata).resolve()

    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}. Run scripts/train_best_model.py first.", file=sys.stderr)
        sys.exit(2)

    # Load model and metadata
    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    threshold = args.threshold if args.threshold is not None else metadata.get("decision_threshold", 0.05)

    # 1. Apply pyverilog patches
    from argus_ast.patch_pyverilog import apply_pyverilog_patches
    apply_pyverilog_patches()

    # 2. Preprocess & Normalize
    from argus_ast.preprocess import normalize_verilog, find_top_module
    from argus_ast import parse_verilog_text
    from argus_cfg import build_cfg
    from argus_dfg import build_dfg
    from argus_digraph.merge import merge_graphs
    from argus_ml.nodefeat import node_features

    files = load_target_files(target_path)
    normalized_text, norm_log = normalize_verilog(files)

    # Top module
    top_mod = args.top
    if not top_mod:
        top_mod = find_top_module(normalized_text)

    # 3. Parse AST
    ast_artifact = parse_verilog_text(normalized_text)
    pyverilog_ast = getattr(ast_artifact, "ast", ast_artifact)

    # 4. CFG + DFG -> Digraph
    cfg = build_cfg(pyverilog_ast, top_mod)
    dfg = build_dfg(pyverilog_ast, top_mod)
    merged_g = merge_graphs(cfg.to_dict(), dfg.to_dict())
    digraph = merged_g.to_dict()

    # 5. Node Features
    X, feature_cols, node_ids = node_features(digraph)

    if len(node_ids) == 0:
        if args.format == "json":
            print(json.dumps({"status": "clean", "total_nodes": 0, "flagged": []}))
        else:
            print("Scan complete: 0 nodes extracted (empty design).")
        sys.exit(0)

    # 6. Predict Probabilities
    probs = model.predict_proba(X)[:, 1]

    # Map node info
    node_info = {}
    nodes_list = digraph.get("nodes", []) if isinstance(digraph, dict) else digraph.nodes
    for n in nodes_list:
        nid = n.get("id") if isinstance(n, dict) else getattr(n, "id")
        ntype = n.get("type") if isinstance(n, dict) else getattr(n, "type")
        nlabel = n.get("label") if isinstance(n, dict) else getattr(n, "label")
        nterm = n.get("termtype") if isinstance(n, dict) else getattr(n, "termtype")
        node_info[nid] = {"type": ntype, "label": nlabel or "", "termtype": nterm or ""}

    # 7. Collect and sort suspicious nodes
    flagged = []
    for i, nid in enumerate(node_ids):
        p = float(probs[i])
        if p >= threshold:
            info = node_info.get(nid, {})
            flagged.append({
                "id": nid,
                "type": info.get("type", "unknown"),
                "label": info.get("label", ""),
                "termtype": info.get("termtype", ""),
                "probability": round(p, 4),
            })

    # Sort descending by suspicion probability
    flagged.sort(key=lambda x: x["probability"], reverse=True)
    for rank, item in enumerate(flagged, start=1):
        item["rank"] = rank

    total_nodes = len(node_ids)
    flagged_count = len(flagged)
    is_suspect = flagged_count > 0

    # 8. Output
    if args.format == "json":
        res = {
            "status": "trojan_suspect" if is_suspect else "clean",
            "target": str(target_path),
            "top_module": top_mod,
            "threshold": threshold,
            "total_nodes": total_nodes,
            "flagged_nodes_count": flagged_count,
            "suspicion_fraction": round(flagged_count / total_nodes, 4) if total_nodes > 0 else 0.0,
            "flagged_nodes": flagged,
        }
        print(json.dumps(res, indent=2))
    else:
        print("================================================================")
        print("ARGUS HARDWARE TROJAN SCANNER REPORT")
        print("================================================================")
        print(f"Target:       {target_path}")
        print(f"Top Module:   {top_mod}")
        print(f"Threshold:    {threshold:.4f}")
        print(f"Total Nodes:  {total_nodes}")
        print(f"Flagged Nodes: {flagged_count} ({flagged_count/total_nodes*100:.2f}%)")
        print("----------------------------------------------------------------")
        if is_suspect:
            print("STATUS: [!] TROJAN SUSPECT IDENTIFIED")
            print("Top Suspicious Nodes:")
            print(f"  {'Rank':4s} | {'Probability':11s} | {'Type':10s} | {'Identifier':35s} | {'Label'}")
            print(f"  {'-'*4} | {'-'*11} | {'-'*10} | {'-'*35} | {'-'*15}")
            for item in flagged[:25]:
                print(f"  #{item['rank']:<3d} | {item['probability']:.4f}      | {item['type']:10s} | {item['id']:35s} | {item['label']}")
            if flagged_count > 25:
                print(f"  ... and {flagged_count - 25} more flagged nodes.")
        else:
            print("STATUS: [OK] CLEAN (No suspicious nodes found)")
        print("================================================================")

    sys.exit(1 if is_suspect else 0)


if __name__ == "__main__":
    main()
