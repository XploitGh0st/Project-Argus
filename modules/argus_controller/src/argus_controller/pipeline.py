"""Argus End-to-End Pipeline Controller.

This module orchestrates the full sequence by importing and executing the individual
decoupled feature modules.

Pipeline Sequence:
1. `argus_ast`: Parse Verilog into AST.
2. `argus_cfg` & `argus_dfg`: Extract CFG and DFG from the AST.
3. `argus_digraph`: Merge CFG and DFG into a Dependency Digraph.
4. `argus_ml`: Convert Digraph into PyTorch Geometric Data and run inference/training.
"""

from typing import Any
import sys
import argparse

def run_pipeline(verilog_file: str, top_module: str, out_dir: str) -> None:
    # 1. Parse AST
    from argus_ast.parser import parse_file
    ast_result = parse_file(verilog_file, top_module)
    ast_result.save(out_dir)

    # 2. Extract CFG
    from argus_cfg.extract import build_cfg
    cfg_result = build_cfg(ast_result.ast, ast_result.top_module)
    cfg_result.save(out_dir)
    
    # 3. Extract DFG
    from argus_dfg.extract import build_dfg
    dfg_result = build_dfg(ast_result.ast, ast_result.top_module)
    dfg_result.save(out_dir)
    
    # 4. Merge Digraph
    from argus_digraph.merge import merge_graphs
    digraph_result = merge_graphs(cfg_result.to_dict(), dfg_result.to_dict())
    digraph_result.save(out_dir)
    
    # 5. ML Inference (Pseudocode, since we'd pass it to dataset/loader)
    print(f"Pipeline completed successfully. Artifacts saved to {out_dir}")
    print(f"Digraph Nodes: {len(digraph_result.nodes)}")
    print(f"Digraph Edges: {len(digraph_result.edges)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("verilog", help="Path to Verilog design")
    ap.add_argument("--top", required=True, help="Top module")
    ap.add_argument("--out", default="out", help="Output directory")
    args = ap.parse_args()
    run_pipeline(args.verilog, args.top, args.out)
