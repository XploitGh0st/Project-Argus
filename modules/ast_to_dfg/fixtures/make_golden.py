"""Script to regenerate test fixtures and expected JSON outputs.

Run via: python modules/ast_to_dfg/fixtures/make_golden.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

# Add src to path to import argus_dfg
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from argus_dfg.extract import build_dfg
from argus_dfg.inputs import AST_KIND, AST_SCHEMA
from ast_builders import ALL_BUILDERS

HERE = Path(__file__).parent
EXPECTED_DIR = HERE / "expected"

def make_golden() -> None:
    EXPECTED_DIR.mkdir(exist_ok=True)
    
    for name, builder in ALL_BUILDERS.items():
        print(f"Building golden for {name}...")
        ast, top = builder()
        
        # 1. Dump AST pickle (what parser would produce)
        pkl_path = EXPECTED_DIR / f"{name}.ast.pkl"
        payload = {
            "kind": AST_KIND,
            "schema_version": AST_SCHEMA,
            "top_module": top,
            "ast": ast,
        }
        with pkl_path.open("wb") as fh:
            pickle.dump(payload, fh)
            
        # 2. Extract DFG and dump JSON
        result = build_dfg(ast, top)
        json_path = EXPECTED_DIR / f"{name}.dfg.json"
        
        with json_path.open("w") as fh:
            json.dump(result.to_dict(), fh, indent=2)
            
    print("Done!")

if __name__ == "__main__":
    make_golden()
