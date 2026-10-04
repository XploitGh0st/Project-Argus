"""Regenerate golden outputs and pre-pickled mock ASTs.

    python fixtures/make_golden.py

Writes
  fixtures/pkl/<name>.ast.pkl       - mock ``ast.pkl`` files (AST contract 1.0)
  fixtures/expected/<name>.cfg.json - golden CFG documents

The golden files are regression anchors; ``tests/test_extract.py`` ALSO asserts the
expected FSM structure by hand, so a wrong golden cannot silently pass.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

from argus_cfg import build_cfg  # noqa: E402
from ast_builders import ALL_BUILDERS  # noqa: E402


def main() -> None:
    (HERE / "pkl").mkdir(exist_ok=True)
    (HERE / "expected").mkdir(exist_ok=True)
    sys.setrecursionlimit(20000)
    for name, builder in ALL_BUILDERS.items():
        ast, top = builder()
        payload = {
            "schema_version": "1.0",
            "kind": "ast",
            "top_module": top,
            "source_files": [f"<stub:{name}>"],
            "ast": ast,
        }
        with (HERE / "pkl" / f"{name}.ast.pkl").open("wb") as fh:
            pickle.dump(payload, fh, protocol=4)
        doc = build_cfg(ast, top).to_dict()
        (HERE / "expected" / f"{name}.cfg.json").write_text(
            json.dumps(doc, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
