"""Load CFG and DFG JSON files."""

import json
from pathlib import Path
from typing import Any

CFG_KIND = "cfg"
DFG_KIND = "dfg"


def load_json(path: str | Path, expected_kind: str) -> dict[str, Any]:
    with Path(path).open() as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")

    kind = data.get("kind")
    if kind != expected_kind:
        raise ValueError(f"{path}: expected kind {expected_kind!r}, got {kind!r}")

    return data
