import json
from pathlib import Path

import pytest

from argus_dfg.extract import build_dfg

HERE = Path(__file__).parent
EXPECTED_DIR = HERE.parent / "fixtures" / "expected"

@pytest.mark.parametrize(
    "design",
    [
        "traffic_light",
        "two_fsms",
        "no_fsm",
    ],
)
def test_build_dfg_matches_golden(design: str) -> None:
    pkl_path = EXPECTED_DIR / f"{design}.ast.pkl"
    json_path = EXPECTED_DIR / f"{design}.dfg.json"

    # Extract DFG from pickle
    result = build_dfg(pkl_path)
    output = result.to_dict()

    # Compare with golden
    with json_path.open() as fh:
        expected = json.load(fh)

    assert output["top_module"] == expected["top_module"]
    assert output["kind"] == expected["kind"]
    assert output["schema_version"] == expected["schema_version"]
    
    # Sort nodes and edges for comparison since dictionary iteration order might differ
    def sort_nodes(nodes):
        return sorted(nodes, key=lambda n: n["id"])
        
    def sort_edges(edges):
        return sorted(edges, key=lambda e: (e["src"], e["dst"], e["label"]))

    assert sort_nodes(output["nodes"]) == sort_nodes(expected["nodes"])
    assert sort_edges(output["edges"]) == sort_edges(expected["edges"])


def test_build_dfg_requires_top_module_for_in_memory_ast():
    class Source:
        pass
        
    mock_source = Source()
    
    with pytest.raises(ValueError, match="top_module is required"):
        build_dfg(mock_source)
