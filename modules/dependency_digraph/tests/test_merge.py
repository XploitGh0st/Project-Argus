from pathlib import Path

from argus_digraph.inputs import CFG_KIND, DFG_KIND, load_json
from argus_digraph.merge import merge_graphs

HERE = Path(__file__).parent
EXPECTED_DIR = HERE.parent / "fixtures" / "expected"

def test_merge_graphs():
    cfg_doc = load_json(EXPECTED_DIR / "traffic_light.cfg.json", CFG_KIND)
    dfg_doc = load_json(EXPECTED_DIR / "traffic_light.dfg.json", DFG_KIND)

    result = merge_graphs(cfg_doc, dfg_doc)
    doc = result.to_dict()

    assert doc["top_module"] == "traffic"
    assert doc["kind"] == "digraph"
    
    # Check that both dataflow and control_dep edges exist
    edge_kinds = {e["kind"] for e in doc["edges"]}
    assert "dataflow" in edge_kinds
    assert "control_dep" in edge_kinds
    assert "fsm_trans" in edge_kinds
    
    # Check that fsm_state nodes are added
    node_types = {n["type"] for n in doc["nodes"]}
    assert "fsm_state" in node_types
    assert "signal" in node_types
    assert "operator" in node_types
