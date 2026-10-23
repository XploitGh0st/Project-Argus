"""Unit tests for argus_ml.labeling (WL-3 graph diffing)."""

from argus_ml.labeling import diff_graphs, validate_labeling


def test_clean_vs_itself_zero_positives():
    clean_graph = {
        "nodes": [
            {"id": "top.clk", "type": "signal", "termtype": "Input"},
            {"id": "top.rst", "type": "signal", "termtype": "Input"},
            {"id": "top.count", "type": "signal", "termtype": "Reg"},
            {"id": "c0", "type": "const", "label": "0"},
            {"id": "c1", "type": "const", "label": "1"},
            {"id": "op_add", "type": "operator", "label": "Plus"},
            {"id": "br0", "type": "branch", "label": "branch"},
        ],
        "edges": [
            {"src": "top.count", "dst": "op_add", "kind": "dataflow"},
            {"src": "c1", "dst": "op_add", "kind": "dataflow"},
            {"src": "op_add", "dst": "top.count", "kind": "dataflow"},
            {"src": "top.rst", "dst": "br0", "kind": "control_dep"},
            {"src": "br0", "dst": "top.count", "kind": "control_dep"},
        ],
    }

    labels = diff_graphs(clean_graph, clean_graph)
    assert all(lbl == 0 for lbl in labels.values()), "Clean vs itself must have 0 positives"
    assert sum(labels.values()) == 0


def test_tiny_diff_comparator_and_register():
    # Base clean counter
    clean_graph = {
        "nodes": [
            {"id": "top.clk", "type": "signal", "termtype": "Input"},
            {"id": "top.rst", "type": "signal", "termtype": "Input"},
            {"id": "top.count", "type": "signal", "termtype": "Reg"},
            {"id": "c1", "type": "const", "label": "1"},
            {"id": "op_add", "type": "operator", "label": "Plus"},
        ],
        "edges": [
            {"src": "top.count", "dst": "op_add", "kind": "dataflow"},
            {"src": "c1", "dst": "op_add", "kind": "dataflow"},
            {"src": "op_add", "dst": "top.count", "kind": "dataflow"},
        ],
    }

    # Infected: has extra trojan register and comparator
    infected_graph = {
        "nodes": [
            {"id": "top.clk", "type": "signal", "termtype": "Input"},
            {"id": "top.rst", "type": "signal", "termtype": "Input"},
            {"id": "top.count", "type": "signal", "termtype": "Reg"},
            {"id": "c1", "type": "const", "label": "1"},
            {"id": "op_add", "type": "operator", "label": "Plus"},
            # Trojan additions:
            {"id": "top.tj_reg", "type": "signal", "termtype": "Reg"},
            {"id": "c_tj", "type": "const", "label": "128"},
            {"id": "op_cmp", "type": "operator", "label": "Eq"},
        ],
        "edges": [
            {"src": "top.count", "dst": "op_add", "kind": "dataflow"},
            {"src": "c1", "dst": "op_add", "kind": "dataflow"},
            {"src": "op_add", "dst": "top.count", "kind": "dataflow"},
            # Trojan edges:
            {"src": "top.count", "dst": "op_cmp", "kind": "dataflow"},
            {"src": "c_tj", "dst": "op_cmp", "kind": "dataflow"},
            {"src": "op_cmp", "dst": "top.tj_reg", "kind": "control_dep"},
        ],
    }

    labels = diff_graphs(clean_graph, infected_graph)
    # The new nodes (tj_reg, c_tj, op_cmp) must be positive (1)
    assert labels["top.tj_reg"] == 1
    assert labels["op_cmp"] == 1
    assert labels["c_tj"] == 1

    # Base signals must be 0
    assert labels["top.clk"] == 0
    assert labels["top.rst"] == 0


def test_validate_labeling_metrics():
    clean_graph = {
        "nodes": [{"id": "top.a", "type": "signal", "termtype": "Wire"}],
        "edges": [],
    }
    infected_graph = {
        "nodes": [
            {"id": "top.a", "type": "signal", "termtype": "Wire"},
            {"id": "top.trojan_trigger", "type": "signal", "termtype": "Wire"},
        ],
        "edges": [],
    }

    metrics = validate_labeling(clean_graph, infected_graph)
    assert metrics["self_positives"] == 0
    assert metrics["total_nodes"] == 2
    assert metrics["total_positives"] == 1
    assert metrics["positive_fraction"] == 0.5
    assert metrics["flag_high_positives"] is True  # 50% > 10%
    assert metrics["agreement_with_new_signals"] == 1.0
