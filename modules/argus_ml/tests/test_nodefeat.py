"""Unit tests for argus_ml.nodefeat."""

import numpy as np
from argus_ml.nodefeat import FEATURE_COLUMNS, node_features


def test_node_features_schema_and_no_nan():
    # Mock clean digraph
    clean_graph = {
        "nodes": [
            {"id": "top.clk", "type": "signal", "termtype": "Input", "label": ""},
            {"id": "top.rst", "type": "signal", "termtype": "Input", "label": ""},
            {"id": "top.count", "type": "signal", "termtype": "Reg", "label": ""},
            {"id": "top.out", "type": "signal", "termtype": "Output", "label": ""},
            {"id": "c1", "type": "const", "termtype": "", "label": "8'h01"},
            {"id": "c_max", "type": "const", "termtype": "", "label": "8'hff"},
            {"id": "op_add", "type": "operator", "termtype": "", "label": "Plus"},
            {"id": "op_eq", "type": "operator", "termtype": "", "label": "Eq"},
            {"id": "br0", "type": "branch", "termtype": "", "label": "branch"},
            {"id": "s_idle", "type": "fsm_state", "termtype": "", "label": "IDLE"},
        ],
        "edges": [
            {"src": "top.count", "dst": "op_add", "kind": "dataflow"},
            {"src": "c1", "dst": "op_add", "kind": "dataflow"},
            {"src": "op_add", "dst": "top.count", "kind": "dataflow"},
            {"src": "top.count", "dst": "op_eq", "kind": "dataflow"},
            {"src": "c_max", "dst": "op_eq", "kind": "dataflow"},
            {"src": "op_eq", "dst": "br0", "kind": "control_dep"},
            {"src": "br0", "dst": "top.out", "kind": "dataflow"},
            {"src": "s_idle", "dst": "s_idle", "kind": "fsm_trans"},
        ],
    }

    X, cols, nids = node_features(clean_graph)

    assert X.shape[0] == 10
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert cols == list(FEATURE_COLUMNS)
    assert len(nids) == 10
    assert not np.isnan(X).any(), "Features array must have 0 NaNs"
    assert not np.isinf(X).any(), "Features array must have 0 Infs"
    assert X.dtype == np.float32


def test_empty_graph():
    X, cols, nids = node_features({"nodes": [], "edges": []})
    assert X.shape == (0, len(FEATURE_COLUMNS))
    assert cols == list(FEATURE_COLUMNS)
    assert len(nids) == 0
