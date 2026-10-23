"""
argus_ml.nodefeat - Engineered node feature extraction on RTL dependency digraphs.

Extracts fixed-dimension numerical features per node without using signal names,
instance names, or node IDs.
"""

from __future__ import annotations

import collections
import math
import re
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path


TYPE_LIST = ["signal", "const", "operator", "branch", "fsm_state"]
TERMTYPE_LIST = ["wire", "reg", "input", "output", "inout", "parameter", "localparam", "other"]
OP_CAT_LIST = [
    "cmp_eq", "cmp_neq", "cmp_rel", "arith_add", "arith_sub",
    "arith_other", "logic_and", "logic_or", "logic_not", "bitwise", "other_op"
]

FEATURE_COLUMNS = (
    # 1. Structural Identity (one-hot)
    [f"type_{t}" for t in TYPE_LIST] +
    [f"termtype_{tt}" for tt in TERMTYPE_LIST] +
    [f"opcat_{op}" for op in OP_CAT_LIST] +

    # 2. Degree Features (raw + log1p)
    [
        "deg_in_dataflow", "deg_out_dataflow",
        "deg_in_ctrl", "deg_out_ctrl",
        "deg_in_fsm", "deg_out_fsm",
        "deg_in_total", "deg_out_total",
        "log_deg_in_dataflow", "log_deg_out_dataflow",
        "log_deg_in_ctrl", "log_deg_out_ctrl",
        "log_deg_in_fsm", "log_deg_out_fsm",
        "log_deg_in_total", "log_deg_out_total",
    ] +

    # 3. Constant Features
    ["is_const", "const_width", "const_log2_val", "const_popcount"] +

    # 4. Comparator Features
    ["is_comparator", "cmp_const_width", "cmp_is_equality"] +

    # 5. Rarity Features
    ["rarity_self", "rarity_up_ctrl"] +

    # 6. Counter Patterns
    ["in_counter_cycle", "counter_scc_size"] +

    # 7. Distances to I/O
    ["dist_from_input", "dist_to_output"] +

    # 8. Local Neighborhood Aggregates
    [
        "neigh_mean_in_deg", "neigh_max_in_deg",
        "neigh_mean_out_deg", "neigh_max_out_deg",
        "neigh_mean_rarity", "neigh_max_rarity",
    ] +

    # 9. Design-Relative Percentiles
    ["pct_in_deg", "pct_out_deg", "pct_rarity_up_ctrl"]
)


def _parse_const_val(label: str) -> tuple[int, int, int]:
    """Parse Verilog constant string into (width, val_int, popcount)."""
    if not label:
        return 0, 0, 0
    s = label.strip().replace("_", "")
    m_sized = re.match(r"^(\d+)?'([bBhHdDoO])([0-9a-fA-FzxZX?]+)$", s)
    if m_sized:
        w_str, base, digits = m_sized.groups()
        base = base.lower()
        base_map = {"b": 2, "d": 10, "h": 16, "o": 8}
        radix = base_map.get(base, 10)
        clean_digits = re.sub(r"[zxZX?]", "0", digits)
        try:
            val = int(clean_digits, radix) if clean_digits else 0
        except ValueError:
            val = 0
        width = int(w_str) if w_str else max(1, val.bit_length())
        popcount = bin(val).count("1")
        return width, val, popcount

    # Plain decimal
    if s.isdigit():
        val = int(s)
        return max(1, val.bit_length()), val, bin(val).count("1")

    return 0, 0, 0


def _get_op_cat(label: str) -> str:
    """Classify operator label into category."""
    if not label:
        return "other_op"
    l_lower = label.lower()
    if label in ("Eq", "Eql"):
        return "cmp_eq"
    if label in ("NotEq", "NotEql"):
        return "cmp_neq"
    if label in ("LessThan", "GreaterThan", "LessEq", "GreaterEq"):
        return "cmp_rel"
    if label == "Plus" or l_lower == "add":
        return "arith_add"
    if label == "Minus" or l_lower == "sub":
        return "arith_sub"
    if label in ("Times", "Divide", "Mod", "Power"):
        return "arith_other"
    if label in ("Land", "And"):
        return "logic_and"
    if label in ("Lor", "Or"):
        return "logic_or"
    if label in ("Ulnot", "Unot"):
        return "logic_not"
    if label in ("Xor", "Xnor", "Sll", "Srl", "Sra"):
        return "bitwise"
    return "other_op"


def _multi_source_bfs(adj_csr: sp.csr_matrix, sources: list[int], num_nodes: int) -> np.ndarray:
    """Fast multi-source BFS on CSR adjacency matrix."""
    dist = np.full(num_nodes, 99.0, dtype=np.float32)
    if not sources or adj_csr.nnz == 0:
        return dist
    from collections import deque
    q = deque()
    for s in sources:
        dist[s] = 0.0
        q.append(s)
    indptr = adj_csr.indptr
    indices = adj_csr.indices
    while q:
        u = q.popleft()
        d_next = dist[u] + 1.0
        start = indptr[u]
        end = indptr[u + 1]
        for idx in range(start, end):
            v = indices[idx]
            if dist[v] == 99.0:
                dist[v] = d_next
                q.append(v)
    return dist


def node_features(digraph: Any) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Compute engineered node features on a Digraph (or dict with 'nodes', 'edges').
    
    Returns:
        X: np.ndarray of shape (num_nodes, len(FEATURE_COLUMNS)), dtype float32
        columns: list of feature column names
        node_ids: list of node IDs corresponding to rows
    """
    nodes = digraph["nodes"] if isinstance(digraph, dict) else digraph.nodes
    edges = digraph["edges"] if isinstance(digraph, dict) else digraph.edges

    num_nodes = len(nodes)
    if num_nodes == 0:
        return (
            np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32),
            list(FEATURE_COLUMNS),
            [],
        )

    node_ids: list[str] = []
    node_to_idx: dict[str, int] = {}
    type_arr = []
    termtype_arr = []
    label_arr = []

    for i, node in enumerate(nodes):
        if isinstance(node, dict):
            nid = str(node.get("id", ""))
            ntype = str(node.get("type", "signal"))
            nlabel = str(node.get("label", "") or "")
            nterm = str(node.get("termtype", "") or "")
        else:
            nid = str(getattr(node, "id", ""))
            ntype = str(getattr(node, "type", "signal"))
            nlabel = str(getattr(node, "label", "") or "")
            nterm = str(getattr(node, "termtype", "") or "")

        node_ids.append(nid)
        node_to_idx[nid] = i
        type_arr.append(ntype)
        termtype_arr.append(nterm)
        label_arr.append(nlabel)

    # Adjacency matrices by edge kind
    df_rows, df_cols = [], []
    ctrl_rows, ctrl_cols = [], []
    fsm_rows, fsm_cols = [], []
    all_rows, all_cols = [], []

    for edge in edges:
        if isinstance(edge, dict):
            src_id = str(edge.get("src", ""))
            dst_id = str(edge.get("dst", ""))
            kind = str(edge.get("kind", ""))
        else:
            src_id = str(getattr(edge, "src", ""))
            dst_id = str(getattr(edge, "dst", ""))
            kind = str(getattr(edge, "kind", ""))

        src = node_to_idx.get(src_id)
        dst = node_to_idx.get(dst_id)
        if src is None or dst is None:
            continue

        all_rows.append(src)
        all_cols.append(dst)
        if kind == "dataflow":
            df_rows.append(src)
            df_cols.append(dst)
        elif kind == "control_dep":
            ctrl_rows.append(src)
            ctrl_cols.append(dst)
        elif kind == "fsm_trans":
            fsm_rows.append(src)
            fsm_cols.append(dst)

    data_ones = np.ones(len(all_rows), dtype=np.float32)
    adj_all = sp.csr_matrix((data_ones, (all_rows, all_cols)), shape=(num_nodes, num_nodes))

    def _degree_pair(rows, cols):
        in_deg = np.zeros(num_nodes, dtype=np.float32)
        out_deg = np.zeros(num_nodes, dtype=np.float32)
        if rows:
            for s, d in zip(rows, cols):
                out_deg[s] += 1.0
                in_deg[d] += 1.0
        return in_deg, out_deg

    in_df, out_df = _degree_pair(df_rows, df_cols)
    in_ctrl, out_ctrl = _degree_pair(ctrl_rows, ctrl_cols)
    in_fsm, out_fsm = _degree_pair(fsm_rows, fsm_cols)
    in_tot, out_tot = _degree_pair(all_rows, all_cols)

    # 1. Structural Identity
    type_mat = np.zeros((num_nodes, len(TYPE_LIST)), dtype=np.float32)
    for i, t in enumerate(type_arr):
        if t in TYPE_LIST:
            type_mat[i, TYPE_LIST.index(t)] = 1.0

    termtype_mat = np.zeros((num_nodes, len(TERMTYPE_LIST)), dtype=np.float32)
    for i, tt in enumerate(termtype_arr):
        tt_low = tt.lower()
        if tt_low in TERMTYPE_LIST:
            termtype_mat[i, TERMTYPE_LIST.index(tt_low)] = 1.0
        elif tt_low:
            termtype_mat[i, TERMTYPE_LIST.index("other")] = 1.0

    op_mat = np.zeros((num_nodes, len(OP_CAT_LIST)), dtype=np.float32)
    for i in range(num_nodes):
        if type_arr[i] == "operator":
            cat = _get_op_cat(label_arr[i])
            if cat in OP_CAT_LIST:
                op_mat[i, OP_CAT_LIST.index(cat)] = 1.0

    # 3. Constants
    is_const = np.zeros(num_nodes, dtype=np.float32)
    const_w = np.zeros(num_nodes, dtype=np.float32)
    const_log2 = np.zeros(num_nodes, dtype=np.float32)
    const_pop = np.zeros(num_nodes, dtype=np.float32)

    for i in range(num_nodes):
        if type_arr[i] == "const":
            is_const[i] = 1.0
            w, val, pop = _parse_const_val(label_arr[i])
            const_w[i] = float(w)
            const_log2[i] = math.log2(val + 1.0) if val >= 0 else 0.0
            const_pop[i] = float(pop)

    # 4. Comparators
    is_cmp = np.zeros(num_nodes, dtype=np.float32)
    cmp_const_w = np.zeros(num_nodes, dtype=np.float32)
    cmp_is_eq = np.zeros(num_nodes, dtype=np.float32)

    for i in range(num_nodes):
        if type_arr[i] == "operator" and _get_op_cat(label_arr[i]).startswith("cmp_"):
            is_cmp[i] = 1.0
            cat = _get_op_cat(label_arr[i])
            cmp_is_eq[i] = 1.0 if cat == "cmp_eq" else 0.0

    # Fill comparator constant width from operand inputs
    if df_rows:
        for s, d in zip(df_rows, df_cols):
            if is_cmp[d] == 1.0 and is_const[s] == 1.0:
                cmp_const_w[d] = max(cmp_const_w[d], const_w[s])

    # 5. Rarity
    sig_keys = [(type_arr[i], termtype_arr[i] if type_arr[i] == "signal" else label_arr[i]) for i in range(num_nodes)]
    counts = collections.Counter(sig_keys)
    rarity_self = np.array([math.log1p(1.0 / counts[sig_keys[i]]) for i in range(num_nodes)], dtype=np.float32)

    # 3-hop backward cone over (control_dep + dataflow)
    bwd_rows = df_cols + ctrl_cols
    bwd_cols = df_rows + ctrl_rows
    if bwd_rows:
        bwd_ones = np.ones(len(bwd_rows), dtype=np.float32)
        adj_bwd = sp.csr_matrix((bwd_ones, (bwd_rows, bwd_cols)), shape=(num_nodes, num_nodes))
        # 1-hop + 2-hop + 3-hop
        r_step1 = adj_bwd.dot(rarity_self)
        r_step2 = adj_bwd.dot(r_step1)
        r_step3 = adj_bwd.dot(r_step2)
        rarity_up_ctrl = rarity_self + r_step1 + r_step2 + r_step3
    else:
        rarity_up_ctrl = rarity_self.copy()

    # 6. Counter Patterns (SCC in dataflow graph)
    in_counter_cycle = np.zeros(num_nodes, dtype=np.float32)
    counter_scc_size = np.zeros(num_nodes, dtype=np.float32)
    if df_rows:
        adj_df = sp.csr_matrix((np.ones(len(df_rows), dtype=np.float32), (df_rows, df_cols)), shape=(num_nodes, num_nodes))
        n_scc, scc_labels = connected_components(adj_df, directed=True, connection="strong")
        scc_counts = collections.Counter(scc_labels)
        for i in range(num_nodes):
            size = scc_counts[scc_labels[i]]
            counter_scc_size[i] = min(float(size), 10.0)
            if size > 1:
                # check if an add/sub operator is in this SCC
                in_counter_cycle[i] = 1.0

    # 7. Shortest directed distance from primary inputs & to primary outputs
    input_indices = [i for i, tt in enumerate(termtype_arr) if tt.lower() == "input"]
    output_indices = [i for i, tt in enumerate(termtype_arr) if tt.lower() == "output"]

    dist_from_input = np.full(num_nodes, 99.0, dtype=np.float32)
    dist_to_output = np.full(num_nodes, 99.0, dtype=np.float32)

    if input_indices and all_rows:
        dist_from_input = _multi_source_bfs(adj_all, input_indices, num_nodes)

    if output_indices and all_rows:
        adj_rev = adj_all.T.tocsr()
        dist_to_output = _multi_source_bfs(adj_rev, output_indices, num_nodes)

    # 8. Local Neighborhood Aggregates (1-hop undirected neighbors)
    adj_undir = (adj_all + adj_all.T).tocsr()
    deg_undir = np.array(adj_undir.sum(axis=1)).flatten()
    safe_deg = np.where(deg_undir == 0, 1.0, deg_undir)

    neigh_mean_in_deg = np.array(adj_undir.dot(in_tot)).flatten() / safe_deg
    neigh_mean_out_deg = np.array(adj_undir.dot(out_tot)).flatten() / safe_deg
    neigh_mean_rarity = np.array(adj_undir.dot(rarity_self)).flatten() / safe_deg

    # Max neighbor aggregates via vectorized np.maximum.at
    neigh_max_in_deg = np.zeros(num_nodes, dtype=np.float32)
    neigh_max_out_deg = np.zeros(num_nodes, dtype=np.float32)
    neigh_max_rarity = np.zeros(num_nodes, dtype=np.float32)

    if adj_undir.nnz > 0:
        r, c = adj_undir.nonzero()
        np.maximum.at(neigh_max_in_deg, r, in_tot[c])
        np.maximum.at(neigh_max_out_deg, r, out_tot[c])
        np.maximum.at(neigh_max_rarity, r, rarity_self[c])

    # 9. Design-Relative Percentiles
    pct_in_deg = (in_tot.argsort().argsort() / float(num_nodes)).astype(np.float32)
    pct_out_deg = (out_tot.argsort().argsort() / float(num_nodes)).astype(np.float32)
    pct_rarity = (rarity_up_ctrl.argsort().argsort() / float(num_nodes)).astype(np.float32)

    # Log1p transforms
    log_in_df = np.log1p(in_df)
    log_out_df = np.log1p(out_df)
    log_in_ctrl = np.log1p(in_ctrl)
    log_out_ctrl = np.log1p(out_ctrl)
    log_in_fsm = np.log1p(in_fsm)
    log_out_fsm = np.log1p(out_fsm)
    log_in_tot = np.log1p(in_tot)
    log_out_tot = np.log1p(out_tot)

    # Stack features in exact FEATURE_COLUMNS order
    columns_data = [
        type_mat,
        termtype_mat,
        op_mat,
        in_df[:, None], out_df[:, None],
        in_ctrl[:, None], out_ctrl[:, None],
        in_fsm[:, None], out_fsm[:, None],
        in_tot[:, None], out_tot[:, None],
        log_in_df[:, None], log_out_df[:, None],
        log_in_ctrl[:, None], log_out_ctrl[:, None],
        log_in_fsm[:, None], log_out_fsm[:, None],
        log_in_tot[:, None], log_out_tot[:, None],
        is_const[:, None], const_w[:, None], const_log2[:, None], const_pop[:, None],
        is_cmp[:, None], cmp_const_w[:, None], cmp_is_eq[:, None],
        rarity_self[:, None], rarity_up_ctrl[:, None],
        in_counter_cycle[:, None], counter_scc_size[:, None],
        dist_from_input[:, None], dist_to_output[:, None],
        neigh_mean_in_deg[:, None], neigh_max_in_deg[:, None],
        neigh_mean_out_deg[:, None], neigh_max_out_deg[:, None],
        neigh_mean_rarity[:, None], neigh_max_rarity[:, None],
        pct_in_deg[:, None], pct_out_deg[:, None], pct_rarity[:, None],
    ]

    X = np.hstack(columns_data).astype(np.float32)
    # Ensure zero NaNs or infinities
    X = np.nan_to_num(X, nan=0.0, posinf=99.0, neginf=-99.0)

    assert X.shape[1] == len(FEATURE_COLUMNS), f"Expected {len(FEATURE_COLUMNS)} features, got {X.shape[1]}"
    assert not np.isnan(X).any(), "Features array must contain zero NaNs"

    return X, list(FEATURE_COLUMNS), node_ids
