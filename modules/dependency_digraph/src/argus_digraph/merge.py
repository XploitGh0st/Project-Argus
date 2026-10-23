"""Merge CFG and DFG into a unified dependency digraph.

Logic:
1. Include all DFG nodes and edges (kind="dataflow").
2. Add CFG FSM states and transitions.
3. Generic control-dependency: for every assign edge, walk back through
   same-cycle (non-sequential) edges; for every branch found, trace its
   COND back to root signal/const nodes and add control_dep root -> assigned
   signal.
"""

from collections import defaultdict
from typing import Any

from argus_digraph.result import DigraphResult, Edge, Node


def merge_graphs(cfg_doc: dict[str, Any], dfg_doc: dict[str, Any]) -> DigraphResult:
    top_module = dfg_doc["top_module"]
    nodes: dict[str, Node] = {}
    edges: list[Edge] = []

    # 1. Load DFG nodes and edges
    for n in dfg_doc["nodes"]:
        nodes[n["id"]] = Node(
            id=n["id"],
            type=n["type"],
            label=n["label"],
            termtype=tuple(n.get("termtype", []))
        )

    dfg_edges = dfg_doc["edges"]
    sequential_set: set[tuple[str, str]] = set()
    for e in dfg_edges:
        edges.append(Edge(
            src=e["src"],
            dst=e["dst"],
            label=e["label"],
            kind="dataflow"
        ))
        if e.get("sequential", False):
            sequential_set.add((e["src"], e["dst"]))

    # Backward adjacency (dst -> list of (src, label, sequential))
    adj: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
    for e in dfg_edges:
        adj[e["dst"]].append((e["src"], e["label"], e.get("sequential", False)))

    # 2. FSM states and transitions from CFG
    fsms = cfg_doc.get("fsms", [])
    for fsm in fsms:
        for state in fsm["states"]:
            state_id = f"state_{fsm['signal']}_{state}"
            nodes[state_id] = Node(id=state_id, type="fsm_state", label=str(state))
            edges.append(Edge(src=fsm["signal"], dst=state_id, label="has_state", kind="control_dep"))
        for t in fsm["transitions"]:
            src_id = f"state_{fsm['signal']}_{t['src']}"
            dst_id = f"state_{fsm['signal']}_{t['dst']}"
            guard = t.get("condition")
            label = f"to_{t['dst']}" if not guard else f"[{guard}] to_{t['dst']}"
            edges.append(Edge(src=src_id, dst=dst_id, label=label, kind="fsm_trans"))

    # 3. Generic control dependency
    # For every assign edge, walk backward through same-cycle edges to find
    # branches.  For each branch, trace COND inputs back to root nodes
    # (signal/const with no further non-sequential inputs) and add
    # control_dep from each root to the assigned signal.
    ctrl_dep_set: set[tuple[str, str]] = set()

    for e in dfg_edges:
        if e["label"] != "assign":
            continue
        assign_src = e["src"]
        assigned_signal = e["dst"]

        # Walk backward from assign_src, never crossing a sequential edge
        branches_found: set[str] = set()
        q = [assign_src]
        visited: set[str] = set()
        while q:
            curr = q.pop()
            if curr in visited:
                continue
            visited.add(curr)
            n = nodes.get(curr)
            if n and n.type == "branch":
                branches_found.add(curr)
            for src_id, lbl, seq in adj.get(curr, []):
                if not seq:
                    q.append(src_id)

        # For each branch, trace COND to root signal/const nodes
        for br_id in branches_found:
            cond_srcs = [s for s, lbl, _ in adj.get(br_id, []) if lbl == "COND"]
            cond_q = list(cond_srcs)
            cond_visited: set[str] = set()
            roots: set[str] = set()
            while cond_q:
                c = cond_q.pop()
                if c in cond_visited:
                    continue
                cond_visited.add(c)
                n = nodes.get(c)
                # Walk backward through non-sequential edges
                preds = [(s, lbl, seq) for s, lbl, seq in adj.get(c, []) if not seq and lbl != "assign"]
                assign_preds = [(s, lbl, seq) for s, lbl, seq in adj.get(c, []) if not seq and lbl == "assign"]
                # A node is a root if it has no non-assign non-sequential predecessors
                # and either: has no predecessors at all, or is a signal/const
                all_nonsec = preds + assign_preds
                if not all_nonsec:
                    roots.add(c)
                elif n and n.type in ("signal", "const") and not assign_preds:
                    roots.add(c)
                elif n and n.type in ("signal", "const") and assign_preds:
                    # Signal with incoming assign: trace through it
                    for s, _, _ in assign_preds:
                        cond_q.append(s)
                # Continue tracing through operators/branches
                for s, _, _ in preds:
                    cond_q.append(s)

            for root in roots:
                pair = (root, assigned_signal)
                if pair not in ctrl_dep_set and root != assigned_signal:
                    ctrl_dep_set.add(pair)
                    edges.append(Edge(src=root, dst=assigned_signal, label="control_dep", kind="control_dep"))

    return DigraphResult(
        top_module=top_module,
        nodes=tuple(nodes.values()),
        edges=tuple(edges)
    )
