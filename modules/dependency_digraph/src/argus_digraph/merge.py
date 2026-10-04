"""Merge CFG and DFG into a unified dependency digraph.

Logic:
1. Include all DFG nodes and edges (kind="dataflow").
2. Read CFG FSMs to identify FSM state registers.
3. For each FSM register `f`, if it's used in the COND of a DFG Branch node,
   and a signal `s` is assigned under that branch (dst of the branch in DFG),
   add a control dependency edge `f -> s` (kind="control_dep").
4. Add CFG states as nodes (type="fsm_state") and transitions as edges (kind="fsm_trans").
5. Add edge `f -> state` (kind="control_dep") from FSM register to its states.
"""

from typing import Any

from argus_digraph.result import DigraphResult, Edge, Node


def merge_graphs(cfg_doc: dict[str, Any], dfg_doc: dict[str, Any]) -> DigraphResult:
    top_module = dfg_doc["top_module"]
    nodes = {}
    edges = []

    # 1. Load DFG nodes and edges
    for n in dfg_doc["nodes"]:
        nodes[n["id"]] = Node(
            id=n["id"],
            type=n["type"],
            label=n["label"],
            termtype=tuple(n.get("termtype", []))
        )

    dfg_edges = dfg_doc["edges"]
    for e in dfg_edges:
        edges.append(Edge(
            src=e["src"],
            dst=e["dst"],
            label=e["label"],
            kind="dataflow"
        ))

    # Build maps to find assignments under branches
    # DFG branches: branches are nodes with type="branch". 
    # Edges into branch are "COND", "TRUE", "FALSE".
    # We want to find if an FSM reg `f` influences a branch COND.
    # We can trace back from branch COND to see if it reads `f`.
    # For simplicity, if `f` is directly connected to COND, or via operators.
    
    # 2. Extract FSM registers from CFG
    fsm_regs = set()
    fsms = cfg_doc.get("fsms", [])
    for fsm in fsms:
        fsm_regs.add(fsm["signal"])
        
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
            # Note: guard in CFG is a string expression, not a direct node ID. 
            # We won't try to parse it back to DFG nodes here, but a rigorous implementation would.

    # 3. Add Control Dependency Edges
    # Find branches influenced by FSM regs
    # A simple approach: we find paths from fsm_regs to branch CONDs.
    
    # Build a graph of DFG to trace dependencies
    from collections import defaultdict
    adj = defaultdict(list)
    for e in dfg_edges:
        adj[e["dst"]].append((e["src"], e["label"]))

    # Find assignments that depend on a branch
    # An assignment is an edge with label="assign" or similar. But in our DFG, 
    # data flows from Branch -> signal.
    for e in dfg_edges:
        if e["label"] == "assign":
            # The src of an assign edge might be a branch or an operator
            src = e["src"]
            dst = e["dst"]
            
            # Walk backwards from src to see if it passes through a branch 
            # and that branch's COND depends on an fsm_reg.
            
            # Find branches in the backward flow
            q = [src]
            visited = set()
            influenced_by = set()
            while q:
                curr = q.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                
                curr_node = nodes.get(curr)
                if curr_node and curr_node.type == "branch":
                    # Check what feeds the COND
                    cond_srcs = [s for s, lbl in adj[curr] if lbl == "COND"]
                    # trace COND to fsm regs
                    cond_q = list(cond_srcs)
                    cond_visited = set()
                    while cond_q:
                        c = cond_q.pop(0)
                        if c in cond_visited:
                            continue
                        cond_visited.add(c)
                        if c in fsm_regs:
                            influenced_by.add(c)
                        for s, _ in adj[c]:
                            cond_q.append(s)
                            
                for s, _ in adj[curr]:
                    q.append(s)
            
            for f in influenced_by:
                edges.append(Edge(src=f, dst=dst, label="control_dep", kind="control_dep"))

    return DigraphResult(
        top_module=top_module,
        nodes=tuple(nodes.values()),
        edges=tuple(edges)
    )
