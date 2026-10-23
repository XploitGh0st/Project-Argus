"""
argus_ml.labeling - Weisfeiler-Lehman graph diffing for Trojan node label generation.

Computes 3-round Weisfeiler-Lehman subtree hashes on clean and infected digraphs.
A node in the infected graph is labeled 1 (Trojan) if:
1. Its hash count exceeds the clean graph's count for that hash (extra structural node), OR
2. It is a signal whose hierarchical name does not exist in the clean graph.
Everything else is labeled 0 (clean).
"""

from __future__ import annotations

import collections
import hashlib
from typing import Any, Sequence


def _get_node_fields(node: Any) -> tuple[str, str, str, str]:
    """Extract (id, type, label, termtype) from node dict or object."""
    if isinstance(node, dict):
        nid = str(node.get("id", ""))
        ntype = str(node.get("type", ""))
        nlabel = str(node.get("label", "") or "")
        ntermtype = str(node.get("termtype", "") or "")
    else:
        nid = str(getattr(node, "id", ""))
        ntype = str(getattr(node, "type", ""))
        nlabel = str(getattr(node, "label", "") or "")
        ntermtype = str(getattr(node, "termtype", "") or "")
    return nid, ntype, nlabel, ntermtype


def _get_edge_fields(edge: Any) -> tuple[str, str, str]:
    """Extract (src, dst, kind) from edge dict or object."""
    if isinstance(edge, dict):
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        kind = str(edge.get("kind", ""))
    else:
        src = str(getattr(edge, "src", ""))
        dst = str(getattr(edge, "dst", ""))
        kind = str(getattr(edge, "kind", ""))
    return src, dst, kind


def compute_wl_hashes(
    nodes: Sequence[Any],
    edges: Sequence[Any],
    rounds: int = 3,
    shared_dict: dict[Any, int] | None = None,
) -> list[int]:
    """
    Compute Weisfeiler-Lehman subtree hashes for graph nodes.
    
    Initial label: (type, operator label or termtype, short signal name).
    Edge kinds are included in neighbor multisets.
    Node IDs are NOT used as features/labels.
    """
    if shared_dict is None:
        shared_dict = {}

    n = len(nodes)
    if n == 0:
        return []

    node_to_idx: dict[str, int] = {}
    init_labels: list[int] = []

    for idx, node in enumerate(nodes):
        nid, ntype, nlabel, ntermtype = _get_node_fields(node)
        node_to_idx[nid] = idx
        
        # Initial label = (type, operator label or termtype, short signal name for signal nodes)
        op_or_tt = nlabel if ntype == "operator" else ntermtype
        short_name = nid.split(".")[-1] if ntype == "signal" else ""
        raw_key = (ntype, op_or_tt, short_name)
        if raw_key not in shared_dict:
            shared_dict[raw_key] = len(shared_dict)
        init_labels.append(shared_dict[raw_key])

    # Build adjacency lists with edge kinds
    in_adj: list[list[tuple[int, str]]] = [[] for _ in range(n)]
    out_adj: list[list[tuple[int, str]]] = [[] for _ in range(n)]

    for edge in edges:
        src_id, dst_id, kind = _get_edge_fields(edge)
        src_idx = node_to_idx.get(src_id)
        dst_idx = node_to_idx.get(dst_id)
        if src_idx is not None and dst_idx is not None:
            out_adj[src_idx].append((dst_idx, kind))
            in_adj[dst_idx].append((src_idx, kind))

    current_labels = init_labels

    for _ in range(rounds):
        next_labels: list[int] = []
        for i in range(n):
            in_multiset = sorted((kind, current_labels[src]) for src, kind in in_adj[i])
            out_multiset = sorted((kind, current_labels[dst]) for dst, kind in out_adj[i])
            sig = (current_labels[i], tuple(in_multiset), tuple(out_multiset))
            if sig not in shared_dict:
                shared_dict[sig] = len(shared_dict)
            next_labels.append(shared_dict[sig])
        current_labels = next_labels

    return current_labels


_CLEAN_DIFF_CACHE: dict[int, tuple[collections.Counter, set[str], dict[Any, int]]] = {}


def diff_graphs(
    clean_graph: Any,
    infected_graph: Any,
    rounds: int = 3,
) -> dict[str, int]:
    """
    Diff clean and infected digraphs using 3-round Weisfeiler-Lehman subtree hashing.
    
    Returns a mapping {infected_node_id: label (0 or 1)}.
    """
    clean_nodes = clean_graph["nodes"] if isinstance(clean_graph, dict) else clean_graph.nodes
    clean_edges = clean_graph["edges"] if isinstance(clean_graph, dict) else clean_graph.edges
    inf_nodes = infected_graph["nodes"] if isinstance(infected_graph, dict) else infected_graph.nodes
    inf_edges = infected_graph["edges"] if isinstance(infected_graph, dict) else infected_graph.edges

    clean_key = id(clean_graph)
    if clean_key in _CLEAN_DIFF_CACHE:
        clean_counts, clean_signal_names, base_dict = _CLEAN_DIFF_CACHE[clean_key]
        shared_dict = base_dict.copy()
    else:
        shared_dict = {}
        clean_hashes = compute_wl_hashes(clean_nodes, clean_edges, rounds=rounds, shared_dict=shared_dict)
        clean_counts = collections.Counter(clean_hashes)
        clean_signal_names = set()
        for node in clean_nodes:
            nid, ntype, _, _ = _get_node_fields(node)
            if ntype == "signal":
                clean_signal_names.add(nid)
        _CLEAN_DIFF_CACHE[clean_key] = (clean_counts, clean_signal_names, shared_dict.copy())

    inf_hashes = compute_wl_hashes(inf_nodes, inf_edges, rounds=rounds, shared_dict=shared_dict)

    labels: dict[str, int] = {}
    inf_seen = collections.Counter()

    for node, h in zip(inf_nodes, inf_hashes):
        nid, ntype, _, _ = _get_node_fields(node)
        is_signal = (ntype == "signal")
        is_new_signal = is_signal and (nid not in clean_signal_names)

        inf_seen[h] += 1
        is_extra_hash = (inf_seen[h] > clean_counts.get(h, 0))

        labels[nid] = 1 if (is_new_signal or is_extra_hash) else 0

    return labels


def validate_labeling(
    clean_graph: Any,
    infected_graph: Any,
    rounds: int = 3,
) -> dict[str, Any]:
    """
    Validate labeling against requirements:
    1. Clean vs itself must produce 0 positives.
    2. Compute positive fraction and signal name agreement.
    3. Return summary metrics.
    """
    clean_self_labels = diff_graphs(clean_graph, clean_graph, rounds=rounds)
    self_positives = sum(clean_self_labels.values())

    inf_labels = diff_graphs(clean_graph, infected_graph, rounds=rounds)
    total_nodes = len(inf_labels)
    total_positives = sum(inf_labels.values())
    pos_fraction = (total_positives / total_nodes) if total_nodes > 0 else 0.0

    clean_nodes = clean_graph["nodes"] if isinstance(clean_graph, dict) else clean_graph.nodes
    inf_nodes = infected_graph["nodes"] if isinstance(infected_graph, dict) else infected_graph.nodes

    clean_signal_names = {
        _get_node_fields(n)[0] for n in clean_nodes if _get_node_fields(n)[1] == "signal"
    }
    inf_signal_names = {
        _get_node_fields(n)[0] for n in inf_nodes if _get_node_fields(n)[1] == "signal"
    }
    infected_only_signals = inf_signal_names - clean_signal_names

    pos_signal_ids = {
        nid for nid, lbl in inf_labels.items()
        if lbl == 1 and nid in inf_signal_names
    }

    if pos_signal_ids:
        agreement = len(pos_signal_ids & infected_only_signals) / len(pos_signal_ids)
    else:
        agreement = 1.0 if not infected_only_signals else 0.0

    positive_samples = [
        {"id": _get_node_fields(n)[0], "type": _get_node_fields(n)[1], "label": _get_node_fields(n)[2]}
        for n in inf_nodes
        if inf_labels.get(_get_node_fields(n)[0], 0) == 1
    ][:15]

    return {
        "self_positives": self_positives,
        "total_nodes": total_nodes,
        "total_positives": total_positives,
        "positive_fraction": pos_fraction,
        "agreement_with_new_signals": agreement,
        "infected_only_signals_count": len(infected_only_signals),
        "flag_high_positives": pos_fraction > 0.10,
        "sample_positive_nodes": positive_samples,
    }
