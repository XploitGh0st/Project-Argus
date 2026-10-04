import json
from pathlib import Path

import networkx as nx
import pytest

from argus_cfg import (
    ANY_STATE,
    CFGResult,
    ContractError,
    build_cfg,
    state_node_id,
    validate_cfg_document,
)
from ast_builders import ALL_BUILDERS, build_no_fsm, build_traffic_light, build_two_fsms

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _fsm(result: CFGResult, suffix: str):
    return next(f for f in result.fsms if f.signal.endswith(suffix))


# ------------------------------------------------------------------ hand-written truth


def test_traffic_light_structure():
    ast, top = build_traffic_light()
    result = build_cfg(ast, top)

    assert result.top_module == "traffic"
    assert len(result.fsms) == 1
    fsm = result.fsms[0]
    assert fsm.signal == "traffic.state"
    assert fsm.states == ("0", "1", "2")
    edges = {(t.src, t.dst) for t in fsm.transitions}
    assert edges == {("0", "1"), ("1", "2"), ("2", "0")}


def test_traffic_light_conditions():
    ast, top = build_traffic_light()
    by_src = {t.src: t for t in build_cfg(ast, top).fsms[0].transitions}
    assert by_src["0"].condition is not None and "start" in by_src["0"].condition
    assert by_src["1"].condition is None  # unconditional
    assert by_src["2"].condition is None


def test_legacy_bug_edges_point_to_destination_states_not_labels():
    # legacy export_cfg_graph appended the edge *label*; here dst is always a state
    ast, top = build_traffic_light()
    fsm = build_cfg(ast, top).fsms[0]
    assert all(t.dst in fsm.states for t in fsm.transitions)


def test_two_independent_fsms():
    ast, top = build_two_fsms()
    result = build_cfg(ast, top)
    assert [f.signal for f in result.fsms] == ["dual.mode", "dual.state"]
    assert _fsm(result, ".mode").states == ("0", "1", "2")
    assert _fsm(result, ".state").states == ("0", "1")


def test_module_name_is_not_hardcoded_to_top():
    # the legacy code passed the literal "top" to the analyser regardless of the design
    ast, top = build_traffic_light()
    assert top != "top"
    assert build_cfg(ast, top).fsms  # works under its real name


def test_no_fsm_gives_empty_cfg():
    ast, top = build_no_fsm()
    result = build_cfg(ast, top)
    assert result.fsms == ()
    assert result.to_networkx().number_of_nodes() == 0


def test_custom_fsm_hints_can_hide_registers():
    ast, top = build_traffic_light()
    assert build_cfg(ast, top, fsm_hints=("nomatch",)).fsms == ()


def test_in_memory_ast_requires_top_module():
    ast, _ = build_traffic_light()
    with pytest.raises(ValueError, match="top_module is required"):
        build_cfg(ast)


def test_rejects_wrong_input_type():
    with pytest.raises(TypeError):
        build_cfg(42, "x")


# ------------------------------------------------------------------ graph view


def test_networkx_view():
    ast, top = build_traffic_light()
    g = build_cfg(ast, top).to_networkx()
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == 3 and g.number_of_edges() == 3
    n0, n1 = state_node_id("traffic.state", "0"), state_node_id("traffic.state", "1")
    assert g.has_edge(n0, n1)
    assert g.edges[n0, n1]["kind"] == "control_flow"
    assert g.nodes[n0]["kind"] == "fsm_state"
    assert nx.is_strongly_connected(g)


def test_any_state_transition_expands_to_every_state():
    from argus_cfg import FSM, Transition

    fsm = FSM("t.s", 0, ("0", "1"), (Transition("0", "1", None), Transition(ANY_STATE, "0", "rst")))
    g = CFGResult("t", (fsm,)).to_networkx()
    assert g.has_edge(state_node_id("t.s", "0"), state_node_id("t.s", "0"))
    assert g.has_edge(state_node_id("t.s", "1"), state_node_id("t.s", "0"))


# ------------------------------------------------------------------ golden + pickle mocks


@pytest.mark.parametrize("name", sorted(ALL_BUILDERS))
def test_matches_golden(name: str):
    ast, top = ALL_BUILDERS[name]()
    expected = json.loads((FIXTURES / "expected" / f"{name}.cfg.json").read_text("utf-8"))
    assert build_cfg(ast, top).to_dict() == expected


@pytest.mark.parametrize("name", sorted(ALL_BUILDERS))
def test_pickled_mock_ast_gives_same_result_as_builder(name: str):
    from_pkl = build_cfg(FIXTURES / "pkl" / f"{name}.ast.pkl")
    ast, top = ALL_BUILDERS[name]()
    assert from_pkl.to_dict() == build_cfg(ast, top).to_dict()


def test_pickle_with_wrong_kind_rejected(tmp_path: Path):
    import pickle

    bad = tmp_path / "x.pkl"
    bad.write_bytes(pickle.dumps({"kind": "dfg", "schema_version": "1.0"}))
    with pytest.raises(ValueError, match="not an AST artifact"):
        build_cfg(bad)


def test_pickle_with_wrong_version_rejected(tmp_path: Path):
    import pickle

    bad = tmp_path / "x.pkl"
    bad.write_bytes(pickle.dumps({"kind": "ast", "schema_version": "9.9"}))
    with pytest.raises(ValueError, match="unsupported AST schema"):
        build_cfg(bad)


# ------------------------------------------------------------------ output contract


@pytest.mark.parametrize("name", sorted(ALL_BUILDERS))
def test_output_conforms_to_contract(name: str):
    ast, top = ALL_BUILDERS[name]()
    validate_cfg_document(build_cfg(ast, top).to_dict())


def test_contract_validator_rejects_bad_documents():
    good = build_cfg(*build_traffic_light()).to_dict()
    for mutate in (
        lambda d: d.update(kind="dfg"),
        lambda d: d.update(schema_version="2.0"),
        lambda d: d["fsms"][0]["transitions"][0].update(dst="99"),
        lambda d: d["fsms"][0].update(states="012"),
    ):
        doc = json.loads(json.dumps(good))
        mutate(doc)
        with pytest.raises(ContractError):
            validate_cfg_document(doc)


def test_save_writes_json(tmp_path: Path):
    path = build_cfg(*build_traffic_light()).save(tmp_path)
    assert json.loads(path.read_text("utf-8"))["kind"] == "cfg"
