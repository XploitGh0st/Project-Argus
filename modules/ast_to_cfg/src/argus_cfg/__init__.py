"""argus_cfg - AST -> Control Flow Graph (Project Argus).

    from argus_cfg import build_cfg
    result = build_cfg("out/ast.pkl")        # or a pyverilog Source node + top_module
    result.save("out/")                       # cfg.json
    graph = result.to_networkx()
"""

from argus_cfg.extract import DEFAULT_FSM_HINTS, build_cfg
from argus_cfg.result import (
    ANY_STATE,
    FSM,
    KIND,
    SCHEMA_VERSION,
    CFGResult,
    ContractError,
    Transition,
    state_node_id,
    validate_cfg_document,
)

__version__ = "0.1.0"

__all__ = [
    "ANY_STATE",
    "DEFAULT_FSM_HINTS",
    "FSM",
    "KIND",
    "SCHEMA_VERSION",
    "CFGResult",
    "ContractError",
    "Transition",
    "__version__",
    "build_cfg",
    "state_node_id",
    "validate_cfg_document",
]
