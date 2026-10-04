"""argus_dfg - AST -> Data Flow Graph (Project Argus).

    from argus_dfg import build_dfg
    result = build_dfg("out/ast.pkl")        # or a pyverilog Source node + top_module
    result.save("out/")                       # dfg.json
    graph = result.to_networkx()
"""

from argus_dfg.extract import build_dfg
from argus_dfg.result import (
    KIND,
    NODE_TYPES,
    SCHEMA_VERSION,
    ContractError,
    DFGResult,
    Edge,
    Node,
    validate_dfg_document,
)

__version__ = "0.1.0"

__all__ = [
    "KIND",
    "NODE_TYPES",
    "SCHEMA_VERSION",
    "ContractError",
    "DFGResult",
    "Edge",
    "Node",
    "__version__",
    "build_dfg",
    "validate_dfg_document",
]
