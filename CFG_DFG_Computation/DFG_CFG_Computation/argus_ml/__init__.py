"""argus_ml — Purpose-built GNN engine for Project Argus hardware security analysis.

This package provides graph neural network models and training utilities
optimised for analysing hardware circuit graphs (DFG, CFG, AST) extracted
from Verilog RTL designs.

Key components
--------------
:class:`GraphEncoder`
    Configurable GNN backbone supporting GCN/GIN convolution, SAGPool/TopKPool
    graph pooling, learnable node embeddings, and max/mean/add readout.

:class:`TrojanTrainer`
    Training loop for Hardware Trojan detection (binary graph classification).

:class:`SimilarityTrainer`
    Training loop for IP piracy detection (pairwise graph similarity via
    cosine embedding loss).

:class:`MetricsResult`
    Immutable container for evaluation metrics (accuracy, F1, precision,
    recall, confusion matrix).

:func:`compute_binary_metrics`
    One-call computation of all binary classification metrics.

:func:`profile_function`
    Decorator for cumulative wall-clock profiling.

:func:`profile_graph_extraction`
    Decorator for profiling graph extraction with per-graph statistics.

Quick start
-----------
::

    from argus_ml import GraphEncoder, TrojanTrainer

    model_config = {
        "num_features": 37,     # DFG node-type vocabulary size
        "hidden_dim": 64,
        "output_dim": 2,        # Trojan-Free vs Trojan-Infected
        "num_layers": 3,
        "conv_type": "gin",     # or "gcn"
        "pool_type": "sagpool",
        "pool_ratio": 0.5,
        "readout": "max",
        "dropout": 0.3,
        "node_encoding": "embedding",
        "device": "cuda",
    }
    model = GraphEncoder(model_config).to("cuda")

    trainer = TrojanTrainer(model, {
        "device": "cuda",
        "epochs": 200,
        "learning_rate": 1e-3,
        "patience": 30,
    })
    best = trainer.fit(train_loader, val_loader)
"""

__version__ = "1.0.0"

# ── Always available (no torch dependency) ────────────────────────────────
from argus_ml.metrics import MetricsResult, compute_binary_metrics, format_metrics
from argus_ml.profiling import profile_function, profile_graph_extraction

# ── Torch-dependent symbols (imported lazily) ─────────────────────────────
# GraphEncoder, TrojanTrainer, and SimilarityTrainer require torch and
# torch_geometric.  They are imported at the top level here so that
# ``from argus_ml import GraphEncoder`` works naturally, but a clear
# ImportError is raised when those libraries are absent rather than a
# cryptic low-level failure.
try:
    from argus_ml.models import GraphEncoder
    from argus_ml.trainers import SimilarityTrainer, TrojanTrainer
except ImportError as _e:  # pragma: no cover
    _msg = (
        "\n\nargus_ml: Could not import GNN components.\n"
        "Please install the required packages:\n"
        "    pip install torch torch_geometric\n"
        f"Original error: {_e}"
    )
    raise ImportError(_msg) from _e

__all__ = [
    # Models
    "GraphEncoder",
    # Trainers
    "TrojanTrainer",
    "SimilarityTrainer",
    # Metrics
    "MetricsResult",
    "compute_binary_metrics",
    "format_metrics",
    # Profiling
    "profile_function",
    "profile_graph_extraction",
]
