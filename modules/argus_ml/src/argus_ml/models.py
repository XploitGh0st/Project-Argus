"""argus_ml.models ÔÇö Graph Neural Network architectures for hardware security analysis.

This module provides the :class:`GraphEncoder`, a configurable GNN backbone that
converts hardware circuit graphs (DFG / CFG / AST) into fixed-dimensional vector
embeddings.  It supports:

- **Node encoding**: Learnable ``nn.Embedding`` (default) or one-hot projection,
  selectable via ``node_encoding`` config key.
- **Graph convolution**: GCN or GIN, selectable via ``conv_type`` config key.
- **Graph pooling**: SAGPool or TopKPool.
- **Graph readout**: Max, mean, or additive global pooling.

Typical usage::

    config = {
        "num_features": 37,
        "hidden_dim": 64,
        "output_dim": 2,
        "num_layers": 3,
        "conv_type": "gin",          # "gcn" or "gin"
        "pool_type": "sagpool",      # "sagpool" or "topkpool"
        "pool_ratio": 0.5,
        "readout": "max",            # "max", "mean", or "add"
        "dropout": 0.3,
        "node_encoding": "embedding",  # "embedding" or "one_hot"
        "device": "cuda",
    }
    model = GraphEncoder(config).to(config["device"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    SAGPooling,
    TopKPooling,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

# ---------------------------------------------------------------------------
# Node Feature Encoder ÔÇö Unified Hybrid Layer
# ---------------------------------------------------------------------------

class HardwareNodeEncoder(nn.Module):
    """Transforms integer node-type IDs into dense feature vectors.

    Supports two modes controlled by the ``mode`` argument:

    - ``"embedding"`` (default): Uses a learnable ``nn.Embedding`` lookup table.
      Produces semantically rich representations where similar hardware operations
      (e.g., AND / OR / XOR) can cluster in vector space during training.
    - ``"one_hot"``: Projects one-hot encoded vectors through a linear layer.
      Deterministic ÔÇö useful for reproducibility or when loading legacy models
      trained with ``F.one_hot``.

    Both modes output tensors of shape ``(N, hidden_dim)``, so downstream GNN
    layers are agnostic to the encoding strategy.

    Args:
        vocab_size: Number of distinct node types in the hardware graph vocabulary.
        hidden_dim: Dimensionality of the output feature vectors.
        mode: ``"embedding"`` or ``"one_hot"``.
        dropout: Dropout probability applied after encoding.
    """

    VALID_MODES = ("embedding", "one_hot")

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        mode: str = "embedding",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"node_encoding must be one of {self.VALID_MODES}, got {mode!r}"
            )

        self.mode = mode
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        if mode == "embedding":
            self.encoder = nn.Embedding(vocab_size, hidden_dim)
        else:  # one_hot
            self.projection = nn.Linear(vocab_size, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Encode integer node IDs into dense vectors.

        Args:
            x: Long tensor of shape ``(N,)`` with values in ``[0, vocab_size)``.

        Returns:
            Float tensor of shape ``(N, hidden_dim)``.
        """
        if self.mode == "embedding":
            out = self.encoder(x)
        else:
            out = F.one_hot(x, num_classes=self.vocab_size).float()
            out = self.projection(out)

        return self.dropout(self.layer_norm(out))


# ---------------------------------------------------------------------------
# Graph Convolution Block ÔÇö GCN or GIN
# ---------------------------------------------------------------------------

class GraphConvBlock(nn.Module):
    """A single graph convolution layer with batch normalisation and activation.

    Supports:
    - ``"gcn"``: Standard Graph Convolutional Network (Kipf & Welling, 2017).
    - ``"gin"``: Graph Isomorphism Network (Xu et al., ICLR 2019) ÔÇö provably
      as powerful as the 1-WL graph isomorphism test. Uses sum aggregation
      with a learnable ``epsilon`` and an internal 2-layer MLP, making it
      superior to GCN for detecting structural anomalies like hardware Trojans.

    Args:
        conv_type: ``"gcn"`` or ``"gin"``.
        in_channels: Input feature dimensionality.
        out_channels: Output feature dimensionality.
        dropout: Dropout probability after activation.
    """

    VALID_TYPES = ("gcn", "gin")

    def __init__(
        self,
        conv_type: str,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if conv_type not in self.VALID_TYPES:
            raise ValueError(
                f"conv_type must be one of {self.VALID_TYPES}, got {conv_type!r}"
            )

        self.conv_type = conv_type

        if conv_type == "gcn":
            self.conv = GCNConv(in_channels, out_channels)
        else:  # gin
            # GIN requires an internal MLP; we use a 2-layer perceptron
            # with BatchNorm for stable training.
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, out_channels),
            )
            self.conv = GINConv(nn=mlp, train_eps=True)

        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Apply convolution ÔåÆ batch norm ÔåÆ ReLU ÔåÆ dropout.

        Args:
            x: Node feature matrix of shape ``(N, in_channels)``.
            edge_index: Graph connectivity in COO format ``(2, E)``.

        Returns:
            Updated node features of shape ``(N, out_channels)``.
        """
        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x, inplace=True)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Graph Pooling & Readout Factories
# ---------------------------------------------------------------------------

_POOL_REGISTRY: dict[str, type] = {
    "sagpool": SAGPooling,
    "topkpool": TopKPooling,
}

_READOUT_REGISTRY: dict[str, Any] = {
    "max": global_max_pool,
    "mean": global_mean_pool,
    "add": global_add_pool,
}


def _build_pool(pool_type: str, in_channels: int, ratio: float) -> nn.Module:
    """Construct a graph pooling layer from a string identifier."""
    if pool_type not in _POOL_REGISTRY:
        raise ValueError(
            f"pool_type must be one of {list(_POOL_REGISTRY)}, got {pool_type!r}"
        )
    return _POOL_REGISTRY[pool_type](in_channels, ratio=ratio)


# ---------------------------------------------------------------------------
# GraphEncoder ÔÇö The Full GNN Backbone
# ---------------------------------------------------------------------------

class GraphEncoder(nn.Module):
    """Configurable Graph Neural Network for hardware design analysis.

    Converts integer-labelled circuit graphs into fixed-dimension embeddings
    suitable for downstream tasks (Trojan detection, IP piracy detection).

    The architecture pipeline is::

        Integer node IDs
              Ôöé
              Ôû╝
        HardwareNodeEncoder  (embedding or one-hot ÔåÆ hidden_dim)
              Ôöé
              Ôû╝
        GraphConvBlock ├ù N   (GCN or GIN, each with BatchNorm + ReLU + Dropout)
              Ôöé
              Ôû╝
        Graph Pooling        (SAGPool or TopKPool ÔÇö learnable node selection)
              Ôöé
              Ôû╝
        Global Readout       (max / mean / add aggregation)
              Ôöé
              Ôû╝
        Classification Head  (nn.Linear ÔåÆ output_dim)

    Args:
        config: Dictionary with the following keys:

            ========== ============ =================================================
            Key        Default      Description
            ========== ============ =================================================
            num_features  (required) Vocabulary size (number of distinct node types)
            hidden_dim    64        Hidden feature dimensionality
            output_dim    2         Number of output classes
            num_layers    3         Number of graph convolution blocks
            conv_type     ``"gcn"`` ``"gcn"`` or ``"gin"``
            pool_type     ``"sagpool"`` ``"sagpool"`` or ``"topkpool"``
            pool_ratio    0.5       Fraction of nodes retained by pooling
            readout       ``"max"`` ``"max"``, ``"mean"``, or ``"add"``
            dropout       0.3       Dropout probability
            node_encoding ``"embedding"`` ``"embedding"`` or ``"one_hot"``
            device        ``"cpu"`` Target device string
            ========== ============ =================================================
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self._config = dict(config)  # defensive copy

        # --- Unpack config with defaults ---
        num_features: int = config["num_features"]
        hidden_dim: int = config.get("hidden_dim", 64)
        output_dim: int = config.get("output_dim", 2)
        num_layers: int = config.get("num_layers", 3)
        conv_type: str = config.get("conv_type", "gcn")
        pool_type: str = config.get("pool_type", "sagpool")
        pool_ratio: float = config.get("pool_ratio", 0.5)
        readout: str = config.get("readout", "max")
        dropout: float = config.get("dropout", 0.3)
        node_encoding: str = config.get("node_encoding", "embedding")

        if readout not in _READOUT_REGISTRY:
            raise ValueError(
                f"readout must be one of {list(_READOUT_REGISTRY)}, got {readout!r}"
            )

        # --- Build layers ---
        self.node_encoder = HardwareNodeEncoder(
            vocab_size=num_features,
            hidden_dim=hidden_dim,
            mode=node_encoding,
            dropout=dropout,
        )

        # Convolution stack
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dim  # all layers operate at hidden_dim after encoding
            self.conv_layers.append(
                GraphConvBlock(conv_type, in_ch, hidden_dim, dropout=dropout)
            )

        # Pooling & readout
        self.pool = _build_pool(pool_type, hidden_dim, pool_ratio)
        self._readout_fn = _READOUT_REGISTRY[readout]
        self._readout_name = readout

        # Classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Full forward pass: encode ÔåÆ convolve ÔåÆ pool ÔåÆ readout ÔåÆ classify.

        Args:
            x: Integer node-type tensor of shape ``(N,)``.
            edge_index: COO edge tensor of shape ``(2, E)``.
            batch: Batch assignment vector of shape ``(N,)``.

        Returns:
            A tuple of ``(logits, attention_weights)`` where:
            - ``logits`` has shape ``(B, output_dim)``
            - ``attention_weights`` is a dict with keys ``pool_perm``,
              ``pool_score``, and ``batch`` (post-pooling).
        """
        graph_emb, attn = self.embed(x, edge_index, batch)
        logits = self.classifier(graph_emb)
        return logits, attn

    def embed(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute graph-level embeddings without the classification head.

        Useful for pairwise similarity tasks (IP piracy detection) where two
        graphs are embedded independently and then compared via cosine similarity.

        Args:
            x: Integer node-type tensor of shape ``(N,)``.
            edge_index: COO edge tensor of shape ``(2, E)``.
            batch: Batch assignment vector of shape ``(N,)``.

        Returns:
            A tuple of ``(embeddings, attention_weights)`` where:
            - ``embeddings`` has shape ``(B, hidden_dim)``
            - ``attention_weights`` dict as described in :meth:`forward`.
        """
        attn: dict[str, Tensor] = {}

        # Encode node features
        h = self.node_encoder(x)

        # Message passing
        for conv in self.conv_layers:
            h = conv(h, edge_index)

        # Graph pooling
        h, edge_index, _, batch, perm, score = self.pool(h, edge_index, batch=batch)
        attn["pool_perm"] = perm
        attn["pool_score"] = score
        attn["batch"] = batch

        # Global readout
        h = self._readout_fn(h, batch)

        return h, attn

    def embed_nodes(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Compute node-level embeddings (no pooling or readout).

        Useful for node-level analysis or visualisation of individual
        signal representations within a hardware design.

        Args:
            x: Integer node-type tensor of shape ``(N,)``.
            edge_index: COO edge tensor of shape ``(2, E)``.

        Returns:
            Node embedding matrix of shape ``(N, hidden_dim)``.
        """
        h = self.node_encoder(x)
        for conv in self.conv_layers:
            h = conv(h, edge_index)
        return h

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the full model (config + weights) to a single file.

        Args:
            path: Destination file path (e.g., ``"models/argus_gnn.pt"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": self._config, "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> GraphEncoder:
        """Load a saved GraphEncoder from disk.

        Args:
            path: Path to the saved ``.pt`` file.
            device: Device to map the model onto.

        Returns:
            A fully initialised :class:`GraphEncoder` with loaded weights.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        """Return a copy of the configuration used to build this model."""
        return dict(self._config)

    def __repr__(self) -> str:
        conv = self._config.get("conv_type", "gcn")
        enc = self._config.get("node_encoding", "embedding")
        layers = self._config.get("num_layers", 3)
        hidden = self._config.get("hidden_dim", 64)
        return (
            f"GraphEncoder(conv={conv}, encoding={enc}, "
            f"layers={layers}, hidden={hidden})"
        )
