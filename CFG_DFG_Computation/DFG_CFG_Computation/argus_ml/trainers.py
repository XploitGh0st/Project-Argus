"""argus_ml.trainers — Training loops for hardware security GNN tasks.

Provides two trainers, each tailored to a specific security analysis workflow:

- :class:`TrojanTrainer`: Binary graph classification for Hardware Trojan detection.
  Trains a :class:`~argus_ml.models.GraphEncoder` to label entire circuit graphs
  as *Trojan-Free* (0) or *Trojan-Infected* (1).

- :class:`SimilarityTrainer`: Pairwise graph similarity for IP piracy detection.
  Trains a :class:`~argus_ml.models.GraphEncoder` using cosine-embedding loss so
  that embeddings of functionally identical designs are close in vector space and
  those of unrelated designs are far apart.

Both trainers support:
- Configurable epochs, learning rate, weight decay, and evaluation frequency.
- Early stopping with a patience parameter.
- Best-model checkpointing.
- Structured metrics via :mod:`argus_ml.metrics`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm

from argus_ml.metrics import MetricsResult, compute_binary_metrics, format_metrics
from argus_ml.models import GraphEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_scalar(t: Tensor) -> float:
    """Detach a scalar tensor and return a plain Python float."""
    return float(t.detach().cpu())


# ---------------------------------------------------------------------------
# TrojanTrainer — Binary Graph Classification
# ---------------------------------------------------------------------------

class TrojanTrainer:
    """Trains a :class:`GraphEncoder` for Hardware Trojan detection.

    This corresponds to the ``"TJ"`` task in the original hw2vec framework,
    re-implemented with structured metrics, early stopping, and clean separation
    of training / inference / evaluation concerns.

    Args:
        model: A :class:`GraphEncoder` instance (already moved to the target device).
        config: Training configuration dict with keys:

            ============== ========= =============================================
            Key            Default   Description
            ============== ========= =============================================
            device         ``"cpu"`` Device string (``"cuda"``, ``"cpu"``, etc.)
            learning_rate  1e-3      Adam learning rate
            weight_decay   5e-4      Adam weight decay (L2 regularisation)
            epochs         100       Maximum number of training epochs
            eval_every     10        Run validation every N epochs
            patience       20        Stop early after N epochs without improvement
            seed           42        Random seed for reproducibility
            save_dir       None      Directory to save best model (optional)
            ============== ========= =============================================

        class_weights: Optional 1-D tensor of per-class weights for imbalanced
            datasets. If ``None``, uniform weighting is used.
    """

    def __init__(
        self,
        model: GraphEncoder,
        config: dict[str, Any],
        class_weights: Tensor | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = config.get("device", "cpu")

        # Reproducibility
        seed = config.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Optimiser
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 5e-4),
        )

        # Loss function (supports class imbalance)
        if class_weights is not None and class_weights.numel() >= 2:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights.float().to(self.device)
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Tracking
        self._best_loss = float("inf")
        self._best_metrics: MetricsResult | None = None
        self._patience_counter = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        """Run the full training loop with periodic validation and early stopping.

        Args:
            train_loader: PyTorch Geometric DataLoader for training data.
            val_loader: PyTorch Geometric DataLoader for validation data.
            epochs: Override the epoch count from config (optional).

        Returns:
            Dictionary with the best validation metrics (see :class:`MetricsResult`).
        """
        epochs = epochs or self.config.get("epochs", 100)
        eval_every = self.config.get("eval_every", 10)
        patience = self.config.get("patience", 20)
        save_dir = self.config.get("save_dir")

        progress = tqdm(range(epochs), desc="Training", unit="epoch")

        for epoch in progress:
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0

            for data in train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()

                logits, _ = self.model(data.x, data.edge_index, data.batch)
                logits = F.log_softmax(logits, dim=1)
                loss = self.loss_fn(logits, data.label)

                loss.backward()
                self.optimizer.step()
                epoch_loss += _to_scalar(loss)

            progress.set_postfix(loss=f"{epoch_loss:.4f}")

            # --- Validate ---
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                train_metrics = self._evaluate(train_loader)
                val_metrics = self._evaluate(val_loader)

                print(f"\n--- Epoch {epoch + 1} ---")
                print(format_metrics(train_metrics, "train"))
                print(format_metrics(val_metrics, "val  "))

                # Best model tracking
                if val_metrics.loss < self._best_loss:
                    self._best_loss = val_metrics.loss
                    self._best_metrics = val_metrics
                    self._patience_counter = 0

                    if save_dir:
                        self.model.save(Path(save_dir) / "best_model.pt")
                else:
                    self._patience_counter += eval_every

                # Early stopping
                if self._patience_counter >= patience:
                    print(
                        f"\nEarly stopping at epoch {epoch + 1} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

        # Final summary
        if self._best_metrics is not None:
            print(f"\n{'=' * 60}")
            print(format_metrics(self._best_metrics, "best "))
            print(f"{'=' * 60}")

        return self._best_metrics.to_dict() if self._best_metrics else {}

    # ------------------------------------------------------------------
    # Inference / Prediction
    # ------------------------------------------------------------------

    def predict(
        self, data_loader
    ) -> tuple[Tensor, Tensor, list[dict[str, list]]]:
        """Run inference on a dataset and return predictions with attention maps.

        Args:
            data_loader: PyTorch Geometric DataLoader.

        Returns:
            A 3-tuple of ``(labels, predictions, attention_maps)`` where:
            - ``labels``: Long tensor of ground-truth class labels.
            - ``predictions``: Long tensor of predicted class labels.
            - ``attention_maps``: List of dicts with pooling attention weights
              per batch (keys: ``original_batch``, ``pool_perm``, ``pool_batch``,
              ``pool_score``).
        """
        all_labels: list[np.ndarray] = []
        all_preds: list[Tensor] = []
        attention_maps: list[dict[str, list]] = []

        with torch.no_grad():
            self.model.eval()

            for data in data_loader:
                data = data.to(self.device)
                logits, attn = self.model(data.x, data.edge_index, data.batch)
                logits = F.log_softmax(logits, dim=1)

                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(data.label.cpu().numpy())

                if "pool_score" in attn:
                    attention_maps.append({
                        "original_batch": data.batch.cpu().numpy().tolist(),
                        "pool_perm": attn["pool_perm"].cpu().numpy().tolist(),
                        "pool_batch": attn["batch"].cpu().numpy().tolist(),
                        "pool_score": attn["pool_score"].cpu().numpy().tolist(),
                    })

        labels = torch.from_numpy(np.concatenate(all_labels))
        predictions = torch.cat(all_preds)
        return labels, predictions, attention_maps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate(self, data_loader) -> MetricsResult:
        """Run a full inference pass and return structured metrics."""
        all_labels: list[np.ndarray] = []
        all_preds: list[Tensor] = []
        total_loss = 0.0

        with torch.no_grad():
            self.model.eval()

            for data in data_loader:
                data = data.to(self.device)
                logits, _ = self.model(data.x, data.edge_index, data.batch)
                logits = F.log_softmax(logits, dim=1)

                loss = self.loss_fn(logits, data.label)
                total_loss += _to_scalar(loss)

                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(data.label.cpu().numpy())

        avg_loss = total_loss / max(len(data_loader), 1)
        labels = np.concatenate(all_labels)
        predictions = torch.cat(all_preds).numpy()

        return compute_binary_metrics(labels, predictions, avg_loss)


# ---------------------------------------------------------------------------
# SimilarityTrainer — Pairwise Graph Similarity (GNN4IP)
# ---------------------------------------------------------------------------

class SimilarityTrainer:
    """Trains a :class:`GraphEncoder` for IP piracy detection via pairwise similarity.

    This corresponds to the ``"IP"`` / GNN4IP task in the original hw2vec framework.
    Two hardware designs are embedded independently, and cosine similarity determines
    whether they are functionally equivalent (pirated copies) or genuinely different.

    Args:
        model: A :class:`GraphEncoder` instance (already moved to the target device).
        config: Training configuration dict (same keys as :class:`TrojanTrainer`,
            plus ``similarity_margin`` defaulting to 0.5).
    """

    def __init__(self, model: GraphEncoder, config: dict[str, Any]) -> None:
        self.model = model
        self.config = config
        self.device = config.get("device", "cpu")

        # Reproducibility
        seed = config.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Optimiser
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 5e-4),
        )

        # Similarity functions
        margin = config.get("similarity_margin", 0.5)
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8).to(self.device)
        self.cos_loss = nn.CosineEmbeddingLoss(margin=margin).to(self.device)

        # Tracking
        self._best_loss = float("inf")
        self._best_metrics: MetricsResult | None = None
        self._patience_counter = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        """Run the full pairwise training loop.

        Each batch from the data loader is expected to yield a 3-tuple of
        ``(graph1, graph2, labels)`` where ``labels`` is +1 (similar) or
        -1 (dissimilar).

        Args:
            train_loader: DataLoader yielding ``(graph1, graph2, labels)`` tuples.
            val_loader: DataLoader for validation.
            epochs: Override the epoch count from config (optional).

        Returns:
            Dictionary with the best validation metrics.
        """
        epochs = epochs or self.config.get("epochs", 100)
        eval_every = self.config.get("eval_every", 10)
        patience = self.config.get("patience", 20)
        save_dir = self.config.get("save_dir")

        progress = tqdm(range(epochs), desc="Training", unit="epoch")

        for epoch in progress:
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                graph1 = batch[0].to(self.device)
                graph2 = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                self.optimizer.zero_grad()

                emb1, _ = self.model.embed(graph1.x, graph1.edge_index, graph1.batch)
                emb2, _ = self.model.embed(graph2.x, graph2.edge_index, graph2.batch)

                # Project through classifier head as a learned MLP
                emb1 = self.model.classifier(emb1)
                emb2 = self.model.classifier(emb2)

                loss = self.cos_loss(emb1, emb2, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += _to_scalar(loss)

            progress.set_postfix(loss=f"{epoch_loss:.4f}")

            # --- Validate ---
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                train_metrics = self._evaluate(train_loader)
                val_metrics = self._evaluate(val_loader)

                print(f"\n--- Epoch {epoch + 1} ---")
                print(format_metrics(train_metrics, "train"))
                print(format_metrics(val_metrics, "val  "))

                if val_metrics.loss < self._best_loss:
                    self._best_loss = val_metrics.loss
                    self._best_metrics = val_metrics
                    self._patience_counter = 0

                    if save_dir:
                        self.model.save(Path(save_dir) / "best_model.pt")
                else:
                    self._patience_counter += eval_every

                if self._patience_counter >= patience:
                    print(
                        f"\nEarly stopping at epoch {epoch + 1} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

        if self._best_metrics is not None:
            print(f"\n{'=' * 60}")
            print(format_metrics(self._best_metrics, "best "))
            print(f"{'=' * 60}")

        return self._best_metrics.to_dict() if self._best_metrics else {}

    # ------------------------------------------------------------------
    # Inference / Prediction
    # ------------------------------------------------------------------

    def predict(
        self, data_loader
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run pairwise inference and return similarity scores.

        Args:
            data_loader: DataLoader yielding ``(graph1, graph2, labels)`` tuples.

        Returns:
            A 3-tuple of ``(labels, similarity_scores, predictions)`` where:
            - ``labels``: Binary tensor (1 = similar, 0 = dissimilar).
            - ``similarity_scores``: Float tensor of cosine similarities in [-1, 1].
            - ``predictions``: Binary tensor (similarity > 0.5 → similar).
        """
        all_labels: list[np.ndarray] = []
        all_sims: list[Tensor] = []

        with torch.no_grad():
            self.model.eval()

            for batch in data_loader:
                graph1 = batch[0].to(self.device)
                graph2 = batch[1].to(self.device)
                labels_batch = batch[2].to(self.device)

                emb1, _ = self.model.embed(
                    graph1.x, graph1.edge_index, graph1.batch
                )
                emb2, _ = self.model.embed(
                    graph2.x, graph2.edge_index, graph2.batch
                )
                emb1 = self.model.classifier(emb1)
                emb2 = self.model.classifier(emb2)

                similarity = self.cos_sim(emb1, emb2)
                all_sims.append(similarity.cpu())
                all_labels.append(labels_batch.cpu().numpy())

        labels = torch.from_numpy(np.concatenate(all_labels))
        labels_binary = (labels > 0).long()
        similarity_scores = torch.cat(all_sims)
        predictions = (similarity_scores > 0.5).long()

        return labels_binary, similarity_scores, predictions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate(self, data_loader) -> MetricsResult:
        """Evaluate pairwise similarity and return structured metrics."""
        all_labels: list[np.ndarray] = []
        all_sims: list[Tensor] = []
        total_loss = 0.0

        with torch.no_grad():
            self.model.eval()

            for batch in data_loader:
                graph1 = batch[0].to(self.device)
                graph2 = batch[1].to(self.device)
                labels_batch = batch[2].to(self.device)

                emb1, _ = self.model.embed(
                    graph1.x, graph1.edge_index, graph1.batch
                )
                emb2, _ = self.model.embed(
                    graph2.x, graph2.edge_index, graph2.batch
                )
                emb1 = self.model.classifier(emb1)
                emb2 = self.model.classifier(emb2)

                loss = self.cos_loss(emb1, emb2, labels_batch)
                total_loss += _to_scalar(loss)

                similarity = self.cos_sim(emb1, emb2)
                all_sims.append(similarity.cpu())
                all_labels.append(labels_batch.cpu().numpy())

        avg_loss = total_loss / max(len(data_loader), 1)
        labels = np.concatenate(all_labels)
        labels_binary = (labels > 0).astype(int)
        predictions = (torch.cat(all_sims).numpy() > 0.5).astype(int)

        return compute_binary_metrics(labels_binary, predictions, avg_loss)
