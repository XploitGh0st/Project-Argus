"""argus_ml.metrics — Structured metrics computation for hardware security GNN tasks.

Provides a dataclass-based MetricsResult and helper functions for computing
binary classification metrics (accuracy, F1, precision, recall, confusion matrix)
used by both the TrojanTrainer and SimilarityTrainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class MetricsResult:
    """Immutable container for a single evaluation pass.

    Attributes:
        loss: Average loss over the evaluation set.
        accuracy: Classification accuracy in [0, 1].
        f1: Binary F1 score.
        precision: Binary precision.
        recall: Binary recall (sensitivity / true-positive rate).
        confusion_matrix: 2×2 ndarray ``[[TN, FP], [FN, TP]]``.
    """

    loss: float
    accuracy: float
    f1: float
    precision: float
    recall: float
    confusion_matrix: np.ndarray = field(repr=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, float | list[list[int]]]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


def compute_binary_metrics(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    loss: float,
) -> MetricsResult:
    """Compute all binary classification metrics in one call.

    Args:
        labels: Ground-truth binary labels (0 or 1).
        predictions: Predicted binary labels (0 or 1).
        loss: Pre-computed average loss for this evaluation pass.

    Returns:
        A :class:`MetricsResult` with all fields populated.
    """
    labels = np.asarray(labels, dtype=int).ravel()
    predictions = np.asarray(predictions, dtype=int).ravel()

    return MetricsResult(
        loss=float(loss),
        accuracy=float(accuracy_score(labels, predictions)),
        f1=float(f1_score(labels, predictions, average="binary", zero_division=0)),
        precision=float(precision_score(labels, predictions, average="binary", zero_division=0)),
        recall=float(recall_score(labels, predictions, average="binary", zero_division=0)),
        confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]),
    )


def format_metrics(result: MetricsResult, phase: str) -> str:
    """Format a :class:`MetricsResult` as a single readable log line.

    Args:
        result: The metrics to format.
        phase: Label such as ``"train"``, ``"val"``, or ``"best"``.

    Returns:
        A human-readable string suitable for ``print()`` or logging.
    """
    cm = np.array2string(result.confusion_matrix, separator=", ")
    return (
        f"{phase} loss: {result.loss:.4f}"
        f" | acc: {result.accuracy:.4f}"
        f" | f1: {result.f1:.4f}"
        f" | prec: {result.precision:.4f}"
        f" | rec: {result.recall:.4f}"
        f" | cm: {cm}"
    )
