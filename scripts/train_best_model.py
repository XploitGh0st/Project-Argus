#!/usr/bin/env python3
"""
scripts/train_best_model.py - Train winning model (M1) on all labeled benchmarks.

Saves:
- models/best/model.joblib
- models/best/metadata.json
"""

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
CACHE_FEAT = ROOT / ".cache" / "features"
MANIFEST_PATH = ROOT / "benchmarks" / "manifest.json"
MODEL_DIR = ROOT / "models" / "best"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    train_X_list = []
    train_y_list = []
    train_fam_list = []
    feature_cols = []

    rng = np.random.RandomState(SEED)

    for b in manifest:
        bid = b["id"]
        family = b["family"]
        feat_file = CACHE_FEAT / bid / "infected.npz"
        if not feat_file.exists():
            continue

        data = np.load(feat_file, allow_pickle=True)
        X_b = data["X"].astype(np.float32)
        y_b = data["y"].astype(np.int64)

        if len(y_b) == 0:
            continue

        if not feature_cols and "columns" in data:
            feature_cols = list(data["columns"])

        pos_idx = np.where(y_b == 1)[0]
        neg_idx = np.where(y_b == 0)[0]

        max_neg = 20 * max(1, len(pos_idx))
        if len(neg_idx) > max_neg:
            sampled_neg = rng.choice(neg_idx, size=max_neg, replace=False)
        else:
            sampled_neg = neg_idx

        selected = np.concatenate([pos_idx, sampled_neg])
        rng.shuffle(selected)

        train_X_list.append(X_b[selected])
        train_y_list.append(y_b[selected])
        train_fam_list.extend([family] * len(selected))

    X_train = np.vstack(train_X_list)
    y_train = np.concatenate(train_y_list)

    # Family sample weights
    fam_counts = {}
    for f in train_fam_list:
        fam_counts[f] = fam_counts.get(f, 0) + 1
    sample_weights = np.array([1.0 / fam_counts[f] for f in train_fam_list], dtype=np.float32)
    sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)

    print(f"Training best M1 model on all {len(y_train)} samples across {len(fam_counts)} families...")
    model = HistGradientBoostingClassifier(
        max_iter=200, l2_regularization=1.0, random_state=SEED
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Compute training threshold at FPR <= 0.05
    train_probs = model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, train_probs)
    valid_fpr = np.where(fpr <= 0.05)[0]
    threshold = float(thresholds[valid_fpr[-1]]) if len(valid_fpr) > 0 else 0.05

    # Save model and metadata
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "M1_HistGradientBoosting",
        "feature_columns": feature_cols,
        "decision_threshold": round(threshold, 4),
        "num_features": len(feature_cols),
        "training_samples": len(y_train),
        "training_positives": int(np.sum(y_train)),
        "families": list(fam_counts.keys()),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH} (threshold={threshold:.4f})")


if __name__ == "__main__":
    main()
