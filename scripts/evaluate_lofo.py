#!/usr/bin/env python3
"""
scripts/evaluate_lofo.py - Leave-One-Family-Out (LOFO) Evaluation.

Evaluates 4 held-out families: AES, RS232, PIC16F84, wb_conmax.
Models:
  B0: Decision tree baseline
  M1: HistGradientBoosting (primary)
  M2: M1 without control_dep features (ablation)
  M3: Typed R-GCN (PyG: RGCNConv, 2 layers, hidden 32, 3 relation types)
  M4: Untyped GCN (ablation: single relation type)
  M5: Ensemble (0.5 * M1 + 0.5 * M3)
Controls:
  - Shuffled labels control on M1 (expected ROC-AUC ~ 0.50)
  - Delta ROC-AUC (M1 vs M2)

Saves detailed metrics to results/lofo_results.json.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

# Seed everything for strict reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "benchmarks" / "manifest.json"
CACHE_FEAT = ROOT / ".cache" / "features"
CACHE_DIGRAPH = ROOT / ".cache" / "digraphs"
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "lofo_results.json"


# ---------------------------------------------------------------------------
# GNN Models (PyG R-GCN and Untyped GCN)
# ---------------------------------------------------------------------------
class RGCNModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, num_relations: int = 3):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, 2, num_relations=num_relations)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


# ---------------------------------------------------------------------------
# Data Loading Helper
# ---------------------------------------------------------------------------
def load_all_data() -> tuple[dict[str, dict[str, Any]], list[str]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    benchmarks_data: dict[str, dict[str, Any]] = {}
    feature_cols: list[str] = []

    for b in manifest:
        bid = b["id"]
        family = b["family"]
        feat_file = CACHE_FEAT / bid / "infected.npz"
        graph_file = CACHE_DIGRAPH / bid / "infected.json"

        if not feat_file.exists() or not graph_file.exists():
            continue

        data = np.load(feat_file, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        node_ids = list(data["node_ids"])
        if not feature_cols and "columns" in data:
            feature_cols = list(data["columns"])

        if len(y) == 0:
            continue

        # Build PyG edges
        nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        g_data = json.loads(graph_file.read_text(encoding="utf-8"))
        src_list: list[int] = []
        dst_list: list[int] = []
        type_list: list[int] = []

        type_map = {"dataflow": 0, "control_dep": 1, "fsm_trans": 2}

        for e in g_data.get("edges", []):
            s = nid_to_idx.get(e.get("src"))
            d = nid_to_idx.get(e.get("dst"))
            if s is not None and d is not None:
                src_list.append(s)
                dst_list.append(d)
                type_list.append(type_map.get(e.get("kind"), 0))

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_type = torch.tensor(type_list, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros((0,), dtype=torch.long)

        benchmarks_data[bid] = {
            "id": bid,
            "family": family,
            "X": X,
            "y": y,
            "node_ids": node_ids,
            "edge_index": edge_index,
            "edge_type": edge_type,
        }

    return benchmarks_data, feature_cols


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos > 0 and n_neg > 0:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        roc_auc = 0.5

    if n_pos > 0:
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        pr_auc = 0.0

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "threshold": round(threshold, 4),
        "n_total": len(y_true),
        "n_pos": n_pos,
    }


def main():
    print("=== LEAVE-ONE-FAMILY-OUT (LOFO) EVALUATION ===")
    t_start = time.time()
    data_by_bench, feature_cols = load_all_data()
    print(f"Loaded {len(data_by_bench)} benchmarks with {len(feature_cols)} features each.")

    # Families present in dataset
    families = sorted(list({b["family"] for b in data_by_bench.values()}))
    print(f"Evaluation families: {families}")

    # Column indices for M2 (no control_dep features)
    non_ctrl_indices = [i for i, col in enumerate(feature_cols) if "ctrl" not in col]
    print(f"Features: total={len(feature_cols)}, non-control (M2)={len(non_ctrl_indices)}")

    # We evaluate LOFO on all 4 families
    lofo_results: dict[str, Any] = {
        "models": ["B0", "M1", "M2", "M3", "M4", "M5"],
        "controls": ["M1_shuffled_control"],
        "families": {},
        "summary": {},
    }

    # Store all test predictions for micro-pooled overall metrics
    pooled_y_true: dict[str, list[int]] = {m: [] for m in ["B0", "M1", "M2", "M3", "M4", "M5", "M1_shuffled"]}
    pooled_y_prob: dict[str, list[float]] = {m: [] for m in ["B0", "M1", "M2", "M3", "M4", "M5", "M1_shuffled"]}
    family_metrics_all: dict[str, dict[str, dict[str, float]]] = {}

    for held_out_family in families:
        print(f"\n=======================================================")
        print(f"FOLD: Held-out family = {held_out_family}")
        print(f"=======================================================")

        # Split train vs test benchmarks
        test_bench_ids = [bid for bid, d in data_by_bench.items() if d["family"] == held_out_family]
        train_bench_ids = [bid for bid, d in data_by_bench.items() if d["family"] != held_out_family]

        # -------------------------------------------------------------------
        # 1. Training Set Downsampling (all positives + 1:20 negatives)
        # -------------------------------------------------------------------
        train_X_list: list[np.ndarray] = []
        train_y_list: list[np.ndarray] = []
        train_fam_list: list[str] = []

        # For GNN training: select up to 3 diverse benchmarks per training family
        gnn_train_bench_ids: list[str] = []
        train_fams = sorted(list({data_by_bench[bid]["family"] for bid in train_bench_ids}))
        for tf in train_fams:
            f_benches = [bid for bid in train_bench_ids if data_by_bench[bid]["family"] == tf]
            # Pick up to 3 representative designs
            gnn_train_bench_ids.extend(f_benches[:3])

        rng = np.random.RandomState(SEED)

        for bid in train_bench_ids:
            bd = data_by_bench[bid]
            X_b = bd["X"]
            y_b = bd["y"]
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
            train_fam_list.extend([bd["family"]] * len(selected))

        X_train = np.vstack(train_X_list)
        y_train = np.concatenate(train_y_list)

        # Family sample weighting (1.0 / len(family_samples))
        fam_counts = {}
        for f in train_fam_list:
            fam_counts[f] = fam_counts.get(f, 0) + 1
        sample_weights = np.array([1.0 / fam_counts[f] for f in train_fam_list], dtype=np.float32)
        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)

        train_pos_rate = float(np.mean(y_train))
        print(f"Train samples: {len(y_train)} (positives={np.sum(y_train)}, pos_rate={train_pos_rate*100:.2f}%)")

        # -------------------------------------------------------------------
        # 2. Train Models
        # -------------------------------------------------------------------
        print("Training B0 (DecisionTree baseline)...")
        b0 = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEED)
        b0.fit(X_train, y_train, sample_weight=sample_weights)

        print("Training M1 (HistGradientBoosting primary)...")
        m1 = HistGradientBoostingClassifier(
            max_iter=200, l2_regularization=1.0, random_state=SEED
        )
        m1.fit(X_train, y_train, sample_weight=sample_weights)

        print("Training M2 (HistGradientBoosting ablation: non-control)...")
        m2 = HistGradientBoostingClassifier(
            max_iter=200, l2_regularization=1.0, random_state=SEED
        )
        m2.fit(X_train[:, non_ctrl_indices], y_train, sample_weight=sample_weights)

        print("Training Shuffled Labels Control on M1...")
        m1_shuffled = HistGradientBoostingClassifier(
            max_iter=200, l2_regularization=1.0, random_state=SEED
        )
        y_train_shuffled = rng.permutation(y_train)
        m1_shuffled.fit(X_train, y_train_shuffled, sample_weight=sample_weights)

        # -------------------------------------------------------------------
        # 3. Train M3 (Typed R-GCN) & M4 (Untyped GCN)
        # -------------------------------------------------------------------
        print(f"Training M3 (Typed R-GCN) on {len(gnn_train_bench_ids)} representative designs...")
        device = torch.device("cpu")
        m3 = RGCNModel(in_channels=len(feature_cols), hidden_channels=32, num_relations=3).to(device)
        optimizer_m3 = torch.optim.Adam(m3.parameters(), lr=0.01)

        # Class weights for PyG loss
        n_pos_train = max(1, int(np.sum(y_train)))
        n_neg_train = max(1, len(y_train) - n_pos_train)
        pos_weight = torch.tensor([1.0, float(n_neg_train / n_pos_train)], dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=pos_weight)

        # 50 Epochs of R-GCN training
        m3.train()
        for epoch in range(50):
            optimizer_m3.zero_grad()
            for bid in gnn_train_bench_ids:
                bd = data_by_bench[bid]
                x_t = torch.from_numpy(bd["X"]).to(device)
                y_t = torch.from_numpy(bd["y"]).to(device)
                ei_t = bd["edge_index"].to(device)
                et_t = bd["edge_type"].to(device)

                if ei_t.shape[1] == 0:
                    continue

                out = m3(x_t, ei_t, et_t)
                loss = loss_fn(out, y_t) / len(gnn_train_bench_ids)
                loss.backward()
            optimizer_m3.step()

        print(f"Training M4 (Untyped GCN ablation)...")
        m4 = RGCNModel(in_channels=len(feature_cols), hidden_channels=32, num_relations=1).to(device)
        optimizer_m4 = torch.optim.Adam(m4.parameters(), lr=0.01)
        m4.train()
        for epoch in range(50):
            optimizer_m4.zero_grad()
            for bid in gnn_train_bench_ids:
                bd = data_by_bench[bid]
                x_t = torch.from_numpy(bd["X"]).to(device)
                y_t = torch.from_numpy(bd["y"]).to(device)
                ei_t = bd["edge_index"].to(device)
                et_zeros = torch.zeros(ei_t.shape[1], dtype=torch.long, device=device)

                if ei_t.shape[1] == 0:
                    continue

                out = m4(x_t, ei_t, et_zeros)
                loss = loss_fn(out, y_t) / len(gnn_train_bench_ids)
                loss.backward()
            optimizer_m4.step()

        # -------------------------------------------------------------------
        # 4. Decision Threshold on Training Fold (FPR <= 0.05)
        # -------------------------------------------------------------------
        train_prob_m1 = m1.predict_proba(X_train)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_train, train_prob_m1)
        valid_fpr = np.where(fpr <= 0.05)[0]
        if len(valid_fpr) > 0:
            best_idx = valid_fpr[-1]
            k_fpr05 = float(thresholds[best_idx])
        else:
            k_fpr05 = 0.5

        # Also median positive rate across train families
        fam_pos_rates = [
            float(np.mean(data_by_bench[bid]["y"])) for bid in train_bench_ids
        ]
        k_med = float(np.median(fam_pos_rates))
        print(f"Decided thresholds: headline (train FPR<=0.05) = {k_fpr05:.4f}, median_pos_rate = {k_med:.4f}")

        # -------------------------------------------------------------------
        # 5. Evaluate on Test Benchmarks
        # -------------------------------------------------------------------
        m3.eval()
        m4.eval()

        fold_preds = {"B0": [], "M1": [], "M2": [], "M3": [], "M4": [], "M5": [], "M1_shuffled": []}
        fold_y = []

        test_benchmark_metrics = []

        for bid in test_bench_ids:
            bd = data_by_bench[bid]
            X_test = bd["X"]
            y_test = bd["y"]
            ei = bd["edge_index"].to(device)
            et = bd["edge_type"].to(device)

            p_b0 = b0.predict_proba(X_test)[:, 1]
            p_m1 = m1.predict_proba(X_test)[:, 1]
            p_m2 = m2.predict_proba(X_test[:, non_ctrl_indices])[:, 1]
            p_shuff = m1_shuffled.predict_proba(X_test)[:, 1]

            with torch.no_grad():
                x_torch = torch.from_numpy(X_test).to(device)
                if ei.shape[1] > 0:
                    out_m3 = m3(x_torch, ei, et)
                    p_m3 = F.softmax(out_m3, dim=-1)[:, 1].cpu().numpy()

                    out_m4 = m4(x_torch, ei, torch.zeros(ei.shape[1], dtype=torch.long, device=device))
                    p_m4 = F.softmax(out_m4, dim=-1)[:, 1].cpu().numpy()
                else:
                    p_m3 = np.full(len(y_test), 0.5, dtype=np.float32)
                    p_m4 = np.full(len(y_test), 0.5, dtype=np.float32)

            p_m5 = 0.5 * p_m1 + 0.5 * p_m3

            # Record benchmark metrics for M1 (primary)
            b_met = compute_metrics(y_test, p_m1, k_fpr05)
            b_met["benchmark"] = bid
            test_benchmark_metrics.append(b_met)

            # Accumulate for fold
            fold_y.append(y_test)
            fold_preds["B0"].append(p_b0)
            fold_preds["M1"].append(p_m1)
            fold_preds["M2"].append(p_m2)
            fold_preds["M3"].append(p_m3)
            fold_preds["M4"].append(p_m4)
            fold_preds["M5"].append(p_m5)
            fold_preds["M1_shuffled"].append(p_shuff)

        # Concatenate fold test arrays
        y_test_fold = np.concatenate(fold_y)
        preds_fold = {m: np.concatenate(fold_preds[m]) for m in fold_preds}

        # Compute metrics per model for this family
        fam_metrics = {}
        for m in fold_preds:
            m_met = compute_metrics(y_test_fold, preds_fold[m], k_fpr05)
            fam_metrics[m] = m_met

        delta_m1_m2 = fam_metrics["M1"]["roc_auc"] - fam_metrics["M2"]["roc_auc"]
        fam_metrics["delta_roc_auc_m1_vs_m2"] = round(delta_m1_m2, 4)

        family_metrics_all[held_out_family] = fam_metrics
        lofo_results["families"][held_out_family] = {
            "headline_threshold": k_fpr05,
            "median_threshold": k_med,
            "metrics": fam_metrics,
            "benchmarks": test_benchmark_metrics,
        }

        # Print family summary
        print(f"\nResults on held-out family [{held_out_family}] (N={len(y_test_fold)}):")
        print(f"  Model  | ROC-AUC | PR-AUC | Prec   | Recall | F1")
        print(f"  -------|---------|--------|--------|--------|------")
        for m in ["B0", "M1", "M2", "M3", "M4", "M5", "M1_shuffled"]:
            met = fam_metrics[m]
            print(f"  {m:6s} | {met['roc_auc']:.4f}  | {met['pr_auc']:.4f} | {met['precision']:.4f} | {met['recall']:.4f} | {met['f1']:.4f}")
        print(f"  Delta ROC-AUC (M1 - M2 non-ctrl): {delta_m1_m2:+.4f}")

        # Accumulate pooled
        for m in fold_preds:
            pooled_y_prob[m].extend(preds_fold[m].tolist())
        pooled_y_true["M1"].extend(y_test_fold.tolist())

    # -----------------------------------------------------------------------
    # Overall Macro-Average and Micro-Pooled Metrics
    # -----------------------------------------------------------------------
    macro_summary = {}
    for m in ["B0", "M1", "M2", "M3", "M4", "M5", "M1_shuffled"]:
        macro_summary[m] = {
            "roc_auc": round(float(np.mean([family_metrics_all[f][m]["roc_auc"] for f in families])), 4),
            "pr_auc": round(float(np.mean([family_metrics_all[f][m]["pr_auc"] for f in families])), 4),
            "precision": round(float(np.mean([family_metrics_all[f][m]["precision"] for f in families])), 4),
            "recall": round(float(np.mean([family_metrics_all[f][m]["recall"] for f in families])), 4),
            "f1": round(float(np.mean([family_metrics_all[f][m]["f1"] for f in families])), 4),
        }

    micro_summary = {}
    y_all = np.array(pooled_y_true["M1"])
    for m in ["B0", "M1", "M2", "M3", "M4", "M5", "M1_shuffled"]:
        p_all = np.array(pooled_y_prob[m])
        # Use median of fold thresholds
        med_thresh = float(np.median([lofo_results["families"][f]["headline_threshold"] for f in families]))
        micro_summary[m] = compute_metrics(y_all, p_all, med_thresh)

    lofo_results["summary"] = {
        "macro_average": macro_summary,
        "micro_pooled": micro_summary,
        "delta_roc_auc_macro": round(macro_summary["M1"]["roc_auc"] - macro_summary["M2"]["roc_auc"], 4),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(lofo_results, indent=2), encoding="utf-8")
    print(f"\n=======================================================")
    print(f"=== OVERALL LOFO EVALUATION COMPLETE ===")
    print(f"=======================================================")
    print(f"Macro-Average Across Families:")
    print(f"  Model  | ROC-AUC | PR-AUC | Prec   | Recall | F1")
    print(f"  -------|---------|--------|--------|--------|------")
    for m in ["B0", "M1", "M2", "M3", "M4", "M5", "M1_shuffled"]:
        met = macro_summary[m]
        print(f"  {m:6s} | {met['roc_auc']:.4f}  | {met['pr_auc']:.4f} | {met['precision']:.4f} | {met['recall']:.4f} | {met['f1']:.4f}")

    print(f"\nDelta ROC-AUC (M1 - M2 ablation): {lofo_results['summary']['delta_roc_auc_macro']:+.4f}")
    print(f"Saved complete evaluation to {RESULTS_JSON}")
    print(f"Total LOFO runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
