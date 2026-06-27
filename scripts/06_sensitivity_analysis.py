#!/usr/bin/env python3
"""Step 6: k/expansion sensitivity analysis.

This script tests the robustness of concept distribution findings to SAE
hyperparameter choices (Section 5, Table 4):

  Experiment 1 -- k sensitivity (expansion = 4x):
    k = 32, 64, 128 on 30 stocks.

  Experiment 2 -- Expansion factor sensitivity (k = 64):
    expansion = 2x, 4x, 8x on 30 stocks.

For each configuration, we train per-stock SAEs, evaluate concept distribution
metrics (variance explained, dead rate, largest family, number of families),
and run null calibration.

Paper reference: Section 5 (Robustness to SAE Configuration), Table 4.
Output: outputs/results/sensitivity_results.json
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sae import TopKSAE, train_sae
from src.statistics import STATISTIC_NAMES, compute_labels_for_windows
from src.utils import (
    load_config,
    load_stock,
    normalize,
    create_windows,
    extract_activations,
    load_kronos,
    set_seed,
    split_windows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_sae_config(
    sae: TopKSAE,
    test_acts: np.ndarray,
    labels: np.ndarray,
    device: str,
    threshold: float = 0.15,
    min_active: int = 5,
) -> dict:
    """Evaluate a single SAE on test data with concept labeling."""
    at = torch.from_numpy(test_acts).float().to(device)

    with torch.no_grad():
        lat_tensor = sae.encode(at)
        lat = lat_tensor.cpu().numpy()
        recon = sae.decode(lat_tensor).cpu().numpy()

    var_total = float(np.var(test_acts))
    mse = float(np.mean((recon - test_acts) ** 2))
    var_explained = float(1.0 - mse / max(var_total, 1e-10))

    dead_mask = (lat != 0).sum(axis=0) == 0
    dead_rate = float(dead_mask.mean())
    alive_count = int((~dead_mask).sum())

    # Feature-statistic correlations
    alive_where = (lat != 0).sum(axis=0) > min_active
    type_dist = defaultdict(int)
    feature_corrs = {}

    for j in np.where(alive_where)[0]:
        active = lat[:, j] != 0
        if active.sum() < min_active:
            continue
        corrs = []
        for k_stat in range(labels.shape[1]):
            c = np.corrcoef(lat[active, j], labels[active, k_stat])[0, 1]
            corrs.append(0.0 if np.isnan(c) else abs(float(c)))
        feature_corrs[int(j)] = corrs
        best_idx = int(np.argmax(corrs))
        if corrs[best_idx] > threshold:
            type_dist[STATISTIC_NAMES[best_idx]] += 1

    total = sum(type_dist.values())
    largest_pct = max(type_dist.values()) / max(total, 1) if type_dist else 0.0
    n_families = len(type_dist)

    return {
        "var_explained": var_explained,
        "dead_rate": dead_rate,
        "alive_count": alive_count,
        "largest_pct": largest_pct,
        "n_families": n_families,
        "type_dist": dict(type_dist),
        "_feature_corrs": feature_corrs,
        "_lat": lat,
        "_labels": labels,
    }


def null_calibrate(
    eval_results: list[dict],
    n_shuffles: int = 50,
    seed: int = 42,
) -> tuple[float, int]:
    """Pooled null calibration across stocks. Returns (null_95, n_families_null)."""
    rng = np.random.RandomState(seed)
    null_maxes = []

    for _ in range(n_shuffles):
        round_maxes = []
        for er in eval_results:
            lat = er["_lat"]
            labels = er["_labels"]
            fc_dict = er["_feature_corrs"]
            if not fc_dict:
                continue
            shuf = labels.copy()
            for c in range(shuf.shape[1]):
                rng.shuffle(shuf[:, c])
            for j, _corrs in fc_dict.items():
                active = lat[:, j] != 0
                if active.sum() < 5:
                    continue
                corrs = [
                    abs(float(np.corrcoef(lat[active, j], shuf[active, k])[0, 1]))
                    if not np.isnan(np.corrcoef(lat[active, j], shuf[active, k])[0, 1])
                    else 0.0
                    for k in range(shuf.shape[1])
                ]
                if corrs:
                    round_maxes.append(max(corrs))
        if round_maxes:
            null_maxes.append(max(round_maxes))

    null_95 = float(np.percentile(null_maxes, 95)) if null_maxes else 0.0

    # Re-count families
    type_dist_null = defaultdict(int)
    for er in eval_results:
        lat = er["_lat"]
        for j, corrs in er["_feature_corrs"].items():
            active = lat[:, j] != 0
            if active.sum() < 5:
                continue
            best = int(np.argmax(corrs))
            if corrs[best] > null_95:
                type_dist_null[STATISTIC_NAMES[best]] += 1

    return null_95, len(type_dist_null)


def run_experiment(
    prepped_stocks: list[dict],
    d_model: int,
    d_hidden: int,
    k: int,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    label: str,
) -> dict:
    """Train per-stock SAEs and evaluate for a single configuration."""
    logger.info("--- %s (d_hidden=%d, k=%d) ---", label, d_hidden, k)
    eval_results = []

    for i, stock in enumerate(prepped_stocks):
        sae = TopKSAE(d_model, d_hidden, k).to(device)
        train_sae(
            sae, stock["train_acts"],
            steps=steps, batch_size=batch_size, lr=lr, grad_clip=grad_clip,
            device=device, log_interval=0,
        )
        er = evaluate_sae_config(sae, stock["test_acts"], stock["labels"], device)
        eval_results.append(er)
        del sae
        torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            logger.info("  [%d/%d] stocks done", i + 1, len(prepped_stocks))

    # Aggregate
    ve = [r["var_explained"] for r in eval_results]
    dr = [r["dead_rate"] for r in eval_results]
    ac = [r["alive_count"] for r in eval_results]
    lp = [r["largest_pct"] for r in eval_results]
    nf = [r["n_families"] for r in eval_results]

    # Null calibration
    null_95, n_fam_null = null_calibrate(eval_results)

    summary = {
        "config": label,
        "n_stocks": len(eval_results),
        "d_hidden": d_hidden,
        "k": k,
        "var_explained_mean": float(np.mean(ve)),
        "var_explained_std": float(np.std(ve)),
        "dead_rate_mean": float(np.mean(dr)),
        "dead_rate_std": float(np.std(dr)),
        "alive_count_mean": float(np.mean(ac)),
        "largest_pct_mean": float(np.mean(lp)),
        "n_families_mean": float(np.mean(nf)),
        "null_95_threshold": null_95,
        "n_families_null": n_fam_null,
    }

    logger.info(
        "  VE=%.4f, dead=%.4f, largest=%.4f, n_fam=%.1f, null_95=%.4f",
        summary["var_explained_mean"], summary["dead_rate_mean"],
        summary["largest_pct_mean"], summary["n_families_mean"],
        summary["null_95_threshold"],
    )

    return summary


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    set_seed(cfg["training"]["seed"])

    device = cfg["device"]
    data_root = Path(cfg["data"]["root"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    d_model = cfg["model"]["d_model"]
    layer = cfg["model"]["layer"]
    window_len = cfg["window"]["length"]
    stride = cfg["window"]["stride"]
    train_frac = cfg["split"]["train"]
    val_frac = cfg["split"]["val"]
    steps = cfg["training"]["steps"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["lr"]
    grad_clip = cfg["training"]["grad_clip"]

    k_values = cfg["sensitivity"]["k_values"]
    expansion_values = cfg["sensitivity"]["expansion_values"]
    n_stocks_max = cfg["sensitivity"]["n_stocks"]

    # Load model
    logger.info("Loading Kronos model...")
    model, tokenizer, model_cfg = load_kronos(
        cfg["model"]["config_path"],
        cfg["model"]["weights_path"],
        cfg["model"]["tokenizer_path"],
        device=device,
    )
    d_model = model_cfg["d_model"]

    # Pre-process stocks (extract activations + labels)
    logger.info("Pre-processing %d stocks...", n_stocks_max)
    all_csvs = sorted([f for f in os.listdir(str(data_root)) if f.endswith(".csv")])
    prepped = []

    for fname in all_csvs:
        if len(prepped) >= n_stocks_max:
            break

        raw = load_stock(str(data_root / fname))
        if raw is None:
            continue
        normed, _, _ = normalize(raw)
        windows = create_windows(normed, window_len, stride, max_windows=1500)
        if windows is None:
            continue

        train_wins, _, test_wins = split_windows(windows, train_frac, val_frac)
        n_train = len(train_wins)

        if n_train < 30 or len(test_wins) < 20:
            continue

        # Extract all activations
        all_acts = extract_activations(model, tokenizer, windows, layer=layer, device=device)
        train_acts = all_acts[:n_train]
        n_val = int(len(windows) * val_frac)
        test_acts = all_acts[n_train + n_val :]
        test_wins_actual = windows[n_train + n_val :]

        labels = compute_labels_for_windows(test_wins_actual)

        prepped.append({
            "ticker": fname.replace(".csv", ""),
            "train_acts": train_acts,
            "test_acts": test_acts,
            "labels": labels,
        })

        torch.cuda.empty_cache()

    logger.info("%d stocks pre-processed", len(prepped))

    results = {}

    # Experiment 1: k sensitivity (expansion = 4x)
    logger.info("\nEXPERIMENT 1: k sensitivity (expansion 4x)")
    exp1 = {}
    for k_val in k_values:
        d_hidden = d_model * 4
        summary = run_experiment(
            prepped, d_model, d_hidden, k_val, device,
            steps, batch_size, lr, grad_clip,
            label=f"k{k_val}_exp4x",
        )
        exp1[f"k{k_val}_exp4x"] = summary
    results["k_sensitivity"] = exp1

    # Experiment 2: expansion sensitivity (k = 64)
    logger.info("\nEXPERIMENT 2: expansion sensitivity (k=64)")
    exp2 = {}
    for exp_factor in expansion_values:
        d_hidden = d_model * exp_factor
        summary = run_experiment(
            prepped, d_model, d_hidden, 64, device,
            steps, batch_size, lr, grad_clip,
            label=f"k64_exp{exp_factor}x",
        )
        exp2[f"k64_exp{exp_factor}x"] = summary
    results["expansion_sensitivity"] = exp2

    # Save
    results["meta"] = {
        "d_model": d_model,
        "layer": layer,
        "n_stocks": len(prepped),
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
    }

    out_path = results_dir / "sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
