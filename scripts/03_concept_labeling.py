#!/usr/bin/env python3
"""Step 3: Statistics-first feature labeling (concept distribution analysis).

This script implements the concept labeling pipeline from the paper (Section 3
and RQ1, Section 4):

  1. Loads the shared SAE and per-stock test activations.
  2. Computes 16 financial statistics per window.
  3. For each alive SAE feature, correlates its activation magnitude with each
     statistic. Assigns the feature to the statistic with highest |r|.
  4. Reports the concept distribution at three tiers:
       - Tier 1: |r| > 0.15  (concept discovery)
       - Tier 2: |r| > 0.35 and |r| > 0.50  (robustness check)
  5. Performs null calibration via label permutation (100 shuffles, 95th pct).
  6. Performs block-permutation calibration (block size = 10 windows).

Paper reference: Section 3 (Labeling), Section 4 (RQ1), Table 1.
Output: outputs/results/concept_labeling.json
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

from src.sae import TopKSAE
from src.statistics import STATISTIC_NAMES, compute_labels_for_windows
from src.utils import (
    load_config,
    load_stock,
    normalize,
    create_windows,
    set_seed,
    split_windows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def label_features(
    lat: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.15,
    min_active: int = 5,
) -> tuple[dict, dict]:
    """Label SAE features by their strongest-correlated statistic.

    Args:
        lat: Latent activations, shape (n_samples, d_hidden).
        labels: Financial statistics, shape (n_samples, n_stats).
        threshold: Minimum |r| for assignment.
        min_active: Minimum non-zero activations for a feature to be considered.

    Returns:
        Tuple of (type_distribution, feature_correlations).
    """
    alive_mask = (lat != 0).sum(axis=0) > min_active
    type_dist = defaultdict(int)
    feature_corrs = {}

    for j in np.where(alive_mask)[0]:
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

    return dict(type_dist), feature_corrs


def null_calibrate(
    per_stock_data: list[dict],
    n_shuffles: int = 100,
    seed: int = 42,
) -> tuple[dict, float]:
    """Permutation-based null calibration across stocks.

    For each shuffle round, label columns are shuffled independently within
    each stock. The 95th percentile of per-round maximum |r| values gives
    the null-calibrated threshold.

    Returns:
        Tuple of (null_type_dist, null_95_threshold).
    """
    rng = np.random.RandomState(seed)
    null_maxes = []

    for _ in range(n_shuffles):
        round_maxes = []
        for stock_data in per_stock_data:
            lat = stock_data["lat"]
            labels = stock_data["labels"]
            feature_corrs = stock_data["feature_corrs"]

            # Shuffle each label column independently
            shuf = labels.copy()
            for c in range(shuf.shape[1]):
                rng.shuffle(shuf[:, c])

            for j, _corrs in feature_corrs.items():
                active = lat[:, j] != 0
                if active.sum() < 5:
                    continue
                corrs = []
                for k_stat in range(shuf.shape[1]):
                    cc = np.corrcoef(lat[active, j], shuf[active, k_stat])[0, 1]
                    corrs.append(0.0 if np.isnan(cc) else abs(float(cc)))
                if corrs:
                    round_maxes.append(max(corrs))

        if round_maxes:
            null_maxes.append(max(round_maxes))

    null_95 = float(np.percentile(null_maxes, 95)) if null_maxes else 0.0

    # Re-count with null-calibrated threshold
    type_dist_null = defaultdict(int)
    for stock_data in per_stock_data:
        lat = stock_data["lat"]
        for j, corrs in stock_data["feature_corrs"].items():
            active = lat[:, j] != 0
            if active.sum() < 5:
                continue
            best_idx = int(np.argmax(corrs))
            if corrs[best_idx] > null_95:
                type_dist_null[STATISTIC_NAMES[best_idx]] += 1

    return dict(type_dist_null), null_95


def block_permutation_null(
    per_stock_data: list[dict],
    block_size: int = 10,
    n_shuffles: int = 50,
    seed: int = 42,
) -> tuple[float, float]:
    """Block-permutation null calibration.

    Shuffles labels in blocks of consecutive windows rather than independently,
    preserving temporal autocorrelation structure.

    Returns:
        Tuple of (block_null_95, standard_null_95) for comparison.
    """
    rng = np.random.RandomState(seed)

    def _block_shuffle(arr: np.ndarray) -> np.ndarray:
        n = len(arr)
        n_blocks = n // block_size
        blocks = [arr[i * block_size : (i + 1) * block_size].copy() for i in range(n_blocks)]
        remainder = arr[n_blocks * block_size :] if n_blocks * block_size < n else None
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        shuffled = np.concatenate([blocks[i] for i in indices], axis=0)
        if remainder is not None and len(remainder) > 0:
            shuffled = np.concatenate([shuffled, remainder], axis=0)
        return shuffled

    block_null_maxes = []
    std_null_maxes = []

    for _ in range(n_shuffles):
        block_round, std_round = [], []

        for stock_data in per_stock_data:
            lat = stock_data["lat"]
            labels = stock_data["labels"]

            # Block shuffle
            shuf_block = _block_shuffle(labels)
            # Standard shuffle
            shuf_std = labels.copy()
            for c in range(shuf_std.shape[1]):
                rng.shuffle(shuf_std[:, c])

            for j in stock_data["feature_corrs"]:
                active = lat[:, j] != 0
                if active.sum() < 5:
                    continue

                # Block
                corrs_b = [
                    abs(float(np.corrcoef(lat[active, j], shuf_block[active, k])[0, 1]))
                    if not np.isnan(np.corrcoef(lat[active, j], shuf_block[active, k])[0, 1])
                    else 0.0
                    for k in range(labels.shape[1])
                ]
                block_round.append(max(corrs_b))

                # Standard
                corrs_s = [
                    abs(float(np.corrcoef(lat[active, j], shuf_std[active, k])[0, 1]))
                    if not np.isnan(np.corrcoef(lat[active, j], shuf_std[active, k])[0, 1])
                    else 0.0
                    for k in range(labels.shape[1])
                ]
                std_round.append(max(corrs_s))

        if block_round:
            block_null_maxes.append(max(block_round))
        if std_round:
            std_null_maxes.append(max(std_round))

    block_95 = float(np.percentile(block_null_maxes, 95)) if block_null_maxes else 0.0
    std_95 = float(np.percentile(std_null_maxes, 95)) if std_null_maxes else 0.0
    return block_95, std_95


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    set_seed(cfg["training"]["seed"])

    device = cfg["device"]
    data_root = Path(cfg["data"]["root"])
    act_dir = Path(cfg["paths"]["activations_dir"])
    sae_dir = Path(cfg["paths"]["sae_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    d_model = cfg["model"]["d_model"]
    expansion = cfg["sae"]["expansion"]
    k = cfg["sae"]["k"]
    d_hidden = d_model * expansion
    window_len = cfg["window"]["length"]
    stride = cfg["window"]["stride"]
    train_frac = cfg["split"]["train"]
    val_frac = cfg["split"]["val"]

    thresholds = [cfg["labeling"]["pre_cal_threshold"]] + cfg["labeling"]["tier2_thresholds"]
    n_shuffles = cfg["labeling"]["null_shuffles"]
    block_size = cfg["labeling"]["block_size"]
    min_active = cfg["labeling"]["min_active_samples"]

    # Load shared SAE
    sae_path = sae_dir / "shared_sae.pt"
    if not sae_path.exists():
        logger.error("Shared SAE not found. Run 02_train_shared_sae.py first.")
        sys.exit(1)

    sae = TopKSAE(d_model, d_hidden, k).to(device)
    sae.load_state_dict(torch.load(str(sae_path), map_location=device, weights_only=True))
    sae.eval()
    logger.info("Loaded shared SAE from %s", sae_path)

    # Load metadata
    with open(act_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Process each stock
    logger.info("Computing concept labels for each stock...")
    per_stock_data = []
    type_dist_all = {t: defaultdict(int) for t in thresholds}

    for si, stock_info in enumerate(metadata["stocks"]):
        ticker = stock_info["ticker"]
        act_path = act_dir / f"{ticker}_acts.npy"
        csv_path = data_root / f"{ticker}.csv"

        if not act_path.exists() or not csv_path.exists():
            continue

        acts = np.load(str(act_path))
        n_total = len(acts)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        test_acts = acts[n_train + n_val :]

        if len(test_acts) < 10:
            continue

        # Get test window labels
        raw = load_stock(str(csv_path))
        if raw is None:
            continue
        normed, _, _ = normalize(raw)
        windows = create_windows(normed, window_len, stride)
        if windows is None:
            continue
        _, _, test_wins = split_windows(windows, train_frac, val_frac)
        test_wins = test_wins[: len(test_acts)]

        labels = compute_labels_for_windows(test_wins)

        # Encode through SAE
        at = torch.from_numpy(test_acts).float().to(device)
        with torch.no_grad():
            lat = sae.encode(at).cpu().numpy()

        # Label features at each threshold
        for threshold in thresholds:
            td, fc = label_features(lat, labels, threshold=threshold, min_active=min_active)
            for concept, count in td.items():
                type_dist_all[threshold][concept] += count

        # Store for null calibration (only at Tier-1 threshold)
        _, feature_corrs = label_features(
            lat, labels, threshold=thresholds[0], min_active=min_active
        )
        per_stock_data.append({
            "ticker": ticker,
            "lat": lat,
            "labels": labels,
            "feature_corrs": feature_corrs,
        })

        if (si + 1) % 20 == 0:
            logger.info("[%d/%d] stocks processed", si + 1, len(metadata["stocks"]))

    logger.info("Processed %d valid stocks", len(per_stock_data))

    # Build tier results
    tier_results = {}
    for threshold in thresholds:
        dist = dict(type_dist_all[threshold])
        total = sum(dist.values())
        pct = {
            concept: round(count / max(total, 1) * 100, 1)
            for concept, count in sorted(dist.items(), key=lambda x: -x[1])
        }
        largest_pct = max(dist.values()) / max(total, 1) if dist else 0.0
        n_families = len(dist)

        tier_results[str(threshold)] = {
            "threshold": threshold,
            "total_assignments": total,
            "n_families": n_families,
            "largest_family_pct": round(largest_pct * 100, 1),
            "distribution_pct": pct,
            "distribution_counts": dist,
        }
        logger.info(
            "Tier |r|>%.2f: %d assignments, %d families, largest=%.1f%%",
            threshold, total, n_families, largest_pct * 100,
        )

    # Null calibration
    logger.info("Running null calibration (%d shuffles)...", n_shuffles)
    null_dist, null_95 = null_calibrate(per_stock_data, n_shuffles=n_shuffles)
    null_total = sum(null_dist.values())
    logger.info(
        "Null-calibrated threshold: %.4f, %d families, %d assignments",
        null_95, len(null_dist), null_total,
    )

    # Block permutation
    logger.info("Running block-permutation calibration (block=%d)...", block_size)
    block_95, std_95 = block_permutation_null(
        per_stock_data, block_size=block_size, n_shuffles=50
    )
    inflation = block_95 / std_95 if std_95 > 0 else 1.0
    logger.info(
        "Block null_95=%.4f, Standard null_95=%.4f, inflation=%.3f",
        block_95, std_95, inflation,
    )

    # Save results
    output = {
        "n_stocks": len(per_stock_data),
        "statistic_names": STATISTIC_NAMES,
        "tier_results": tier_results,
        "null_calibration": {
            "n_shuffles": n_shuffles,
            "null_95_threshold": null_95,
            "n_families": len(null_dist),
            "total_assignments": null_total,
            "distribution": null_dist,
        },
        "block_permutation": {
            "block_size": block_size,
            "block_null_95": block_95,
            "standard_null_95": std_95,
            "inflation_factor": inflation,
        },
    }

    out_path = results_dir / "concept_labeling.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
