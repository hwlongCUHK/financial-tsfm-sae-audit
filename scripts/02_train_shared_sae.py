#!/usr/bin/env python3
"""Step 2: Train a shared SAE on pooled activations from 120 stocks.

This script implements the primary analysis from the paper (Section 3):
  1. Loads pre-extracted activations for all stocks.
  2. Pools training-split activations across all 120 stocks (~7000 windows).
  3. Trains a single shared TopK SAE (k=64, 4x expansion, 5000 steps, batch 512).
  4. Evaluates per-stock: variance explained, dead rate, alive count, ablation cosine.
  5. Saves SAE weights and per-stock evaluation results.

Paper reference: Section 3 (SAE Training -- Shared SAE).
Output: outputs/sae/shared_sae.pt                  (trained SAE weights)
        outputs/sae/shared_sae_results.json         (evaluation metrics)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sae import TopKSAE, train_sae
from src.utils import load_config, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_per_stock(
    sae: TopKSAE,
    test_acts: np.ndarray,
    device: str,
    top_k_ablate: int = 50,
) -> dict:
    """Evaluate SAE on a single stock's test activations.

    Returns variance explained, dead rate, alive count, and ablation cosine.
    """
    at = torch.from_numpy(test_acts).float().to(device)

    with torch.no_grad():
        lat_tensor = sae.encode(at)
        lat = lat_tensor.cpu().numpy()
        recon = sae.decode(lat_tensor).cpu().numpy()

    # Variance explained
    var_total = float(np.var(test_acts))
    mse = float(np.mean((recon - test_acts) ** 2))
    var_explained = float(1.0 - mse / max(var_total, 1e-10))

    # Dead features
    dead_mask = (lat != 0).sum(axis=0) == 0
    dead_rate = float(dead_mask.mean())
    alive_count = int((~dead_mask).sum())

    # Top-K ablation: cosine similarity
    freq = (lat != 0).sum(axis=0)
    top_feats = np.argsort(freq)[-top_k_ablate:]

    with torch.no_grad():
        lat_ablated = lat_tensor.clone()
        lat_ablated[:, top_feats] = 0
        recon_ablated = sae.decode(lat_ablated).cpu().numpy()

    cos_sims = []
    for i in range(len(recon)):
        norm_a = np.linalg.norm(recon_ablated[i])
        norm_b = np.linalg.norm(recon[i])
        if norm_a > 1e-10 and norm_b > 1e-10:
            cos_sims.append(np.dot(recon_ablated[i], recon[i]) / (norm_a * norm_b))
    ablation_cosine = float(np.mean(cos_sims)) if cos_sims else 0.0

    return {
        "var_explained": var_explained,
        "dead_rate": dead_rate,
        "alive_count": alive_count,
        "ablation_cosine": ablation_cosine,
    }


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    set_seed(cfg["training"]["seed"])

    device = cfg["device"]
    act_dir = Path(cfg["paths"]["activations_dir"])
    sae_dir = Path(cfg["paths"]["sae_dir"])
    sae_dir.mkdir(parents=True, exist_ok=True)

    # SAE config
    d_model = cfg["model"]["d_model"]
    expansion = cfg["sae"]["expansion"]
    k = cfg["sae"]["k"]
    d_hidden = d_model * expansion
    steps = cfg["training"]["steps_shared"]
    batch_size = cfg["training"]["batch_size_shared"]
    lr = cfg["training"]["lr"]
    grad_clip = cfg["training"]["grad_clip"]

    train_frac = cfg["split"]["train"]
    val_frac = cfg["split"]["val"]
    top_k_ablate = cfg["ablation"]["top_k_features"]

    # Load metadata
    meta_path = act_dir / "metadata.json"
    if not meta_path.exists():
        logger.error("No metadata.json found. Run 01_extract_activations.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    d_model = metadata["d_model"]
    d_hidden = d_model * expansion

    # Phase 1: Collect all training activations
    logger.info("Collecting training activations from all stocks...")
    all_train_acts = []
    per_stock_test = []  # (ticker, test_acts, n_test)

    for stock_info in metadata["stocks"]:
        ticker = stock_info["ticker"]
        act_path = act_dir / f"{ticker}_acts.npy"
        if not act_path.exists():
            continue

        acts = np.load(str(act_path))
        n_total = len(acts)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        n_test = n_total - n_train - n_val

        if n_test < 10:
            continue

        all_train_acts.append(acts[:n_train])
        per_stock_test.append({
            "ticker": ticker,
            "test_acts": acts[n_train + n_val :],
            "n_test": n_test,
        })

    if not all_train_acts:
        logger.error("No valid stocks found.")
        sys.exit(1)

    pooled_train = np.concatenate(all_train_acts, axis=0)
    n_stocks_valid = len(per_stock_test)
    logger.info(
        "%d valid stocks, %d pooled training windows, d_model=%d",
        n_stocks_valid, len(pooled_train), d_model,
    )

    # Phase 2: Train shared SAE
    logger.info(
        "Training shared SAE (k=%d, expansion=%dx, %d steps, batch %d)...",
        k, expansion, steps, batch_size,
    )
    sae = TopKSAE(d_model, d_hidden, k).to(device)
    t0 = time.time()
    train_sae(
        sae, pooled_train,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        grad_clip=grad_clip,
        device=device,
    )
    train_time = time.time() - t0
    logger.info("Training done in %.0fs", train_time)

    # Save SAE weights
    sae_path = sae_dir / "shared_sae.pt"
    torch.save(sae.state_dict(), str(sae_path))
    logger.info("Saved SAE weights to %s", sae_path)

    # Phase 3: Evaluate per stock
    logger.info("Evaluating on %d stocks...", n_stocks_valid)
    per_stock_results = []
    agg_ve, agg_dead, agg_alive, agg_cos = [], [], [], []

    for ps in per_stock_test:
        result = evaluate_per_stock(sae, ps["test_acts"], device, top_k_ablate)
        result["ticker"] = ps["ticker"]
        result["n_test"] = ps["n_test"]
        per_stock_results.append(result)

        agg_ve.append(result["var_explained"])
        agg_dead.append(result["dead_rate"])
        agg_alive.append(result["alive_count"])
        agg_cos.append(result["ablation_cosine"])

    # Aggregate
    output = {
        "config": f"k{k}_exp{expansion}x",
        "n_stocks": n_stocks_valid,
        "n_train_windows": int(len(pooled_train)),
        "train_steps": steps,
        "train_time_s": int(train_time),
        "var_explained_mean": float(np.mean(agg_ve)),
        "var_explained_std": float(np.std(agg_ve)),
        "dead_rate_mean": float(np.mean(agg_dead)),
        "dead_rate_std": float(np.std(agg_dead)),
        "alive_count_mean": float(np.mean(agg_alive)),
        "alive_count_std": float(np.std(agg_alive)),
        "ablation_cosine_mean": float(np.mean(agg_cos)),
        "ablation_cosine_std": float(np.std(agg_cos)),
        "per_stock": per_stock_results,
    }

    out_path = sae_dir / "shared_sae_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", out_path)
    logger.info(
        "VE: %.4f +/- %.4f, Dead: %.4f, Alive: %.0f, Ablation cos: %.4f",
        output["var_explained_mean"], output["var_explained_std"],
        output["dead_rate_mean"], output["alive_count_mean"],
        output["ablation_cosine_mean"],
    )


if __name__ == "__main__":
    main()
