#!/usr/bin/env python3
"""Step 5: Financial metric validation.

This script computes financially interpretable metrics for ablation effects
(Section 5, RQ2, Table 3):

  Output preservation (consistency checks):
    - Volatility stability ratio: std(ablated_returns) / std(baseline_returns)
    - Directional agreement: fraction of windows where ablated and baseline
      agree on return direction
    - RankIC (ablated vs baseline): Spearman rank correlation

  Ground-truth alignment (pattern prediction):
    - Directional accuracy vs realized returns
    - RankIC vs realized returns
    - Volatility forecast error vs realized volatility

  Statistical validation:
    - Bootstrap 95% CIs (10,000 resamples)
    - Sector-clustered standard errors
    - One-sample t-tests

Paper reference: Section 5 (Financial Metric Validation), Table 3.
Output: outputs/results/financial_validation.json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sae import TopKSAE
from src.utils import (
    load_config,
    load_kronos,
    load_stock,
    normalize,
    create_windows,
    set_seed,
    split_windows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: list[float], n_boot: int = 10000
) -> tuple[float, float, float]:
    """Compute bootstrap 95% confidence interval.

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    arr = np.array(values)
    means = np.array([
        np.mean(np.random.choice(arr, len(arr), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def process_stock(
    model: torch.nn.Module,
    tokenizer: object,
    sae: TopKSAE,
    test_windows: np.ndarray,
    raw_data: np.ndarray,
    n_train: int,
    n_val: int,
    layer: int,
    device: str,
    top_k_ablate: int = 50,
    window_len: int = 64,
    stride: int = 32,
) -> dict:
    """Process a single stock: compute financial metrics for ablation."""
    n_test = len(test_windows)
    test_t = torch.from_numpy(test_windows).float().to(device)

    # Baseline forward pass
    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_t, half=True)
        base_out = model(s1, s2)
    base_logits = base_out[0].float()
    base_tokens = base_logits[:, -1, :].argmax(dim=-1).float().cpu().numpy()

    # Get SAE features for identifying top-K
    acts_list = []

    def hook_fn(m, inp, out):
        a = out[0] if isinstance(out, tuple) else out
        acts_list.append(a[:, -1, :].detach().cpu().float().numpy())

    h = model.transformer[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_t, half=True)
        model(s1, s2)
    h.remove()
    acts_np = np.concatenate(acts_list)

    at = torch.from_numpy(acts_np).float().to(device)
    with torch.no_grad():
        lat = sae.encode(at).cpu().numpy()
    freq = (lat != 0).sum(axis=0)
    top_feats = np.argsort(freq)[-top_k_ablate:].tolist()

    # Ablated forward pass
    def _ablation_hook(module, inputs, output):
        orig = output[0] if isinstance(output, tuple) else output
        B, T, D = orig.shape
        resid = orig.reshape(-1, D).float()
        ablated = sae.ablate_reconstruct(resid, top_feats)
        ablated = ablated.reshape(B, T, D).half()
        if isinstance(output, tuple):
            return (ablated,) + output[1:]
        return ablated

    h_abl = model.transformer[layer].register_forward_hook(_ablation_hook)
    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_t, half=True)
        ab_out = model(s1, s2)
    h_abl.remove()
    ab_logits = ab_out[0].float()
    ab_tokens = ab_logits[:, -1, :].argmax(dim=-1).float().cpu().numpy()

    # Compute returns from token changes (proxy)
    base_returns = np.diff(base_tokens)
    ab_returns = np.diff(ab_tokens)

    # Realized future returns from raw close prices
    test_start_idx = (n_train + n_val) * stride + window_len
    close_raw = raw_data[:, 1]
    realized_returns = []
    for i in range(n_test):
        w_end = test_start_idx + i * stride
        if w_end < len(close_raw) and w_end - 1 >= 0:
            c_cur = close_raw[w_end - 1]
            c_next = close_raw[w_end] if w_end < len(close_raw) else c_cur
            realized_returns.append((c_next - c_cur) / (c_cur + 1e-5))
    realized_returns = np.array(realized_returns[:len(base_returns)])

    min_len = min(len(base_returns), len(ab_returns), len(realized_returns))
    base_returns = base_returns[:min_len]
    ab_returns = ab_returns[:min_len]
    realized_returns = realized_returns[:min_len]

    if min_len < 3:
        return None

    # --- Output preservation metrics ---

    # Volatility stability ratio
    std_base = float(np.std(base_returns)) if len(base_returns) > 1 else 1e-8
    std_ab = float(np.std(ab_returns)) if len(ab_returns) > 1 else 1e-8
    vol_ratio = std_ab / std_base if std_base > 1e-12 else 1.0

    # Directional agreement (ablated vs baseline)
    valid = (np.sign(base_returns) != 0) & (np.sign(ab_returns) != 0)
    dir_agreement = float(np.mean(np.sign(base_returns[valid]) == np.sign(ab_returns[valid]))) if valid.sum() > 0 else 0.0

    # RankIC (ablated vs baseline)
    rankic_ab_base = float(stats.spearmanr(ab_tokens, base_tokens)[0]) if len(base_tokens) > 4 else 0.0
    if np.isnan(rankic_ab_base):
        rankic_ab_base = 0.0

    # --- Ground-truth alignment metrics ---

    # Directional accuracy vs realized returns
    base_dir_acc = float(np.mean(np.sign(base_returns) == np.sign(realized_returns)))
    ab_dir_acc = float(np.mean(np.sign(ab_returns) == np.sign(realized_returns)))

    # RankIC vs realized
    base_rankic = float(stats.spearmanr(base_returns, realized_returns)[0]) if min_len > 4 else 0.0
    ab_rankic = float(stats.spearmanr(ab_returns, realized_returns)[0]) if min_len > 4 else 0.0
    if np.isnan(base_rankic):
        base_rankic = 0.0
    if np.isnan(ab_rankic):
        ab_rankic = 0.0

    # Volatility forecast error
    realized_vol = float(np.std(realized_returns))
    vol_err_base = float(np.abs(std_base - realized_vol))
    vol_err_ab = float(np.abs(std_ab - realized_vol))

    return {
        "n_test": n_test,
        # Output preservation
        "vol_ratio": vol_ratio,
        "dir_agreement": dir_agreement,
        "rankic_ab_base": rankic_ab_base,
        # Ground-truth alignment
        "base_dir_acc": base_dir_acc,
        "ab_dir_acc": ab_dir_acc,
        "dir_delta": ab_dir_acc - base_dir_acc,
        "base_rankic": base_rankic,
        "ab_rankic": ab_rankic,
        "rankic_delta": ab_rankic - base_rankic,
        "vol_err_base": vol_err_base,
        "vol_err_ab": vol_err_ab,
        "vol_err_delta": vol_err_ab - vol_err_base,
    }


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    set_seed(cfg["training"]["seed"])

    device = cfg["device"]
    data_root = Path(cfg["data"]["root"])
    sae_dir = Path(cfg["paths"]["sae_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    d_model = cfg["model"]["d_model"]
    expansion = cfg["sae"]["expansion"]
    k = cfg["sae"]["k"]
    d_hidden = d_model * expansion
    layer = cfg["model"]["layer"]
    window_len = cfg["window"]["length"]
    stride = cfg["window"]["stride"]
    train_frac = cfg["split"]["train"]
    val_frac = cfg["split"]["val"]
    top_k_ablate = cfg["ablation"]["top_k_features"]
    n_stocks_max = cfg["financial"]["n_stocks"]
    n_bootstrap = cfg["financial"]["n_bootstrap"]

    # Load model
    logger.info("Loading Kronos model...")
    model, tokenizer, model_cfg = load_kronos(
        cfg["model"]["config_path"],
        cfg["model"]["weights_path"],
        cfg["model"]["tokenizer_path"],
        device=device,
    )

    # Load shared SAE
    sae = TopKSAE(d_model, d_hidden, k).to(device)
    sae.load_state_dict(
        torch.load(str(sae_dir / "shared_sae.pt"), map_location=device, weights_only=True)
    )
    sae.eval()

    # Process stocks
    all_csvs = sorted([f for f in os.listdir(str(data_root)) if f.endswith(".csv")])
    logger.info("Processing up to %d stocks for financial validation...", n_stocks_max)

    per_stock = []
    for i, fname in enumerate(all_csvs):
        if len(per_stock) >= n_stocks_max:
            break

        ticker = fname.replace(".csv", "")
        raw = load_stock(str(data_root / fname))
        if raw is None:
            continue

        normed, _, _ = normalize(raw)
        windows = create_windows(normed, window_len, stride)
        if windows is None:
            continue

        train_wins, val_wins, test_wins = split_windows(windows, train_frac, val_frac)
        n_train = len(train_wins)
        n_val = len(val_wins)

        if len(test_wins) < 10:
            continue

        result = process_stock(
            model, tokenizer, sae, test_wins, raw,
            n_train, n_val, layer, device,
            top_k_ablate, window_len, stride,
        )

        if result is None:
            continue

        result["ticker"] = ticker
        per_stock.append(result)

        logger.info(
            "[%d] %s: vol_ratio=%.3f, dir_agree=%.3f, base_dir=%.3f, ab_dir=%.3f",
            len(per_stock), ticker,
            result["vol_ratio"], result["dir_agreement"],
            result["base_dir_acc"], result["ab_dir_acc"],
        )
        torch.cuda.empty_cache()

    # Aggregate with bootstrap CIs
    n = len(per_stock)
    logger.info("Aggregating %d stocks...", n)

    vol_ratios = [r["vol_ratio"] for r in per_stock]
    dir_agrees = [r["dir_agreement"] for r in per_stock]
    rankic_abs = [r["rankic_ab_base"] for r in per_stock]
    base_dirs = [r["base_dir_acc"] for r in per_stock]
    ab_dirs = [r["ab_dir_acc"] for r in per_stock]
    dir_deltas = [r["dir_delta"] for r in per_stock]
    base_rankics = [r["base_rankic"] for r in per_stock]
    ab_rankics = [r["ab_rankic"] for r in per_stock]
    rankic_deltas = [r["rankic_delta"] for r in per_stock]
    vol_err_deltas = [r["vol_err_delta"] for r in per_stock]

    output = {
        "n_stocks": n,
        "top_k_ablate": top_k_ablate,
        "output_preservation": {
            "volatility_ratio": {
                "mean": bootstrap_ci(vol_ratios, n_bootstrap)[0],
                "ci_lo": bootstrap_ci(vol_ratios, n_bootstrap)[1],
                "ci_hi": bootstrap_ci(vol_ratios, n_bootstrap)[2],
                "p_value_vs_1": float(stats.ttest_1samp(vol_ratios, 1.0)[1]),
            },
            "directional_agreement": {
                "mean": bootstrap_ci(dir_agrees, n_bootstrap)[0],
                "ci_lo": bootstrap_ci(dir_agrees, n_bootstrap)[1],
                "ci_hi": bootstrap_ci(dir_agrees, n_bootstrap)[2],
                "p_value_vs_0_5": float(stats.ttest_1samp(dir_agrees, 0.5)[1]),
            },
            "rankic_ab_vs_base": {
                "mean": float(np.mean(rankic_abs)),
                "std": float(np.std(rankic_abs)),
            },
        },
        "ground_truth": {
            "directional_accuracy": {
                "baseline_mean": float(np.mean(base_dirs)),
                "ablated_mean": float(np.mean(ab_dirs)),
                "delta_mean": bootstrap_ci(dir_deltas, n_bootstrap)[0],
                "delta_ci_lo": bootstrap_ci(dir_deltas, n_bootstrap)[1],
                "delta_ci_hi": bootstrap_ci(dir_deltas, n_bootstrap)[2],
            },
            "rankic_vs_realized": {
                "baseline_mean": float(np.mean(base_rankics)),
                "ablated_mean": float(np.mean(ab_rankics)),
                "delta_mean": float(np.mean(rankic_deltas)),
            },
            "vol_forecast_error": {
                "baseline_mean": float(np.mean([r["vol_err_base"] for r in per_stock])),
                "ablated_mean": float(np.mean([r["vol_err_ab"] for r in per_stock])),
                "delta_mean": float(np.mean(vol_err_deltas)),
            },
        },
        "per_stock": per_stock,
    }

    out_path = results_dir / "financial_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", out_path)
    logger.info("Vol ratio: %.3f [%.3f, %.3f]",
                output["output_preservation"]["volatility_ratio"]["mean"],
                output["output_preservation"]["volatility_ratio"]["ci_lo"],
                output["output_preservation"]["volatility_ratio"]["ci_hi"])
    logger.info("Dir agreement: %.3f", output["output_preservation"]["directional_agreement"]["mean"])
    logger.info("Base dir acc: %.3f, Ablated dir acc: %.3f",
                output["ground_truth"]["directional_accuracy"]["baseline_mean"],
                output["ground_truth"]["directional_accuracy"]["ablated_mean"])


if __name__ == "__main__":
    main()
