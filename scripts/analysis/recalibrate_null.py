#!/usr/bin/env python3
"""Recalibrate null thresholds separately for SAE, PCA, and Random bases.

For each basis, we:
1. Compute feature-activation correlations with 30+ financial statistics
2. Shuffle labels 100 times to build a null distribution of max |r|
3. The 95th percentile of the null max |r| becomes that basis's own threshold
4. Report: null threshold, % features above threshold, max |r| above threshold

This ensures a FAIR comparison: SAE vs PCA vs Random each judged by their own
null-calibrated threshold, not by a shared threshold that may favor one method.
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Financial statistics computation (30+ metrics)
# ---------------------------------------------------------------------------

def compute_financial_statistics(windows: np.ndarray, prices_raw: np.ndarray) -> dict:
    """Compute 30+ financial statistics for each time window.

    windows: (n_windows, lookback, 6) with cols [open, close, high, low, volume, amount]
    prices_raw: (total_rows, 6) raw price data (same indexing as windows source)

    Returns dict mapping stat_name -> (n_windows,) float array
    """
    n_wins, lookback, _ = windows.shape
    stats = {}

    # Price columns: close=1 (primary), high=2, low=3, open=0
    close = windows[:, :, 1]  # (n_wins, lookback)
    high = windows[:, :, 2]
    low = windows[:, :, 3]
    open_p = windows[:, :, 0]
    volume = windows[:, :, 4]
    amount = windows[:, :, 5]

    # --- Returns ---
    returns = np.diff(close, axis=1) / (close[:, :-1] + 1e-8)  # (n_wins, lookback-1)
    log_returns = np.log(close[:, 1:] / (close[:, :-1] + 1e-8))

    # 1. Mean return
    stats["mean_return"] = np.mean(returns, axis=1)

    # 2. Volatility (std of returns)
    stats["volatility"] = np.std(returns, axis=1)

    # 3. Sharpe ratio (annualized proxy)
    stats["sharpe"] = stats["mean_return"] / (stats["volatility"] + 1e-8)

    # 4. Skewness
    stats["skewness"] = np.array([float(pd.Series(r).skew()) for r in returns])

    # 5. Kurtosis
    stats["kurtosis"] = np.array([float(pd.Series(r).kurtosis()) for r in returns])

    # 6. Max return
    stats["max_return"] = np.max(returns, axis=1)

    # 7. Min return
    stats["min_return"] = np.min(returns, axis=1)

    # 8. Range of returns
    stats["return_range"] = stats["max_return"] - stats["min_return"]

    # 9. Cumulative return over window
    stats["cum_return"] = close[:, -1] / (close[:, 0] + 1e-8) - 1.0

    # 10. Price range (high-low) / close
    price_range = (high - low) / (close + 1e-8)
    stats["price_range_mean"] = np.mean(price_range, axis=1)
    stats["price_range_max"] = np.max(price_range, axis=1)

    # 11. Volume statistics
    stats["volume_mean"] = np.mean(volume, axis=1)
    stats["volume_std"] = np.std(volume, axis=1)
    stats["volume_trend"] = np.array([np.polyfit(np.arange(lookback), v, 1)[0] for v in volume])

    # 12. Volume-price correlation
    stats["volume_price_corr"] = np.array([np.corrcoef(v, c)[0, 1] if np.std(v) > 0 and np.std(c) > 0 else 0.0
                                            for v, c in zip(volume, close)])

    # 13. Turnover / amount stats
    stats["amount_mean"] = np.mean(amount, axis=1)
    stats["amount_std"] = np.std(amount, axis=1)

    # 14. Volatility clustering (autocorrelation of absolute returns)
    abs_returns = np.abs(returns)
    stats["vol_clustering"] = np.array([
        np.corrcoef(abs_returns[i, :-1], abs_returns[i, 1:])[0, 1]
        if np.std(abs_returns[i]) > 1e-8 else 0.0
        for i in range(n_wins)
    ])

    # 15. Autocorrelation of returns (lag 1, 2, 3)
    for lag in [1, 2, 3]:
        r_lag = np.array([
            np.corrcoef(returns[i, :-lag], returns[i, lag:])[0, 1]
            if len(returns[i]) > lag and np.std(returns[i, :-lag]) > 1e-8 else 0.0
            for i in range(n_wins)
        ])
        stats[f"autocorr_lag{lag}"] = r_lag

    # 16. Trend (slope of linear fit to close prices)
    x_arr = np.arange(lookback)
    stats["trend_slope"] = np.array([np.polyfit(x_arr, c, 1)[0] for c in close])

    # 17. R^2 of linear trend fit
    stats["trend_r2"] = np.array([
        np.corrcoef(x_arr, c)[0, 1] ** 2 for c in close
    ])

    # 18. Maximum drawdown
    drawdowns = np.zeros(n_wins)
    for i in range(n_wins):
        peak = np.maximum.accumulate(close[i])
        dd = (peak - close[i]) / (peak + 1e-8)
        drawdowns[i] = np.max(dd)
    stats["max_drawdown"] = drawdowns

    # 19. Average drawdown
    avg_dd = np.zeros(n_wins)
    for i in range(n_wins):
        peak = np.maximum.accumulate(close[i])
        dd = (peak - close[i]) / (peak + 1e-8)
        avg_dd[i] = np.mean(dd)
    stats["avg_drawdown"] = avg_dd

    # 20. Positive return ratio
    stats["pos_ratio"] = np.mean(returns > 0, axis=1)

    # 21. Up/down volatility
    up_vol = np.zeros(n_wins)
    down_vol = np.zeros(n_wins)
    for i in range(n_wins):
        r_i = returns[i]
        up_vol[i] = np.std(r_i[r_i > 0]) if np.any(r_i > 0) else 0.0
        down_vol[i] = np.std(r_i[r_i < 0]) if np.any(r_i < 0) else 0.0
    stats["up_volatility"] = up_vol
    stats["down_volatility"] = down_vol

    # 22. Volatility of volatility (std of rolling std)
    vol_of_vol = np.zeros(n_wins)
    for i in range(n_wins):
        rolling_std = np.array([np.std(returns[i, max(0, j-5):j+1]) for j in range(returns.shape[1])])
        vol_of_vol[i] = np.std(rolling_std)
    stats["vol_of_vol"] = vol_of_vol

    # 23. Hurst exponent (rough estimate via R/S)
    hurst = np.zeros(n_wins)
    for i in range(n_wins):
        c_i = close[i]
        if len(c_i) > 4:
            # Simplified R/S
            lags = np.arange(2, min(21, len(c_i)))
            rs = np.zeros(len(lags))
            for j, lag in enumerate(lags):
                chunks = len(c_i) // lag
                if chunks < 2:
                    break
                r_vals = np.zeros(chunks)
                for k in range(chunks):
                    chunk = c_i[k*lag:(k+1)*lag]
                    r_vals[k] = (np.max(chunk) - np.min(chunk)) / (np.std(chunk) + 1e-8)
                if len(np.unique(np.log(lags[j]))) > 0:
                    rs[j] = np.mean(r_vals)
            valid = rs > 0
            if np.sum(valid) >= 3:
                hurst[i] = np.polyfit(np.log(lags[valid]), np.log(rs[valid]), 1)[0]
    stats["hurst"] = hurst

    # 24. Up/down capture ratio
    stats["up_capture"] = up_vol / (stats["volatility"] + 1e-8)
    stats["down_capture"] = down_vol / (stats["volatility"] + 1e-8)

    # 25. Gap (open vs previous close)
    if n_wins > 0:
        gap = (open_p[:, 1:] - close[:, :-1]) / (close[:, :-1] + 1e-8)
        stats["gap_mean"] = np.mean(gap, axis=1)
        stats["gap_std"] = np.std(gap, axis=1)
        stats["gap_max"] = np.max(np.abs(gap), axis=1)

    # 26. Realized variance
    stats["realized_var"] = np.sum(returns ** 2, axis=1)

    # 27. High-low spread
    stats["hl_spread"] = np.mean((high - low) / (close + 1e-8), axis=1)

    # 28. Open-close similarity
    stats["oc_corr"] = np.array([np.corrcoef(o, c)[0, 1] for o, c in zip(open_p, close)])

    # 29. Volume bursts (max volume / mean volume)
    stats["volume_burst"] = np.max(volume, axis=1) / (np.mean(volume, axis=1) + 1e-8)

    # 30. Signed volume (volume[t] paired with return[t])
    signed_vol = volume[:, :-1] * np.sign(returns[:, :])
    stats["signed_vol_mean"] = np.mean(signed_vol, axis=1)

    # 31. Close location within daily range
    close_loc = (close - low) / (high - low + 1e-8)
    stats["close_location"] = np.mean(close_loc, axis=1)

    # 32. Percent changes (5-period, 10-period)
    for period in [5, 10]:
        if lookback > period:
            pct = close[:, period:] / (close[:, :-period] + 1e-8) - 1.0
            stats[f"pct_{period}d"] = pct[:, -1]  # most recent

    # 33. Bollinger band position
    ma20 = np.array([np.convolve(c, np.ones(10)/10, mode="valid") for c in close])
    if ma20.shape[1] > 0:
        close_aligned = close[:, 9:]
        std20 = np.array([np.array([np.std(close[i, max(0, j-9):j+1]) for j in range(9, lookback)])
                          for i in range(n_wins)])
        bb_pos = (close_aligned[:, -1] - ma20[:, -1]) / (std20[:, -1] + 1e-8)
        stats["bb_position"] = bb_pos

    # 34. Information coefficient proxy: correlation between consecutive returns in adjacent windows
    # (not computed here - needs cross-window data)

    # Clean up NaNs
    for key in list(stats.keys()):
        arr = stats[key]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip extreme values
        arr = np.clip(arr, -10, 10)
        stats[key] = arr.astype(np.float32)

    return stats


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_kronos_model(device: str):
    """Load Kronos model and tokenizer, return (tokenizer, model, n_layers, d_model)."""
    sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
    from model.kronos import Kronos, KronosTokenizer

    print("Loading KronosTokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("/data/houwanlong/models/Kronos-Tokenizer-base")
    tokenizer = tokenizer.to(device).eval()

    print("Loading Kronos model...")
    with open("/data/houwanlong/models/Kronos-base/config.json") as f:
        cfg = json.load(f)

    model = Kronos(
        s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"],
        n_layers=cfg["n_layers"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
        ffn_dropout_p=cfg["ffn_dropout_p"], attn_dropout_p=cfg["attn_dropout_p"],
        resid_dropout_p=cfg["resid_dropout_p"], token_dropout_p=cfg["token_dropout_p"],
        learn_te=cfg["learn_te"],
    )

    state_dict = load_safetensors("/data/houwanlong/models/Kronos-base/model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).half().eval()

    n_layers = len(model.transformer)
    d_model = cfg["d_model"]
    return tokenizer, model, n_layers, d_model


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_residual_stream(
    tokenizer, model, layer_idx: int, windows: np.ndarray,
    batch_size: int, device: str
) -> np.ndarray:
    """Extract residual stream activations for a specific layer.

    Returns: (n_windows, d_model) array — mean-pooled across sequence.
    """
    acts = []

    def hook_fn(mod, inp, out):
        if isinstance(out, tuple):
            act = out[0]
        else:
            act = out
        # Mean over sequence dimension
        acts.append(act.detach().cpu().float().mean(dim=1).numpy())

    hook = model.transformer[layer_idx].register_forward_hook(hook_fn)

    n_windows = windows.shape[0]
    n_batches = (n_windows + batch_size - 1) // batch_size

    with torch.no_grad():
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_windows)
            batch = torch.from_numpy(windows[start:end]).float().to(device)
            # We only need the last batch_size samples, pad if needed
            if batch.shape[0] < batch_size:
                pad = torch.zeros(batch_size - batch.shape[0], *batch.shape[1:],
                                  dtype=batch.dtype, device=device)
                batch = torch.cat([batch, pad], dim=0)

            s1_ids, s2_ids = tokenizer.encode(batch, half=True)
            _ = model(s1_ids, s2_ids)

    hook.remove()

    acts_arr = np.concatenate(acts, axis=0)[:n_windows]
    return acts_arr.astype(np.float32)


# ---------------------------------------------------------------------------
# SAE encoding
# ---------------------------------------------------------------------------

def encode_sae(activations: np.ndarray, sae_path: str) -> np.ndarray:
    """Encode activations through the SAE to get feature activations.

    activations: (n_samples, d_model)
    Returns: (n_samples, n_features)
    """
    sd = torch.load(sae_path, map_location="cpu", weights_only=True)

    w_enc = sd["encoder.weight"].numpy()   # (n_features, d_model)
    b_enc = sd["encoder.bias"].numpy()      # (n_features,)
    b_pre = sd["b_pre"].numpy()             # (d_model,)

    # Center and encode
    x_centered = activations - b_pre[None, :]  # (n, d_model)
    pre_act = x_centered @ w_enc.T + b_enc[None, :]  # (n, n_features)
    features = np.maximum(0, pre_act)  # ReLU

    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Random orthogonal basis
# ---------------------------------------------------------------------------

def random_orthogonal_basis(d_model: int, k: int, seed: int = 42) -> np.ndarray:
    """Generate k random orthogonal directions in R^d_model.

    Returns: (k, d_model) with orthonormal rows.
    """
    rng = np.random.RandomState(seed)
    A = rng.randn(d_model, k)  # (d_model, k)
    Q, _ = np.linalg.qr(A)     # (d_model, k)
    return Q.T.astype(np.float32)  # (k, d_model)


# ---------------------------------------------------------------------------
# Null threshold calibration via label shuffling
# ---------------------------------------------------------------------------

def calibrate_null_threshold(
    basis_features: np.ndarray,     # (n_samples, k_features)
    stat_names: list,
    stat_arrays: dict,              # stat_name -> (n_samples,)
    n_shuffles: int = 100,
    percentile: float = 95.0,
    seed: int = 42,
) -> dict:
    """Calibrate null threshold for a specific basis via label shuffling.

    For each shuffle:
      1. Shuffle stat labels independently
      2. Compute Pearson |r| between each feature and each stat
      3. Record the maximum |r| across all (feature, stat) pairs

    The null threshold = <percentile>th percentile of these max |r| values.

    Returns:
      null_threshold: the calibrated threshold
      above_threshold_pct: percentage of (feature, stat) pairs with |r| > threshold
      above_threshold: list of (feature_idx, stat_name, |r|) above threshold
      null_distribution: list of max |r| values from shuffles
      observed_max: the max |r| in the observed (unshuffled) data
    """
    rng = np.random.RandomState(seed)
    n_samples, k_features = basis_features.shape
    n_stats = len(stat_names)

    # Build stat matrix
    Y = np.column_stack([stat_arrays[name] for name in stat_names])  # (n_samples, n_stats)

    # --- Observed correlations ---
    # Standardize
    X_std = (basis_features - basis_features.mean(axis=0)) / (basis_features.std(axis=0) + 1e-8)
    Y_std = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)

    obs_corr = (X_std.T @ Y_std) / (n_samples - 1)  # (k_features, n_stats)
    obs_corr = np.abs(obs_corr)
    obs_max = obs_corr.max()
    obs_mean_max = np.max(np.abs(obs_corr), axis=0).mean()  # mean max |r| per stat

    # --- Null distribution via shuffling ---
    null_maxima = np.zeros(n_shuffles)

    for s in range(n_shuffles):
        Y_shuf = Y.copy()
        # Shuffle each stat column independently
        for j in range(n_stats):
            Y_shuf[:, j] = rng.permutation(Y_shuf[:, j])

        Y_shuf_std = (Y_shuf - Y_shuf.mean(axis=0)) / (Y_shuf.std(axis=0) + 1e-8)
        shuf_corr = (X_std.T @ Y_shuf_std) / (n_samples - 1)
        null_maxima[s] = np.abs(shuf_corr).max()

    null_threshold = np.percentile(null_maxima, percentile)

    # --- Count above threshold ---
    above_mask = obs_corr > null_threshold
    n_above = above_mask.sum()
    total_pairs = k_features * n_stats
    pct_above = 100.0 * n_above / total_pairs

    # --- Details for features above threshold ---
    above_list = []
    for fi in range(k_features):
        for sj, sn in enumerate(stat_names):
            if obs_corr[fi, sj] > null_threshold:
                above_list.append({
                    "feature": int(fi),
                    "stat": sn,
                    "|r|": float(obs_corr[fi, sj]),
                })
    above_list.sort(key=lambda x: x["|r|"], reverse=True)

    # Per-stat summary
    per_stat = {}
    for sj, sn in enumerate(stat_names):
        stat_above = obs_corr[:, sj] > null_threshold
        per_stat[sn] = {
            "n_above": int(stat_above.sum()),
            "pct_above": float(100.0 * stat_above.sum() / k_features),
            "max_r": float(obs_corr[:, sj].max()),
        }

    return {
        "null_threshold": float(null_threshold),
        "n_features": int(k_features),
        "n_stats": int(n_stats),
        "n_shuffles": int(n_shuffles),
        "percentile": float(percentile),
        "observed_max_r": float(obs_max),
        "observed_mean_perstat_max_r": float(obs_mean_max),
        "above_threshold_pairs": int(n_above),
        "above_threshold_pct": float(pct_above),
        "top_above_threshold": above_list[:50],
        "per_stat_summary": per_stat,
        "null_distribution": {
            "mean": float(null_maxima.mean()),
            "std": float(null_maxima.std()),
            "p50": float(np.percentile(null_maxima, 50)),
            "p95": float(np.percentile(null_maxima, 95)),
            "p99": float(np.percentile(null_maxima, 99)),
            "max": float(null_maxima.max()),
        },
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stocks", type=int, default=20)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--random-components", type=int, default=64)
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-shuffles", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str,
                        default="/data/houwanlong/finllm-mi/outputs/sae/recalibrated_null.json")
    parser.add_argument("--data-dir", type=str,
                        default="/data/houwanlong/finllm-mi/data/scale120")
    parser.add_argument("--sae-path", type=str,
                        default="/data/houwanlong/finllm-mi/outputs/sae/sae_layer6.pt")
    parser.add_argument("--model-dir", type=str,
                        default="/data/houwanlong/models/Kronos-base")
    parser.add_argument("--tokenizer-dir", type=str,
                        default="/data/houwanlong/models/Kronos-Tokenizer-base")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    print(f"Config: {json.dumps(vars(args), indent=2, default=str)}")

    # --- List stocks ---
    data_dir = Path(args.data_dir)
    stock_files = sorted(data_dir.glob("sh*.csv"))[:args.n_stocks]
    tickers = [f.stem for f in stock_files]
    print(f"\nAnalyzing {len(tickers)} stocks: {tickers}")

    # --- Load model ---
    print("\n" + "=" * 60)
    print("Loading Kronos model...")
    tokenizer, model, n_layers, d_model = load_kronos_model(device)
    print(f"Model loaded: {n_layers} layers, d_model={d_model}")

    # --- Load SAE ---
    print("\nLoading SAE...")
    sd = torch.load(args.sae_path, map_location="cpu", weights_only=True)
    n_sae_features = sd["encoder.weight"].shape[0]
    print(f"SAE: {n_sae_features} features")

    # --- Precompute random orthogonal basis (same for all stocks) ---
    print("\nGenerating random orthogonal basis...")
    random_basis = random_orthogonal_basis(d_model, args.random_components, seed=42)
    print(f"Random basis: {random_basis.shape}")

    # --- Process each stock ---
    all_results = {
        "config": vars(args),
        "methodology": (
            "For EACH basis (SAE, PCA, Random), we separately calibrate the null "
            "threshold via 100 label shuffles. The threshold is the 95th percentile "
            "of the max |r| across all (feature, stat) pairs under independent label "
            "shuffling. This ensures a FAIR comparison: each method is judged by its "
            "own null distribution, not by a shared threshold."
        ),
        "per_stock": [],
    }

    # Accumulate for cross-stock aggregate
    all_activations = []  # list of (n_wins, d_model) per stock
    all_fin_stats = []    # list of dicts per stock
    stat_names_set = set()

    for si, (stock_file, ticker) in enumerate(zip(stock_files, tickers)):
        print(f"\n{'=' * 60}")
        print(f"Stock {si + 1}/{len(tickers)}: {ticker}")
        print(f"{'=' * 60}")

        # Load CSV data
        df = pd.read_csv(stock_file)
        # Fix BOM in column names
        df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

        price_cols = ["open", "close", "high", "low"]
        vol_col = "volume"
        amt_col = "amount"

        for col in price_cols + [vol_col, amt_col]:
            if col not in df.columns:
                df[col] = 0.0

        raw_data = df[price_cols + [vol_col, amt_col]].values.astype(np.float32)
        valid = ~np.isnan(raw_data).any(axis=1)
        raw_data = raw_data[valid]

        # Normalize
        mean = raw_data.mean(axis=0)
        std = raw_data.std(axis=0)
        data_norm = (raw_data - mean) / (std + 1e-5)
        data_norm = np.clip(data_norm, -5, 5)

        # Create sliding windows
        windows = []
        for i in range(0, len(data_norm) - args.lookback, args.stride):
            windows.append(data_norm[i:i + args.lookback])
        windows = np.stack(windows, axis=0)  # (n_wins, lookback, 6)
        n_wins = windows.shape[0]
        print(f"  Windows: {n_wins}")

        if n_wins < 20:
            print(f"  Skipping {ticker}: only {n_wins} windows")
            continue

        # --- Extract layer-6 residual stream ---
        print(f"  Extracting layer-{args.layer} activations...")
        activations = extract_residual_stream(
            tokenizer, model, args.layer, windows, args.batch_size, device
        )  # (n_wins, d_model)
        print(f"  Activations: {activations.shape}")

        all_activations.append(activations)

        # --- Compute financial statistics ---
        print(f"  Computing financial statistics...")
        fin_stats = compute_financial_statistics(windows, raw_data)
        stat_names = sorted(fin_stats.keys())
        stat_names_set.update(stat_names)
        all_fin_stats.append(fin_stats)
        print(f"  Stats: {len(stat_names)} computed")

        # --- SAE features ---
        print(f"  Encoding SAE...")
        sae_features = encode_sae(activations, args.sae_path)
        print(f"  SAE features: {sae_features.shape}")

        # --- PCA features ---
        pca_k = min(args.pca_components, n_wins, d_model)
        print(f"  Computing PCA ({pca_k} components)...")
        pca = PCA(n_components=pca_k, random_state=42)
        pca_features = pca.fit_transform(activations)  # (n_wins, pca_k)
        pca_basis = pca.components_  # (k, d_model) — already orthonormal
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

        # --- Random basis features (use same k as PCA for fair comparison) ---
        rand_k = pca_k  # match PCA components for fair comparison
        random_features = activations @ random_basis[:rand_k].T  # (n_wins, rand_k)

        # --- Calibrate null thresholds separately for each basis ---
        print(f"\n  Calibrating null thresholds ({args.n_shuffles} shuffles each)...")
        t0 = time.time()

        # SAE
        sae_calib = calibrate_null_threshold(
            sae_features, stat_names, fin_stats,
            n_shuffles=args.n_shuffles, percentile=95.0, seed=42 + si
        )

        # PCA
        pca_calib = calibrate_null_threshold(
            pca_features, stat_names, fin_stats,
            n_shuffles=args.n_shuffles, percentile=95.0, seed=1000 + si
        )

        # Random
        rand_calib = calibrate_null_threshold(
            random_features, stat_names, fin_stats,
            n_shuffles=args.n_shuffles, percentile=95.0, seed=2000 + si
        )

        elapsed = time.time() - t0
        print(f"  Calibration done in {elapsed:.1f}s")

        stock_result = {
            "ticker": ticker,
            "n_windows": int(n_wins),
            "d_model": int(d_model),
            "n_fin_stats": len(stat_names),
            "sae": {
                "n_features": int(n_sae_features),
                "calibration": sae_calib,
            },
            "pca": {
                "n_components": int(pca_k),
                "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
                "calibration": pca_calib,
            },
            "random": {
                "n_components": int(rand_k),
                "calibration": rand_calib,
            },
        }

        # Print summary for this stock
        print(f"\n  --- Comparison for {ticker} ---")
        print(f"  {'Basis':<12} {'Null Thresh':>12} {'Above Thresh %':>15} {'Obs Max |r|':>13} {'# Pairs Above':>13}")
        print(f"  {'-'*12} {'-'*12} {'-'*15} {'-'*13} {'-'*13}")
        for name, calib in [("SAE", sae_calib), ("PCA", pca_calib), ("Random", rand_calib)]:
            print(f"  {name:<12} {calib['null_threshold']:12.4f} {calib['above_threshold_pct']:14.2f}% "
                  f"{calib['observed_max_r']:13.4f} {calib['above_threshold_pairs']:13d}")

        all_results["per_stock"].append(stock_result)

    # --- Cross-stock aggregate ---
    print(f"\n{'=' * 60}")
    print("Cross-Stock Aggregate")
    print(f"{'=' * 60}")

    # Pool all activations and stats for aggregate analysis
    all_acts_pooled = np.concatenate(all_activations, axis=0)  # (total_wins, d_model)
    print(f"Total pooled samples: {all_acts_pooled.shape[0]}")

    all_stat_names = sorted(stat_names_set)
    pooled_stats = {}
    for sn in all_stat_names:
        pooled_stats[sn] = np.concatenate([fs[sn] for fs in all_fin_stats], axis=0)

    # SAE on pooled data
    print("Encoding pooled SAE...")
    pooled_sae = encode_sae(all_acts_pooled, args.sae_path)

    pooled_pca_k = min(args.pca_components, all_acts_pooled.shape[0], d_model)
    print(f"Pooled PCA ({pooled_pca_k} components)...")
    pooled_pca_model = PCA(n_components=pooled_pca_k, random_state=42)
    pooled_pca = pooled_pca_model.fit_transform(all_acts_pooled)
    pooled_rand_k = pooled_pca_k
    pooled_random = all_acts_pooled @ random_basis[:pooled_rand_k].T

    print("Pooled random...")
    pooled_random = all_acts_pooled @ random_basis.T

    print(f"Calibrating on pooled data ({args.n_shuffles} shuffles each)...")
    t0 = time.time()
    pooled_sae_calib = calibrate_null_threshold(
        pooled_sae, all_stat_names, pooled_stats,
        n_shuffles=args.n_shuffles, percentile=95.0, seed=99
    )
    pooled_pca_calib = calibrate_null_threshold(
        pooled_pca, all_stat_names, pooled_stats,
        n_shuffles=args.n_shuffles, percentile=95.0, seed=199
    )
    pooled_rand_calib = calibrate_null_threshold(
        pooled_random, all_stat_names, pooled_stats,
        n_shuffles=args.n_shuffles, percentile=95.0, seed=299
    )
    pooled_time = time.time() - t0

    pooled_aggregate = {
        "n_total_samples": int(all_acts_pooled.shape[0]),
        "n_stocks": len(all_activations),
        "calibration_time_s": float(pooled_time),
        "sae": pooled_sae_calib,
        "pca": pooled_pca_calib,
        "random": pooled_rand_calib,
    }
    all_results["pooled_aggregate"] = pooled_aggregate

    # --- Final comparison table ---
    print(f"\n{'=' * 80}")
    print("FAIR COMPARISON TABLE (each basis with own null-calibrated threshold)")
    print(f"{'=' * 80}")
    print(f"{'Basis':<12} {'Null Thresh':>12} {'Above Thresh %':>15} {'Obs Max |r|':>13} {'# Pairs Above':>13} {'Mean Per-Stat Max':>18}")
    print(f"{'':-<12} {'':-<12} {'':-<15} {'':-<13} {'':-<13} {'':-<18}")
    for name, calib in [("SAE", pooled_sae_calib), ("PCA", pooled_pca_calib), ("Random", pooled_rand_calib)]:
        print(f"  {name:<12} {calib['null_threshold']:12.4f} {calib['above_threshold_pct']:14.2f}% "
              f"{calib['observed_max_r']:13.4f} {calib['above_threshold_pairs']:13d} "
              f"{calib['observed_mean_perstat_max_r']:18.4f}")

    # Per-stock averages
    print(f"\n{'=' * 80}")
    print("PER-STOCK AVERAGES")
    print(f"{'=' * 80}")

    stocks_with_results = all_results["per_stock"]
    for name in ["sae", "pca", "random"]:
        thresh_vals = [s[name]["calibration"]["null_threshold"] for s in stocks_with_results]
        above_vals = [s[name]["calibration"]["above_threshold_pct"] for s in stocks_with_results]
        obs_vals = [s[name]["calibration"]["observed_max_r"] for s in stocks_with_results]
        null_mean = [s[name]["calibration"]["null_distribution"]["mean"] for s in stocks_with_results]

        print(f"  {name}:")
        print(f"    Null threshold: {np.mean(thresh_vals):.4f} +/- {np.std(thresh_vals):.4f}")
        print(f"    Above thresh %:  {np.mean(above_vals):.2f}% +/- {np.std(above_vals):.2f}%")
        print(f"    Obs max |r|:     {np.mean(obs_vals):.4f} +/- {np.std(obs_vals):.4f}")
        print(f"    Null mean:       {np.mean(null_mean):.4f} +/- {np.std(null_mean):.4f}")

    all_results["per_stock_averages"] = {
        name: {
            "null_threshold_mean": float(np.mean([s[name]["calibration"]["null_threshold"] for s in stocks_with_results])),
            "null_threshold_std": float(np.std([s[name]["calibration"]["null_threshold"] for s in stocks_with_results])),
            "above_threshold_pct_mean": float(np.mean([s[name]["calibration"]["above_threshold_pct"] for s in stocks_with_results])),
            "above_threshold_pct_std": float(np.std([s[name]["calibration"]["above_threshold_pct"] for s in stocks_with_results])),
            "observed_max_r_mean": float(np.mean([s[name]["calibration"]["observed_max_r"] for s in stocks_with_results])),
            "observed_max_r_std": float(np.std([s[name]["calibration"]["observed_max_r"] for s in stocks_with_results])),
            "null_mean_mean": float(np.mean([s[name]["calibration"]["null_distribution"]["mean"] for s in stocks_with_results])),
            "null_mean_std": float(np.std([s[name]["calibration"]["null_distribution"]["mean"] for s in stocks_with_results])),
        }
        for name in ["sae", "pca", "random"]
    }

    # --- Save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # --- Key conclusion ---
    print(f"\n{'=' * 80}")
    print("KEY FINDING FOR PAPER")
    print(f"{'=' * 80}")
    sae_th = pooled_sae_calib["null_threshold"]
    pca_th = pooled_pca_calib["null_threshold"]
    rand_th = pooled_rand_calib["null_threshold"]
    print(f"  Null thresholds:     SAE={sae_th:.4f}  PCA={pca_th:.4f}  Random={rand_th:.4f}")
    print(f"  SAE above own null:  {pooled_sae_calib['above_threshold_pct']:.2f}%")
    print(f"  PCA above own null:  {pooled_pca_calib['above_threshold_pct']:.2f}%")
    print(f"  Rand above own null: {pooled_rand_calib['above_threshold_pct']:.2f}%")
    print()
    print(f"  The null threshold varies by basis because different representations")
    print(f"  have different intrinsic correlation structures with financial statistics.")
    print(f"  A fair comparison judges each by its OWN threshold, not a shared one.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
