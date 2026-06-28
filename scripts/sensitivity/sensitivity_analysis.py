#!/usr/bin/env python3
"""SAE sensitivity analysis on Kronos financial model.

Experiments:
  1. k sensitivity:       k=32, 64, 128  (expansion=4x, 20 stocks)
  2. Expansion factor:    2x, 4x, 8x     (k=64, 20 stocks)
  3. Shared SAE on 40 stocks:            (k=64, expansion=4x)
     Compare single shared SAE vs per-stock aggregate.

Output: /data/houwanlong/finllm-mi/outputs/sae/sensitivity_results.json
"""

import torch
import numpy as np
import json
import time
import os
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer  # noqa: E402
from safetensors.torch import load_file               # noqa: E402

device = torch.device("cuda:0")
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/sensitivity_results.json"

LAYER = 6
WINDOW = 64
STRIDE = 32
STEPS = 3000
BATCH_SIZE = 256
LR = 1e-4
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.1                       # reserved, not used for SAE training
PRE_CAL_THRESHOLD = 0.15
NULL_SHUFFLES = 50

LABEL_NAMES = [
    "momentum_5", "trend", "volatility", "vol_persistence",
    "autocorr_lag1", "autocorr_lag5", "max_drawdown", "var_95",
    "max_1day_gain", "max_1day_loss", "skewness", "kurtosis",
    "price_range", "vol_clustering", "volume_trend", "volume_price_corr",
]

# ---------------------------------------------------------------------------
# Load model and tokenizer (once)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading Kronos model and tokenizer ...")
t0_load = time.time()

tok = KronosTokenizer.from_pretrained(
    "/data/houwanlong/models/Kronos-Tokenizer-base"
).to(device).eval()

with open("/data/houwanlong/models/Kronos-base/config.json") as f:
    model_cfg = json.load(f)

model = Kronos(
    s1_bits=model_cfg["s1_bits"],
    s2_bits=model_cfg["s2_bits"],
    n_layers=model_cfg["n_layers"],
    d_model=model_cfg["d_model"],
    n_heads=model_cfg["n_heads"],
    ff_dim=model_cfg["ff_dim"],
    ffn_dropout_p=model_cfg["ffn_dropout_p"],
    attn_dropout_p=model_cfg["attn_dropout_p"],
    resid_dropout_p=model_cfg["resid_dropout_p"],
    token_dropout_p=model_cfg["token_dropout_p"],
    learn_te=model_cfg["learn_te"],
)
sd = load_file("/data/houwanlong/models/Kronos-base/model.safetensors")
model.load_state_dict(sd, strict=False)
model = model.to(device).half().eval()
d_model = model_cfg["d_model"]

print(f"  d_model = {d_model}")
print(f"  Model loaded in {time.time() - t0_load:.1f}s")

# ---------------------------------------------------------------------------
# SAE definition
# ---------------------------------------------------------------------------
class TopKSAE(torch.nn.Module):
    """Top-K Sparse Autoencoder with configurable hidden dim and k."""

    def __init__(self, d_input: int, d_hidden: int, k: int):
        super().__init__()
        self.enc = torch.nn.Linear(d_input, d_hidden, bias=True)
        self.dec = torch.nn.Linear(d_hidden, d_input, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(d_input))
        self.k = k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xc = x - self.b
        lat = self.enc(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat)
        mask.scatter_(-1, idx, 1.0)
        return lat * mask

    def decode(self, lat: torch.Tensor) -> torch.Tensor:
        return self.dec(lat) + self.b


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_stock_windows(fname: str):
    """Load a CSV, normalise, create sliding windows.

    Returns (windows_array, normalised_data) or (None, None) if insufficient data.
    """
    df = pd.read_csv(DATA / fname)
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        if col not in df.columns:
            df[col] = 0.0
    data = df[["open", "close", "high", "low", "volume", "amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 200:
        return None, None

    mn = data.mean(axis=0)
    st = data.std(axis=0)
    dn = np.clip((data - mn) / (st + 1e-5), -5.0, 5.0)

    n_windows = min(1500, (len(dn) - WINDOW) // STRIDE)
    if n_windows < 50:
        return None, None

    wins = np.stack([
        dn[i * STRIDE : i * STRIDE + WINDOW] for i in range(n_windows)
    ])
    return wins, dn


def extract_activations(wins: np.ndarray) -> np.ndarray:
    """Run the model on *all* windows, returning residual-stream acts at layer LAYER,
    final token position, as a float32 numpy array.
    """
    acts_list = []

    def hook_fn(m, i, o):
        a = o[0] if isinstance(o, tuple) else o
        acts_list.append(a[:, -1, :].detach().cpu().float().numpy())

    hook = model.transformer[LAYER].register_forward_hook(hook_fn)
    with torch.no_grad():
        for b in range(0, len(wins), 64):
            batch = torch.from_numpy(wins[b:b + 64]).float().to(device)
            s1, s2 = tok.encode(batch, half=True)
            model(s1, s2)
    hook.remove()
    return np.concatenate(acts_list, axis=0)


def compute_labels(wins: np.ndarray) -> np.ndarray:
    """Compute 16 financial labels per window.  Returns shape (n_wins, 16)."""
    all_labels = []
    for w in wins:
        c = w[:, 1]                          # close
        r = np.diff(c) / (c[:-1] + 1e-5)     # returns
        vol = w[:, 4]                        # volume

        feats = [
            # momentum_5
            c[-1] / c[-6] - 1.0 if len(c) >= 6 else 0.0,
            # trend
            float(np.polyfit(np.arange(WINDOW), c, 1)[0]),
            # volatility
            float(np.std(r)),
            # vol_persistence
            float(np.corrcoef(np.abs(r[1:]), np.abs(r[:-1]))[0, 1]) if len(r) > 2 else 0.0,
            # autocorr_lag1
            float(np.corrcoef(r[1:], r[:-1])[0, 1]) if len(r) > 2 else 0.0,
            # autocorr_lag5
            float(np.corrcoef(r[5:], r[:-5])[0, 1]) if len(r) > 6 else 0.0,
            # max_drawdown
            float(np.min(c / np.maximum.accumulate(c) - 1.0)),
            # var_95
            float(np.percentile(r, 5)),
            # max_1day_gain
            float(np.max(r)),
            # max_1day_loss
            float(np.min(r)),
            # skewness
            float(pd.Series(r).skew()) if len(r) > 2 else 0.0,
            # kurtosis
            float(pd.Series(r).kurtosis()) if len(r) > 3 else 0.0,
            # price_range
            float((c.max() - c.min()) / max(c.mean(), 1e-5)),
            # vol_clustering
            float(np.mean(r ** 2) / (np.var(r) + 1e-10)),
            # volume_trend
            float(np.mean(np.diff(vol) / (vol[:-1] + 1e-5))),
            # volume_price_corr
            float(np.corrcoef(r, np.diff(vol)[:len(r)] / (vol[:-1][:len(r)] + 1e-5))[0, 1])
            if len(r) > 2 else 0.0,
        ]
        all_labels.append(feats)
    return np.array(all_labels, dtype=np.float32)


# ---------------------------------------------------------------------------
# SAE training
# ---------------------------------------------------------------------------
def train_sae(sae, train_acts):
    """Train the SAE for STEPS iterations, batch size BATCH_SIZE, lr LR."""
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    at = torch.from_numpy(train_acts).float().to(device)
    for _step in range(STEPS):
        idx = torch.randint(0, len(at), (BATCH_SIZE,), device=device)
        xr = sae.encode(at[idx])
        loss = torch.nn.functional.mse_loss(sae.decode(xr), at[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
    return sae


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_sae(sae, test_acts, labels):
    """Return dict with variance explained, dead rate, alive count, and per-feature
    correlation info needed for concept-family analysis.
    """
    d_hidden = sae.enc.out_features
    at = torch.from_numpy(test_acts).float().to(device)

    with torch.no_grad():
        lat_tensor = sae.encode(at)
        lat = lat_tensor.cpu().numpy()
        recon = sae.decode(lat_tensor).cpu().numpy()

    # variance explained
    var_total = float(np.var(test_acts))
    mse = float(np.mean((recon - test_acts) ** 2))
    var_explained = float(1.0 - mse / max(var_total, 1e-10))

    # dead features: never fired on test set
    dead_mask = (lat != 0).sum(axis=0) == 0
    dead_rate = float(dead_mask.mean())
    alive_count = int((~dead_mask).sum())

    # per-alive-feature correlations (for concept labelling)
    alive_where_mask = (lat != 0).sum(axis=0) > 10
    feature_corrs = {}          # {feat_idx: [corr_to_label_0, ...]}
    type_dist_pre = defaultdict(int)

    for j in np.where(alive_where_mask)[0]:
        active = lat[:, j] != 0
        if active.sum() < 5:
            continue
        corrs = []
        for k in range(labels.shape[1]):
            c = np.corrcoef(lat[active, j], labels[active, k])[0, 1]
            corrs.append(0.0 if np.isnan(c) else abs(float(c)))
        feature_corrs[j] = corrs
        best = int(np.argmax(corrs))
        if corrs[best] > PRE_CAL_THRESHOLD:
            type_dist_pre[LABEL_NAMES[best]] += 1

    total_assigned = sum(type_dist_pre.values())
    largest_pct = (
        max(type_dist_pre.values()) / max(total_assigned, 1)
        if type_dist_pre else 0.0
    )
    n_families_pre = len(type_dist_pre)

    return {
        "var_explained": var_explained,
        "dead_rate": dead_rate,
        "alive_count": alive_count,
        "d_hidden": d_hidden,
        "largest_pct": largest_pct,
        "n_families_pre_cal": n_families_pre,
        "type_dist_pre_cal": dict(type_dist_pre),
        "_feature_corrs": feature_corrs,   # retained for null calibration
        "_lat": lat,
        "_labels": labels,
    }


# ---------------------------------------------------------------------------
# Null calibration (pooled across stocks, matches fast_null2.py per-shuffle max)
# ---------------------------------------------------------------------------
def null_calibrate_experiment(eval_results, n_shuffles=NULL_SHUFFLES, seed=42):
    """Pool features across stocks, run shuffle-based null calibration.

    For each shuffle round, we shuffle label columns within each stock, compute
    max |r| for every alive feature, then take the global maximum across all
    features.  null_95 is the 95th percentile of these per-round maxima.
    This matches the approach in fast_null2.py, extended to multi-stock pooled data.

    Returns (null_95_threshold, recalibrated_type_dist).
    """
    rng = np.random.RandomState(seed)

    # Group features by stock (they share a label array within each stock).
    stock_features = []  # [(lat_2d, labels_2d, {feat_idx: [corr_per_label]})]
    for er in eval_results:
        fc_dict = er["_feature_corrs"]
        if not fc_dict:
            continue
        stock_features.append((er["_lat"], er["_labels"], fc_dict))

    if not stock_features:
        return 0.0, {}

    # --- Per-shuffle-round null distribution ---
    null_maxes = []
    for _ in range(n_shuffles):
        round_maxes = []
        for lat, labels, fc_dict in stock_features:
            # shuffle each label column independently
            shuf = labels.copy()
            for c in range(shuf.shape[1]):
                rng.shuffle(shuf[:, c])
            for j, _corrs in fc_dict.items():
                active = lat[:, j] != 0
                if active.sum() < 5:
                    continue
                corrs = []
                for k in range(shuf.shape[1]):
                    cc = np.corrcoef(lat[active, j], shuf[active, k])[0, 1]
                    corrs.append(0.0 if np.isnan(cc) else abs(float(cc)))
                if corrs:
                    round_maxes.append(max(corrs))
        if round_maxes:
            null_maxes.append(max(round_maxes))

    null_95 = float(np.percentile(null_maxes, 95)) if null_maxes else 0.0

    # --- Re-count concept families with null-calibrated threshold ---
    type_dist_null = defaultdict(int)
    for lat, labels, fc_dict in stock_features:
        for j, corrs in fc_dict.items():
            best = int(np.argmax(corrs))
            if corrs[best] > null_95:
                type_dist_null[LABEL_NAMES[best]] += 1

    return null_95, dict(type_dist_null)


# ---------------------------------------------------------------------------
# Aggregate per-stock results into experiment-level summary
# ---------------------------------------------------------------------------
def aggregate(eval_results):
    """Compute mean +- std for scalar metrics and merge type distributions."""
    n = len(eval_results)
    ve = [r["var_explained"] for r in eval_results]
    dr = [r["dead_rate"] for r in eval_results]
    ac = [r["alive_count"] for r in eval_results]
    lp = [r["largest_pct"] for r in eval_results]
    nf = [r["n_families_pre_cal"] for r in eval_results]

    merged = defaultdict(int)
    for r in eval_results:
        for k, v in r["type_dist_pre_cal"].items():
            merged[k] += v

    total = sum(merged.values())
    merged_pct = {k: round(v / max(total, 1) * 100, 1) for k, v in merged.items()}
    largest_merged = max(merged.values()) / max(total, 1) if merged else 0.0

    return {
        "n_stocks": n,
        "var_explained_mean": float(np.mean(ve)),
        "var_explained_std": float(np.std(ve)),
        "dead_rate_mean": float(np.mean(dr)),
        "dead_rate_std": float(np.std(dr)),
        "alive_count_mean": float(np.mean(ac)),
        "alive_count_std": float(np.std(ac)),
        "largest_pct_mean": float(np.mean(lp)),
        "largest_pct_std": float(np.std(lp)),
        "n_families_pre_cal_mean": float(np.mean(nf)),
        "n_families_pre_cal_std": float(np.std(nf)),
        "merged_type_dist_pct": merged_pct,
        "merged_largest_pct": float(largest_merged),
        "_eval_results": eval_results,
    }


# ---------------------------------------------------------------------------
# Stock pre-processing (done once for all stocks, reused across experiments)
# ---------------------------------------------------------------------------
def preprocess_stocks(fnames):
    """Pre-extract windows, activations, labels for a list of CSV filenames.

    Returns list of dicts: {fname, train_acts, test_acts, test_labels}
    or None per stock.
    """
    prepped = []
    for fname in fnames:
        wins, _dn = load_stock_windows(fname)
        if wins is None:
            prepped.append(None)
            continue

        n_total = len(wins)
        n_train = int(n_total * TRAIN_SPLIT)
        n_val = int(n_total * VAL_SPLIT)
        n_test = n_total - n_train - n_val

        if n_train < 30 or n_test < 20:
            prepped.append(None)
            continue

        all_acts = extract_activations(wins)
        train_acts = all_acts[:n_train]
        test_acts = all_acts[n_train + n_val:]
        test_wins = wins[n_train + n_val:]
        test_labels = compute_labels(test_wins)

        prepped.append({
            "fname": fname,
            "train_acts": train_acts,
            "test_acts": test_acts,
            "test_labels": test_labels,
            "n_train": n_train,
            "n_test": n_test,
        })

    return prepped


# Global cache: key=(fname, d_hidden, k) -> eval dict, avoids redundant SAE training
_sae_cache = {}

def train_eval_on_prepped(prepped, d_hidden, k):
    """Train and evaluate an SAE on already-prepped stock data.

    Returns eval dict or None.  Results are cached by (fname, d_hidden, k).
    """
    cache_key = (prepped["fname"], d_hidden, k)
    if cache_key in _sae_cache:
        return _sae_cache[cache_key]

    sae = TopKSAE(d_model, d_hidden, k).to(device)
    train_sae(sae, prepped["train_acts"])
    eval_r = evaluate_sae(sae, prepped["test_acts"], prepped["test_labels"])
    del sae
    torch.cuda.empty_cache()

    _sae_cache[cache_key] = eval_r
    return eval_r


# ---------------------------------------------------------------------------
# Helper: run one experiment config over a list of prepped stocks
# ---------------------------------------------------------------------------
def run_experiment_config(prepped_list, d_hidden, k, label=""):
    """Train per-stock SAEs, aggregate, null-calibrate, return summary dict."""
    t0 = time.time()
    eval_results = []
    for i, p in enumerate(prepped_list):
        if p is None:
            continue
        er = train_eval_on_prepped(p, d_hidden, k)
        if er is not None:
            er["_fname"] = p["fname"]
            eval_results.append(er)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(prepped_list)}] stocks done  ({label})")

    summary = aggregate(eval_results)

    # null calibration
    null_95, type_dist_null = null_calibrate_experiment(eval_results)
    null_total = sum(type_dist_null.values())
    summary["null_95_threshold"] = null_95
    summary["n_families_null_cal"] = len(type_dist_null)
    summary["merged_type_dist_null_cal"] = type_dist_null
    merged_largest_null = (
        max(type_dist_null.values()) / max(null_total, 1)
        if type_dist_null else 0.0
    )
    summary["merged_largest_pct_null_cal"] = merged_largest_null
    summary["_eval_results"] = eval_results  # retained for downstream reuse

    summary["elapsed_s"] = round(time.time() - t0, 1)
    return summary


# ===================================================================
# MAIN
# ===================================================================
def main():
    t_start = time.time()

    all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])
    print(f"Found {len(all_csvs)} CSV files in {DATA}")

    # -------------------------------------------------------------------
    # Pre-process all 40 stocks ONCE (activations, labels, splits)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pre-processing 40 stocks (activations + labels) ...")
    t_pre = time.time()
    all_prepped = preprocess_stocks(all_csvs[:30])
    valid_indices = [i for i, p in enumerate(all_prepped) if p is not None]
    valid_prepped = [all_prepped[i] for i in valid_indices]
    print(f"  {len(valid_prepped)} / 40 stocks have sufficient data")
    print(f"  Pre-processing done in {time.time() - t_pre:.0f}s")

    if len(valid_prepped) < 5:
        print("ERROR: too few valid stocks. Exiting.")
        return

    results = {}

    # -------------------------------------------------------------------
    # Experiment 1: k sensitivity (k = 32, 64, 128, expansion = 4x)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: k sensitivity  (k = 32, 64, 128)  expansion = 4x")
    print("=" * 60)

    k_stocks = valid_prepped[:30]
    print(f"Using {len(k_stocks)} stocks")

    exp1 = {}
    for k_val in [32, 64, 128]:
        d_hidden = d_model * 4
        cfg_label = f"k{k_val}_exp4x"
        print(f"\n--- {cfg_label} (d_hidden={d_hidden}) ---")

        summary = run_experiment_config(k_stocks, d_hidden, k_val, label=cfg_label)
        exp1[cfg_label] = summary

        print(f"  var_explained = {summary['var_explained_mean']:.4f} "
              f"+/- {summary['var_explained_std']:.4f}")
        print(f"  dead_rate     = {summary['dead_rate_mean']:.4f} "
              f"+/- {summary['dead_rate_std']:.4f}")
        print(f"  alive_count   = {summary['alive_count_mean']:.1f} "
              f"+/- {summary['alive_count_std']:.1f}")
        print(f"  largest_pct   = {summary['largest_pct_mean']:.4f}")
        print(f"  n_families_pre = {summary['n_families_pre_cal_mean']:.1f}")
        print(f"  null_95       = {summary['null_95_threshold']:.4f}, "
              f"n_families_null = {summary['n_families_null_cal']}")

    results["k_sensitivity"] = exp1

    # -------------------------------------------------------------------
    # Experiment 2: expansion factor sensitivity (2x, 4x, 8x, k = 64)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: expansion factor sensitivity  (2x, 4x, 8x)  k = 64")
    print("=" * 60)

    exp2_stocks = valid_prepped[:30]
    print(f"Using {len(exp2_stocks)} stocks")

    exp2 = {}
    for exp_factor in [2, 4, 8]:
        d_hidden = d_model * exp_factor
        cfg_label = f"k64_exp{exp_factor}x"
        print(f"\n--- {cfg_label} (d_hidden={d_hidden}) ---")

        summary = run_experiment_config(exp2_stocks, d_hidden, k=64, label=cfg_label)
        exp2[cfg_label] = summary

        print(f"  var_explained = {summary['var_explained_mean']:.4f} "
              f"+/- {summary['var_explained_std']:.4f}")
        print(f"  dead_rate     = {summary['dead_rate_mean']:.4f} "
              f"+/- {summary['dead_rate_std']:.4f}")
        print(f"  alive_count   = {summary['alive_count_mean']:.1f} "
              f"+/- {summary['alive_count_std']:.1f}")
        print(f"  largest_pct   = {summary['largest_pct_mean']:.4f}")
        print(f"  n_families_pre = {summary['n_families_pre_cal_mean']:.1f}")
        print(f"  null_95       = {summary['null_95_threshold']:.4f}, "
              f"n_families_null = {summary['n_families_null_cal']}")

    results["expansion_sensitivity"] = exp2

    # -------------------------------------------------------------------
    # Experiment 3: Shared SAE on 40 stocks (k=64, expansion=4x)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Shared SAE on 30 stocks  (k=64, expansion=4x)")
    print("=" * 60)

    shared_stocks = valid_prepped  # all 30 valid
    n_shared = len(shared_stocks)
    print(f"Using {n_shared} stocks")

    d_hidden_shared = d_model * 4

    # 3a: single shared SAE
    print("\n--- Shared SAE ---")
    t_shared = time.time()

    all_train_acts = np.concatenate(
        [p["train_acts"] for p in shared_stocks], axis=0
    )
    print(f"  Concatenated training acts: {all_train_acts.shape}")

    shared_sae = TopKSAE(d_model, d_hidden_shared, k=64).to(device)
    train_sae(shared_sae, all_train_acts)

    shared_eval_results = []
    for idx, p in enumerate(shared_stocks):
        er = evaluate_sae(shared_sae, p["test_acts"], p["test_labels"])
        if er is not None:
            er["_fname"] = p["fname"]
            shared_eval_results.append(er)
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{n_shared}] stocks evaluated")

    del shared_sae
    torch.cuda.empty_cache()

    shared_summary = aggregate(shared_eval_results)
    null_95_s, type_dist_null_s = null_calibrate_experiment(shared_eval_results)
    null_total_s = sum(type_dist_null_s.values())
    shared_summary["null_95_threshold"] = null_95_s
    shared_summary["n_families_null_cal"] = len(type_dist_null_s)
    shared_summary["merged_type_dist_null_cal"] = type_dist_null_s
    shared_summary["merged_largest_pct_null_cal"] = (
        max(type_dist_null_s.values()) / max(null_total_s, 1)
        if type_dist_null_s else 0.0
    )
    for er in shared_eval_results:
        for key in ["_feature_corrs", "_lat", "_labels", "_fname"]:
            er.pop(key, None)
    shared_summary.pop("_eval_results", None)
    shared_summary["elapsed_s"] = round(time.time() - t_shared, 1)

    # 3b: per-stock SAEs (same stocks, same config)
    print("\n--- Per-stock SAEs (baseline for comparison) ---")
    perstock_summary = run_experiment_config(
        shared_stocks, d_hidden_shared, k=64, label="per-stock"
    )

    # comparison table
    exp3 = {
        "shared_sae": shared_summary,
        "per_stock_sae": perstock_summary,
        "n_stocks": n_shared,
        "config": "k64_exp4x",
        "comparison": {
            "var_explained_shared": shared_summary["var_explained_mean"],
            "var_explained_perstock": perstock_summary["var_explained_mean"],
            "dead_rate_shared": shared_summary["dead_rate_mean"],
            "dead_rate_perstock": perstock_summary["dead_rate_mean"],
            "n_families_pre_cal_shared": shared_summary["n_families_pre_cal_mean"],
            "n_families_pre_cal_perstock": perstock_summary["n_families_pre_cal_mean"],
            "n_families_null_cal_shared": shared_summary["n_families_null_cal"],
            "n_families_null_cal_perstock": perstock_summary["n_families_null_cal"],
            "largest_pct_shared": shared_summary["largest_pct_mean"],
            "largest_pct_perstock": perstock_summary["largest_pct_mean"],
        },
    }
    results["shared_sae"] = exp3

    print(f"\n  Shared   var_expl={shared_summary['var_explained_mean']:.4f}  "
          f"dead_rate={shared_summary['dead_rate_mean']:.4f}  "
          f"n_fam_pre={shared_summary['n_families_pre_cal_mean']:.1f}")
    print(f"  PerStock var_expl={perstock_summary['var_explained_mean']:.4f}  "
          f"dead_rate={perstock_summary['dead_rate_mean']:.4f}  "
          f"n_fam_pre={perstock_summary['n_families_pre_cal_mean']:.1f}")

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    results["meta"] = {
        "d_model": d_model,
        "layer": LAYER,
        "window": WINDOW,
        "stride": STRIDE,
        "train_steps": STEPS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "train_split": TRAIN_SPLIT,
        "val_split": VAL_SPLIT,
        "pre_cal_threshold": PRE_CAL_THRESHOLD,
        "null_shuffles": NULL_SHUFFLES,
        "n_stocks_preprocessed": len(valid_prepped),
        "total_elapsed_s": round(time.time() - t_start, 1),
    }

    # Clean non-serializable internals before saving
    def clean_internals(obj):
        if isinstance(obj, dict):
            for k in ["_feature_corrs", "_lat", "_labels", "_fname", "_eval_results"]:
                obj.pop(k, None)
            for v in obj.values():
                clean_internals(v)
        elif isinstance(obj, list):
            for v in obj:
                clean_internals(v)
    clean_internals(results)

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT}")
    print(f"Total elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
