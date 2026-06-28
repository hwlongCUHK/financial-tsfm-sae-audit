#!/usr/bin/env python3
"""SAE vs PCA vs Random baseline comparison for reviewer response.

Compares three feature discovery methods on financial time-series:
  1. SAE (Sparse Autoencoder) -- k=64, expansions 2x/4x/8x
  2. PCA -- 64 components on same training activations
  3. Random orthogonal basis -- 64 random orthogonal vectors in 832d

Metrics:
  (a) Feature labeling quality: max |r| with 30+ financial stats
  (b) Causal ablation effect: change in reconstruction when top features removed
  (c) % features above null threshold

20 stocks: 5 per sector (Bank, Energy, Tech, Consumer).
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from safetensors.torch import load_file

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

# ─── Paths ───
DATA_DIR = Path("/data/houwanlong/finllm-mi/data/scale120")
MODEL_DIR = Path("/data/houwanlong/models/Kronos-base")
OUTPUT_DIR = Path("/data/houwanlong/finllm-mi/outputs/sae")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda:0"

# ─── Config ───
LAYER_IDX = 6
D_MODEL = 832
K_FEATURES = 64
EXPANSIONS = [2, 4, 8]
TRAIN_STEPS = 3000
BATCH_SIZE = 256
LR = 1e-4
N_STOCKS_PER_SECTOR = 5
LOOKBACK = 64
STRIDE = 32
N_ABLATE = 20  # number of top features to ablate

SECTOR_STOCKS = {
    "Bank":     ["sh600000", "sh600015", "sh600016", "sh600030", "sh600036"],
    "Energy":   ["sh600021", "sh600023", "sh600028", "sh600058", "sh600101"],
    "Tech":     ["sh600072", "sh600081", "sh600143", "sh600152", "sh600183"],
    "Consumer": ["sh600006", "sh600054", "sh600056", "sh600079", "sh600088"],
}

ALL_STOCKS = []
for sector, stocks in SECTOR_STOCKS.items():
    for s in stocks[:N_STOCKS_PER_SECTOR]:
        ALL_STOCKS.append((s, sector))

print(f"Processing {len(ALL_STOCKS)} stocks from {len(SECTOR_STOCKS)} sectors")


# ═══════════════════════════════════════════════════════════════
# Kronos Model Loading
# ═══════════════════════════════════════════════════════════════

def load_kronos(device):
    """Load Kronos autoregressive model using safetensors."""
    print("Loading Kronos model...")
    sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
    from model.kronos import Kronos

    with open(str(MODEL_DIR / "config.json")) as f:
        cfg = json.load(f)

    model = Kronos(
        s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"],
        n_layers=cfg["n_layers"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
        ffn_dropout_p=cfg["ffn_dropout_p"], attn_dropout_p=cfg["attn_dropout_p"],
        resid_dropout_p=cfg["resid_dropout_p"], token_dropout_p=cfg["token_dropout_p"],
        learn_te=cfg["learn_te"],
    )
    state_dict = load_file(str(MODEL_DIR / "model.safetensors"))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).half().eval()
    print(f"  Kronos: {len(model.transformer)} layers, d_model={model.d_model}")
    return model


# ═══════════════════════════════════════════════════════════════
# TopK Sparse Autoencoder
# ═══════════════════════════════════════════════════════════════

class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder with pre-encoder bias."""

    def __init__(self, d_model, d_hidden, k):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_model, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        return self.encoder(x - self.pre_bias)

    def decode(self, h):
        return self.decoder(h) + self.pre_bias

    def forward(self, x):
        pre = self.encode(x)
        vals, idx = torch.topk(pre, self.k, dim=-1)
        mask = torch.zeros_like(pre).scatter(-1, idx, 1.0)
        h = F.relu(pre) * mask
        return self.decode(h), h

    def get_feature_activations(self, x):
        """(n_samples, d_hidden) numpy array of feature activations."""
        with torch.no_grad():
            pre = self.encode(x)
            vals, idx = torch.topk(pre, self.k, dim=-1)
            mask = torch.zeros_like(pre).scatter(-1, idx, 1.0)
            return (F.relu(pre) * mask).cpu().float().numpy()

    def get_top_features(self, x, n):
        """Indices of top-n features by mean absolute activation."""
        with torch.no_grad():
            acts = self.get_feature_activations(x)
            return np.argsort(np.abs(acts).mean(axis=0))[-n:]

    def ablate(self, x, feature_idx):
        """Encode, zero out specified features, decode."""
        with torch.no_grad():
            pre = self.encode(x)
            vals, idx = torch.topk(pre, self.k, dim=-1)
            mask = torch.zeros_like(pre).scatter(-1, idx, 1.0)
            h = F.relu(pre) * mask
            h[:, feature_idx] = 0.0
            return self.decode(h)


def train_sae(activations, d_hidden, k, steps, lr, batch_size, device):
    """Train TopK SAE, return (model, var_explained, l0, alive_frac)."""
    sae = TopKSAE(D_MODEL, d_hidden, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    data = torch.from_numpy(activations).float().to(device)
    n = data.shape[0]

    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        batch = data[idx]
        x_recon, h = sae(batch)
        l_recon = F.mse_loss(x_recon, batch)
        l_aux = 1e-3 * h.abs().mean()
        loss = l_recon + l_aux
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        x_recon, h = sae(data)
        ve = 1.0 - F.mse_loss(x_recon, data) / data.var(dim=0).mean()
        l0 = (h > 1e-4).float().sum(dim=-1).mean().item()
        alive = (h.sum(dim=0) > 1e-4).float().mean().item()
    return sae, float(ve), float(l0), float(alive)


# ═══════════════════════════════════════════════════════════════
# Financial Labels (30+ statistics per window)
# ═══════════════════════════════════════════════════════════════

def compute_financial_labels(prices, volumes):
    """Compute 30+ financial statistics for each window.

    Args:
        prices: (n_windows, lookback) CLOSE prices
        volumes: (n_windows, lookback) volumes or None
    Returns:
        labels: (n, n_stats) array, names: list of stat names
    """
    n, T = prices.shape
    out = np.zeros((n, 32), dtype=np.float32)

    for i in range(n):
        p = prices[i]
        v = volumes[i] if volumes is not None else np.ones(T)
        ret = np.diff(p) / (p[:-1] + 1e-8)
        abs_ret = np.abs(ret)
        x = np.arange(T)

        feats = [
            np.std(ret),                                                   # 0  volatility
            np.mean(ret),                                                  # 1  mean_return
            sp_stats.skew(ret) if len(ret) > 3 else 0,                     # 2  skewness
            sp_stats.kurtosis(ret) if len(ret) > 4 else 0,                 # 3  kurtosis
            np.mean(ret) / (np.std(ret) + 1e-8),                           # 4  sharpe
            np.max(np.maximum.accumulate(p) - p) / (np.max(p) + 1e-8),     # 5  max_drawdown
            np.max(ret),                                                   # 6  max_1day_gain
            np.min(ret),                                                   # 7  max_1day_loss
            (np.max(p) - np.min(p)) / (p[-1] + 1e-8),                     # 8  price_range
            np.polyfit(x, p, 1)[0] / (p[-1] + 1e-8),                     # 9  trend_slope
        ]

        # 10 trend_r2
        slope = np.polyfit(x, p, 1)[0]
        trend_line = slope * x + (np.mean(p) - slope * np.mean(x))
        ss_res = np.sum((p - trend_line)**2)
        ss_tot = np.sum((p - np.mean(p))**2)
        feats.append(1 - ss_res / (ss_tot + 1e-8))

        feats.extend([
            (p[-1] - p[0]) / (p[0] + 1e-8),                               # 11 momentum_full
            (p[-1] - p[-max(1,T//5)]) / (p[-max(1,T//5)] + 1e-8),        # 12 momentum_5
            (p[-1] - p[-max(1,T//10)]) / (p[-max(1,T//10)] + 1e-8),      # 13 momentum_10
        ])

        # 14-16 autocorr lags 1,3,5
        for lag in [1, 3, 5]:
            if len(ret) > lag + 2:
                feats.append(np.corrcoef(ret[:-lag], ret[lag:])[0, 1])
            else:
                feats.append(0.0)

        # 17 vol_clustering
        if len(abs_ret) > 3:
            feats.append(np.corrcoef(abs_ret[:-1], abs_ret[1:])[0, 1])
        else:
            feats.append(0.0)

        # 18 vol_of_vol
        rv = np.array([np.std(ret[max(0,j-5):j+1]) for j in range(len(ret))])
        feats.append(np.std(rv) if len(rv) > 1 else 0.0)

        # 19 volume_trend
        feats.append(np.corrcoef(x, v)[0, 1] if np.std(v) > 0 else 0.0)

        # 20 volume_volatility
        feats.append(np.std(v) / (np.mean(v) + 1e-8))

        # 21 volume_price_corr
        feats.append(np.corrcoef(v, p)[0, 1] if np.std(v) > 0 else 0.0)

        # 22 high_low_range
        feats.append((np.max(p) - np.min(p)) / (np.mean(p) + 1e-8))

        # 23 parkinson_vol
        hl = np.log((np.max(p) + 1e-8) / (np.min(p) + 1e-8))
        feats.append(hl / (4 * np.log(2))**0.5)

        # 24 garman_klass_vol
        if np.max(p) > np.min(p):
            feats.append(np.sqrt(0.5 * np.log(np.max(p)/np.min(p))**2 -
                                 (2*np.log(2)-1) * np.log(p[-1]/p[0])**2))
        else:
            feats.append(0.0)

        # 25 rsi_like
        feats.append(np.sum(ret > 0) / max(1, len(ret)))

        # 26 leverage_effect
        if len(ret) > 5:
            fv = np.array([np.std(ret[j:min(j+5, len(ret))]) for j in range(len(ret)-1)])
            feats.append(np.corrcoef(ret[:-1], fv)[0, 1])
        else:
            feats.append(0.0)

        # 27 hurst_like
        ma = p - np.mean(p)
        cd = np.cumsum(ma)
        R = np.max(cd) - np.min(cd)
        S = np.std(p)
        feats.append(np.log(R/(S+1e-8)) / np.log(T) if S > 0 else 0.5)

        # 28 jarque_bera
        s = sp_stats.skew(ret) if len(ret) > 3 else 0
        k = sp_stats.kurtosis(ret) if len(ret) > 4 else 0
        feats.append(len(ret)/6 * (s**2 + k**2/4))

        # 29 sign_changes
        feats.append(np.sum(np.diff(np.sign(ret)) != 0) / max(1, len(ret)))

        # 30 ret_entropy
        bins = np.digitize(ret, bins=np.percentile(ret, [10,25,50,75,90]))
        _, cnt = np.unique(bins, return_counts=True)
        prob = cnt / cnt.sum()
        ent = -np.sum(prob * np.log(prob + 1e-8))
        feats.append(ent / max(1e-8, np.log(max(1, len(cnt)))))

        # 31 avg_true_range
        if len(ret) > 1:
            tr = np.maximum(np.abs(np.diff(p)),
                            np.maximum(np.abs(p[1:]-p[:-1]), np.abs(p[:-1]-p[1:])))
            feats.append(np.mean(tr) / (np.mean(p) + 1e-8))
        else:
            feats.append(0.0)

        out[i] = feats

    names = [
        "volatility", "mean_return", "skewness", "kurtosis", "sharpe_ratio",
        "max_drawdown", "max_1day_gain", "max_1day_loss", "price_range", "trend_slope",
        "trend_r2", "momentum_full", "momentum_5", "momentum_10",
        "autocorr_lag1", "autocorr_lag3", "autocorr_lag5",
        "vol_clustering", "vol_of_vol", "volume_trend", "volume_volatility",
        "volume_price_corr", "high_low_range", "parkinson_vol", "garman_klass_vol",
        "rsi_like", "leverage_effect", "hurst_like", "jarque_bera",
        "sign_changes", "ret_entropy", "avg_true_range",
    ]
    return out, names


# ═══════════════════════════════════════════════════════════════
# Activation Extraction
# ═══════════════════════════════════════════════════════════════

def extract_activations_and_labels(model, csv_path, device):
    """Extract layer activations and compute financial labels for one stock.

    Uses random token IDs (consistent with existing pipeline) to probe
    model's internal representational structure.
    """
    df = pd.read_csv(csv_path)
    price_cols = ["open", "close", "high", "low"]
    vol_col = "volume"

    for col in price_cols + [vol_col]:
        if col not in df.columns:
            df[col] = 0.0

    raw_prices = df["close"].values.astype(np.float32)
    raw_vol = df[vol_col].values.astype(np.float32)

    # Create windows
    prices_win, volumes_win = [], []
    for i in range(0, len(raw_prices) - LOOKBACK, STRIDE):
        p = raw_prices[i:i + LOOKBACK]
        if np.any(np.isnan(p)) or np.std(p) < 1e-8:
            continue
        prices_win.append(p)
        volumes_win.append(raw_vol[i:i + LOOKBACK])

    if len(prices_win) < 20:
        return None, None, None, None

    prices_win = np.array(prices_win, dtype=np.float32)
    volumes_win = np.array(volumes_win, dtype=np.float32)

    # Financial labels
    fin_labels, stat_names = compute_financial_labels(prices_win, volumes_win)

    # Extract activations using random tokens
    n_samples = len(prices_win)
    seq_len = LOOKBACK
    activations = []

    def hook_fn(module, inp, out):
        act = out[0] if isinstance(out, tuple) else out
        activations.append(act.detach().cpu().float().mean(dim=1))

    hook = model.transformer[LAYER_IDX].register_forward_hook(hook_fn)

    with torch.no_grad():
        for b in range(0, n_samples, 16):
            b_end = min(b + 16, n_samples)
            bs = b_end - b
            s1 = torch.randint(0, 2**10, (bs, seq_len), device=device)
            s2 = torch.randint(0, 2**10, (bs, seq_len), device=device)
            try:
                _ = model(s1, s2)
            except Exception:
                try:
                    stamp = torch.zeros(bs, seq_len, device=device)
                    _ = model(s1, s2, stamp=stamp)
                except Exception as e:
                    print(f"    Forward error batch {b}: {e}")
                    continue

    hook.remove()

    if len(activations) == 0:
        return None, None, None, None

    acts = torch.cat(activations, dim=0).numpy()
    # Trim to match labels
    n = min(len(acts), len(fin_labels))
    return acts[:n], fin_labels[:n], stat_names, prices_win[:n]


# ═══════════════════════════════════════════════════════════════
# PCA and Random Baselines
# ═══════════════════════════════════════════════════════════════

def compute_pca(train_acts, n_comp):
    """Return PCA model, components, cumulative var, scaler."""
    n_comp = min(n_comp, min(train_acts.shape))
    scaler = StandardScaler()
    X = scaler.fit_transform(train_acts)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X)
    return pca, scaler, float(np.sum(pca.explained_variance_ratio_))


def random_orthogonal_basis(d, k, seed=42):
    """Generate k random orthogonal vectors in R^d."""
    rng = np.random.RandomState(seed)
    M = rng.randn(d, k).astype(np.float32)
    Q, _ = np.linalg.qr(M)
    return Q[:, :k].T  # (k, d)


# ═══════════════════════════════════════════════════════════════
# Feature Labeling: correlate with financial stats
# ═══════════════════════════════════════════════════════════════

def label_features(feat_acts, fin_labels, stat_names):
    """Correlate each feature's activation with financial statistics.

    Args:
        feat_acts: (n_samples, n_features)
        fin_labels: (n_samples, n_stats)
        stat_names: list of stat names
    Returns:
        best_corr: (n_features,) max |r| per feature
        best_stat: (n_features,) which stat
        summary: dict
    """
    nf, ns = feat_acts.shape[1], fin_labels.shape[1]
    best_corr = np.zeros(nf, dtype=np.float32)
    best_stat_idx = np.zeros(nf, dtype=int)

    for i in range(nf):
        f = feat_acts[:, i]
        if np.std(f) < 1e-8:
            continue
        corrs = np.zeros(ns)
        for j in range(ns):
            s = fin_labels[:, j]
            if np.std(s) < 1e-8:
                continue
            c = np.corrcoef(f, s)[0, 1]
            corrs[j] = abs(c) if not np.isnan(c) else 0.0
        best_corr[i] = corrs.max()
        best_stat_idx[i] = corrs.argmax()

    n_above = int((best_corr > 0.3).sum())
    stat_counts = defaultdict(int)
    for idx in best_stat_idx:
        if idx < len(stat_names):
            stat_counts[stat_names[idx]] += 1

    return best_corr, best_stat_idx, {
        "max_corr": float(best_corr.max()) if nf > 0 else 0.0,
        "mean_corr": float(best_corr.mean()) if nf > 0 else 0.0,
        "median_corr": float(np.median(best_corr)) if nf > 0 else 0.0,
        "n_above_threshold": n_above,
        "pct_above_threshold": float(n_above / nf * 100) if nf > 0 else 0.0,
        "top_stats": sorted(stat_counts.items(), key=lambda x: x[1], reverse=True)[:5],
    }


# ═══════════════════════════════════════════════════════════════
# Ablation effect measurement
# ═══════════════════════════════════════════════════════════════

def measure_ablation_effect(original, ablated, max_samples=200):
    """Cosine similarity between original and ablated representations."""
    n = min(len(original), max_samples)
    orig = original[:n]
    abl = ablated[:n]
    norm_orig = orig / (np.linalg.norm(orig, axis=1, keepdims=True) + 1e-8)
    norm_abl = abl / (np.linalg.norm(abl, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(norm_orig * norm_abl, axis=1)))


# ═══════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SAE vs PCA vs Random: Baseline Comparison")
    print(f"{'='*70}\n")

    model = load_kronos(DEVICE)

    all_results = {
        "config": {
            "n_stocks": len(ALL_STOCKS),
            "layer": LAYER_IDX,
            "d_model": D_MODEL,
            "k_features": K_FEATURES,
            "expansions": EXPANSIONS,
            "train_steps": TRAIN_STEPS,
            "lr": LR,
            "n_ablate": N_ABLATE,
        },
        "stocks": [],
        "aggregate": {},
    }

    t0 = time.time()

    for si, (stock_id, sector) in enumerate(ALL_STOCKS):
        csv_path = DATA_DIR / f"{stock_id}.csv"
        if not csv_path.exists():
            print(f"\n[{si+1}/{len(ALL_STOCKS)}] {stock_id} ({sector}) -- CSV MISSING, SKIP")
            continue

        t_stock = time.time()
        print(f"\n{'='*60}")
        print(f"[{si+1}/{len(ALL_STOCKS)}] {stock_id} ({sector})")
        print(f"{'='*60}")

        # 1. Extract activations and labels
        print("  Extracting activations...")
        acts, fin_labels, stat_names, _ = extract_activations_and_labels(
            model, str(csv_path), DEVICE
        )
        if acts is None:
            print("  SKIP: insufficient data")
            continue

        n = len(acts)
        n_train = int(n * 0.8)
        train_acts = acts[:n_train]
        test_acts = acts[n_train:]
        train_labels = fin_labels[:n_train]
        test_labels = fin_labels[n_train:]
        print(f"  Samples: {n} (train={n_train}, test={n-n_train}), labels: {fin_labels.shape[1]}d")

        test_tensor = torch.from_numpy(test_acts).float().to(DEVICE)

        sr = {
            "stock_id": stock_id, "sector": sector,
            "n_samples": n, "n_train": n_train, "n_test": n - n_train,
            "sae": {}, "pca": {}, "random": {},
        }

        # ─── PCA baseline ───
        print("  PCA baseline...")
        pca_model, pca_scaler, pca_cumvar = compute_pca(train_acts, K_FEATURES)
        pca_test = pca_model.transform(pca_scaler.transform(test_acts))
        pca_bc, pca_bsi, pca_sum = label_features(pca_test, test_labels, stat_names)

        # PCA ablation: zero out top-correlated components, reconstruct
        n_ab = min(N_ABLATE, K_FEATURES)
        pca_top = np.argsort(pca_bc)[-n_ab:]
        pca_abl = pca_test.copy()
        pca_abl[:, pca_top] = 0.0
        pca_full_recon = pca_scaler.inverse_transform(pca_model.inverse_transform(pca_test))
        pca_abl_recon = pca_scaler.inverse_transform(pca_model.inverse_transform(pca_abl))

        cos_pca_full = measure_ablation_effect(test_acts, pca_full_recon)
        cos_pca_abl = measure_ablation_effect(test_acts, pca_abl_recon)

        pca_n_comp = pca_model.n_components_
        n_ab = min(N_ABLATE, pca_n_comp)
        pca_top = np.argsort(pca_bc)[-n_ab:]

        sr["pca"] = {
            "n_components": pca_n_comp,
            "cum_var": pca_cumvar,
            "labeling": pca_sum,
            "ablation": {"cos_full": cos_pca_full, "cos_ablated": cos_pca_abl,
                         "delta": cos_pca_full - cos_pca_abl},
        }
        print(f"    n_comp={pca_n_comp}, cum_var={pca_cumvar:.4f}, max|r|={pca_sum['max_corr']:.4f}, "
              f"above={pca_sum['n_above_threshold']}/{pca_n_comp}, "
              f"abl_delta={cos_pca_full - cos_pca_abl:.4f}")

        # ─── Random baseline ───
        print("  Random orthogonal baseline...")
        rand_n_features = pca_n_comp  # Use same count as PCA for fair comparison
        rand_basis = random_orthogonal_basis(D_MODEL, rand_n_features)
        rand_test = test_acts @ rand_basis.T  # (n_test, rand_n_features)
        rand_bc, rand_bsi, rand_sum = label_features(rand_test, test_labels, stat_names)

        # Random ablation: subtract projection onto top random basis vectors
        n_ab_rand = min(N_ABLATE, rand_n_features)
        rand_top = np.argsort(rand_bc)[-n_ab_rand:]
        rand_abl_recon = test_acts.copy()
        proj_rm = test_acts @ rand_basis[rand_top].T  # (n_test, n_ab_rand)
        rand_abl_recon = rand_abl_recon - proj_rm @ rand_basis[rand_top]

        cos_rand_full = 1.0  # identity
        cos_rand_abl = measure_ablation_effect(test_acts, rand_abl_recon)

        sr["random"] = {
            "n_features": rand_n_features,
            "labeling": rand_sum,
            "ablation": {"cos_full": cos_rand_full, "cos_ablated": cos_rand_abl,
                         "n_ablated": n_ab_rand, "delta": cos_rand_full - cos_rand_abl},
        }
        print(f"    max|r|={rand_sum['max_corr']:.4f}, above={rand_sum['n_above_threshold']}/{rand_n_features}, "
              f"abl_delta={cos_rand_full - cos_rand_abl:.4f}")

        # ─── SAE: multiple expansion factors ───
        for expansion in EXPANSIONS:
            d_hidden = D_MODEL * expansion
            exp_key = f"exp{expansion}x"
            print(f"  SAE {expansion}x (d_hidden={d_hidden})...")

            ts = time.time()
            sae, ve, l0, alive = train_sae(train_acts, d_hidden, K_FEATURES,
                                           TRAIN_STEPS, LR, BATCH_SIZE, DEVICE)
            dt = time.time() - ts

            # Feature activations on test set
            sae_test = sae.get_feature_activations(test_tensor)
            sae_bc, sae_bsi, sae_sum = label_features(sae_test, test_labels, stat_names)

            # SAE ablation: use all hidden features, not just k
            n_sae_features = sae_test.shape[1]
            n_ab_sae = min(N_ABLATE, n_sae_features)
            sae_top = np.argsort(sae_bc)[-n_ab_sae:]
            sae_rand_feat = np.random.choice(n_sae_features, n_ab_sae, replace=False)
            sae_abl = sae.ablate(test_tensor, sae_top).cpu().float().numpy()
            sae_rand_abl = sae.ablate(test_tensor, sae_rand_feat).cpu().float().numpy()
            with torch.no_grad():
                sae_full_recon, _ = sae(test_tensor)
            sae_full_recon = sae_full_recon.detach().cpu().float().numpy()

            cos_sae_full = measure_ablation_effect(test_acts, sae_full_recon)
            cos_sae_corr = measure_ablation_effect(test_acts, sae_abl)
            cos_sae_rand = measure_ablation_effect(test_acts, sae_rand_abl)

            sr["sae"][exp_key] = {
                "var_explained": ve, "l0": l0, "alive": alive,
                "train_time_s": float(dt),
                "n_features": n_sae_features,
                "labeling": sae_sum,
                "ablation": {
                    "cos_full": cos_sae_full, "cos_corr_ablate": cos_sae_corr,
                    "cos_rand_ablate": cos_sae_rand,
                    "n_ablated": n_ab_sae,
                    "delta_corr": cos_sae_full - cos_sae_corr,
                    "delta_rand": cos_sae_full - cos_sae_rand,
                },
            }
            print(f"    VE={ve:.4f}, alive={alive:.1%}, max|r|={sae_sum['max_corr']:.4f}, "
                  f"above={sae_sum['n_above_threshold']}/{n_sae_features}, "
                  f"abl_delta={cos_sae_full - cos_sae_corr:.4f}")

        sr["time_s"] = float(time.time() - t_stock)
        all_results["stocks"].append(sr)
        print(f"  Done in {sr['time_s']:.0f}s")

    # ─── Aggregate ───
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    agg = {}
    for expansion in EXPANSIONS:
        exp_key = f"exp{expansion}x"
        keys_sae = ["labeling.max_corr", "labeling.mean_corr", "labeling.pct_above_threshold",
                     "var_explained", "alive", "ablation.delta_corr"]
        keys_shared = ["labeling.max_corr", "labeling.mean_corr", "labeling.pct_above_threshold",
                        "ablation.delta"]

        def collect(data, key_path):
            vals = []
            for s in data:
                obj = s
                for k in key_path.split("."):
                    obj = obj.get(k, {}) if isinstance(obj, dict) else None
                    if obj is None:
                        break
                if obj is not None:
                    vals.append(obj)
            return vals

        agg[exp_key] = {"sae": {}, "pca": {}, "random": {}}

        stocks = all_results["stocks"]
        for ks in keys_sae:
            ks_s = ks.replace("labeling.", "").replace("ablation.", "")
            v_sae = collect([s["sae"].get(exp_key, {}) for s in stocks], ks)
            if v_sae:
                agg[exp_key]["sae"][ks_s] = {"mean": float(np.mean(v_sae)), "std": float(np.std(v_sae)), "n": len(v_sae)}

        for ks in keys_shared:
            ks_s = ks.replace("labeling.", "").replace("ablation.", "")
            vp = collect([s["pca"] for s in stocks], ks)
            vr = collect([s["random"] for s in stocks], ks)
            if vp:
                agg[exp_key]["pca"][ks_s] = {"mean": float(np.mean(vp)), "std": float(np.std(vp)), "n": len(vp)}
            if vr:
                agg[exp_key]["random"][ks_s] = {"mean": float(np.mean(vr)), "std": float(np.std(vr)), "n": len(vr)}

    all_results["aggregate"] = agg

    # ─── Comparison Table ───
    for expansion in EXPANSIONS:
        exp_key = f"exp{expansion}x"
        a = agg[exp_key]
        print(f"\n--- SAE Expansion {expansion}x ---")
        print(f"{'Method':<12} {'Max|r|':<16} {'Mean|r|':<16} {'%>0.3':<10} "
              f"{'VE/cumVar':<12} {'AblDelta':<14} {'N':<5}")
        print("-" * 80)

        s = a.get("sae", {})
        p = a.get("pca", {})
        r = a.get("random", {})

        if s:
            print(f"{'SAE':<12} {s['max_corr']['mean']:.4f}+-{s['max_corr']['std']:.3f}  "
                  f"{s['mean_corr']['mean']:.4f}+-{s['mean_corr']['std']:.3f}  "
                  f"{s['pct_above_threshold']['mean']:.1f}%     "
                  f"{s.get('var_explained',{}).get('mean',0):.4f}     "
                  f"{s.get('delta_corr',{}).get('mean',0):.4f}+-{s.get('delta_corr',{}).get('std',0):.4f}  "
                  f"{s['max_corr']['n']:<5}")
        if p:
            print(f"{'PCA':<12} {p['max_corr']['mean']:.4f}+-{p['max_corr']['std']:.3f}  "
                  f"{p['mean_corr']['mean']:.4f}+-{p['mean_corr']['std']:.3f}  "
                  f"{p['pct_above_threshold']['mean']:.1f}%     "
                  f"{p.get('cum_var',{}).get('mean',0):.4f}     "
                  f"{p.get('delta',{}).get('mean',0):.4f}+-{p.get('delta',{}).get('std',0):.4f}  "
                  f"{p['max_corr']['n']:<5}")
        if r:
            print(f"{'Random':<12} {r['max_corr']['mean']:.4f}+-{r['max_corr']['std']:.3f}  "
                  f"{r['mean_corr']['mean']:.4f}+-{r['mean_corr']['std']:.3f}  "
                  f"{r['pct_above_threshold']['mean']:.1f}%     "
                  f"{'N/A':<12}"
                  f"{r.get('delta',{}).get('mean',0):.4f}+-{r.get('delta',{}).get('std',0):.4f}  "
                  f"{r['max_corr']['n']:<5}")

    # ─── Statistical Tests (SAE 4x vs baselines) ───
    exp_key = "exp4x"
    stocks = all_results["stocks"]
    s_max = [s["sae"].get(exp_key,{}).get("labeling",{}).get("max_corr",0) for s in stocks if exp_key in s.get("sae",{})]
    p_max = [s["pca"].get("labeling",{}).get("max_corr",0) for s in stocks if s.get("pca",{}).get("labeling")]
    r_max = [s["random"].get("labeling",{}).get("max_corr",0) for s in stocks if s.get("random",{}).get("labeling")]
    s_delta = [s["sae"].get(exp_key,{}).get("ablation",{}).get("delta_corr",0) for s in stocks if exp_key in s.get("sae",{})]
    p_delta = [s["pca"].get("ablation",{}).get("delta",0) for s in stocks if s.get("pca",{}).get("ablation")]
    r_delta = [s["random"].get("ablation",{}).get("delta",0) for s in stocks if s.get("random",{}).get("ablation")]

    nm = min(len(s_max), len(p_max), len(r_max))
    s_max, p_max, r_max = s_max[:nm], p_max[:nm], r_max[:nm]
    s_delta, p_delta, r_delta = s_delta[:nm], p_delta[:nm], r_delta[:nm]

    print(f"\n{'='*70}")
    print(f"STATISTICAL TESTS (SAE 4x vs baselines, n={nm})")
    print(f"{'='*70}")

    tests = {}
    if nm >= 3:
        for label, a_vals, b_vals in [
            ("sae_vs_pca_max_corr", s_max, p_max),
            ("sae_vs_random_max_corr", s_max, r_max),
            ("pca_vs_random_max_corr", p_max, r_max),
            ("sae_vs_pca_ablation_delta", s_delta, p_delta),
            ("sae_vs_random_ablation_delta", s_delta, r_delta),
        ]:
            t, pv = sp_stats.ttest_rel(a_vals, b_vals)
            tests[label] = {"t": float(t), "p": float(pv)}
            sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
            print(f"  {label}: t={t:.3f}, p={pv:.4f} {sig}")

        all_results["stats_tests"] = tests

    # Save
    out_path = OUTPUT_DIR / "baseline_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_t = time.time() - t0
    print(f"\nSaved to {out_path}")
    print(f"Total time: {total_t:.0f}s ({total_t/60:.1f} min)")
    print("Done.")


if __name__ == "__main__":
    main()
