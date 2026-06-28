"""Stability experiments: seed and layer robustness for Kronos SAEs + bootstrap CIs for three-way comparison.

Experiments:
  1. Seed stability: 5 stocks x 3 seeds (42, 123, 456) on Kronos layer 6
  2. Layer stability: 5 stocks x 5 layers (0, 3, 6, 9, 11) on Kronos
  3. Bootstrap 95% CIs for Kronos / Chronos / FinText comparison across 111 stocks
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from scipy import stats

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file
from transformers import T5Model


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

DEVICE = "cuda"  # CUDA_VISIBLE_DEVICES=5 set externally
KRONOS_MODEL_PATH = "/data/houwanlong/models/Kronos-base"
KRONOS_TOKENIZER_PATH = "/data/houwanlong/models/Kronos-Tokenizer-base"
CHRONOS_MODEL_PATH = "/data/houwanlong/models/chronos-t5-small"
FINTEXT_MODEL_PATH = "/data/houwanlong/models/FinText-Chronos-Small"
DATA_DIR = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/stability_results.json"
FINANCIAL_IMPACT_PATH = "/data/houwanlong/finllm-mi/outputs/sae/financial_impact.json"

K = 64
EXPANSION = 4
STEPS = 3000
BATCH = 512
LR = 1e-4
LOOKBACK = 64
STRIDE = 32
MAX_WINDOWS = 2000
SEEDS = [42, 123, 456]
LAYERS = [0, 3, 6, 9, 11]

SECTORS_FILE = "/tmp/sectors120.json"

# 5 stocks: one per sector + Alibaba
STOCK_SELECTION = {
    "Bank": "sh600000",      # SPD Bank
    "Energy": "sh600028",    # Sinopec
    "Tech": "sh600072",      # CSSC
    "Consumer": "sh600519",  # Kweichow Moutai
    "Alibaba": "HK_ali_09988",
}

ALI_CSV = "/data/houwanlong/finllm-mi/code/finetune_csv/data/HK_ali_09988_kline_5min_all.csv"

t_total = time.time()


# ═══════════════════════════════════════════════════════════════
# SAE Implementation
# ═══════════════════════════════════════════════════════════════

class TopKSAE(torch.nn.Module):
    def __init__(self, d_model, d_hidden, k=64):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = torch.nn.Linear(d_hidden, d_model, bias=False)
        self.b_pre = torch.nn.Parameter(torch.zeros(d_model))
        self.k = k

    def forward(self, x):
        xc = x - self.b_pre
        lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat)
        mask.scatter_(-1, idx, 1.0)
        return self.decoder(lat * mask) + self.b_pre, lat * mask

    def ablate_reconstruct(self, x, ids):
        xc = x - self.b_pre
        lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat)
        mask.scatter_(-1, idx, 1.0)
        mask[:, ids] = 0
        return self.decoder(lat * mask) + self.b_pre


# ═══════════════════════════════════════════════════════════════
# Financial Statistics
# ═══════════════════════════════════════════════════════════════

def compute_financial_stats(data_window):
    """Compute 30+ financial statistics over a 64-period OHLCV window."""
    close = data_window[:, 1]
    high = data_window[:, 2]
    low = data_window[:, 3]
    volume = data_window[:, 4]
    amount = data_window[:, 5]

    returns = np.diff(close) / (close[:-1] + 1e-5)
    T = len(returns)

    features = {}

    # Momentum (7)
    features["momentum_5"] = float(close[-1] / (close[-6] + 1e-5) - 1) if T >= 5 else 0.0
    features["momentum_10"] = float(close[-1] / (close[-11] + 1e-5) - 1) if T >= 10 else 0.0
    features["momentum_20"] = float(close[-1] / (close[-21] + 1e-5) - 1) if T >= 20 else 0.0
    features["momentum_64"] = float(close[-1] / (close[0] + 1e-5) - 1)

    features["ma_cross_5_20"] = float(np.mean(close[-5:]) - np.mean(close[-20:])) / (close.mean() + 1e-5) if T >= 20 else 0.0
    features["ma_cross_5_60"] = float(np.mean(close[-5:]) - np.mean(close[-60:])) / (close.mean() + 1e-5) if T >= 60 else 0.0

    gains = np.maximum(returns[-14:], 0)
    losses = np.abs(np.minimum(returns[-14:], 0))
    avg_gain = gains.mean() if len(gains) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 1e-5
    features["rsi_14"] = float(100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-5)))

    # Volatility (7)
    features["vol_realized"] = float(returns.std() * np.sqrt(T))
    features["vol_parkinson"] = float(np.sqrt(np.mean(np.log(high[:-1] / (low[:-1] + 1e-5))**2)))

    rolling_vols = np.array([returns[max(0, i-5):i+5].std() for i in range(0, T, 5)])
    features["vol_of_vol"] = float(rolling_vols.std()) if len(rolling_vols) > 1 else 0.0

    if len(rolling_vols) >= 4:
        features["vol_persistence"] = float(np.corrcoef(rolling_vols[1:], rolling_vols[:-1])[0, 1]) if np.std(rolling_vols[:-1]) > 0 else 0.0
    else:
        features["vol_persistence"] = 0.0

    sq_ret = returns**2
    features["vol_clustering"] = float(np.corrcoef(sq_ret[1:], sq_ret[:-1])[0, 1]) if len(sq_ret) > 1 and sq_ret.std() > 0 else 0.0

    features["close_range"] = float((close.max() - close.min()) / (close.mean() + 1e-5))
    features["hl_ratio_mean"] = float(np.mean(high[:-1] / (low[:-1] + 1e-5)))

    # Autocorrelation (4)
    features["autocorr_1"] = float(np.corrcoef(returns[1:], returns[:-1])[0, 1]) if T > 1 and returns.std() > 0 else 0.0
    features["autocorr_5"] = float(np.corrcoef(returns[5:], returns[:-5])[0, 1]) if T > 5 and returns.std() > 0 else 0.0

    if T > 20:
        rs_vals = []
        for lag in [8, 16, 32]:
            if lag <= T:
                segments = T // lag
                rs_seg = []
                for s in range(segments):
                    seg = returns[s*lag:(s+1)*lag]
                    if seg.std() > 1e-10:
                        mean_adj = seg - seg.mean()
                        cum = np.cumsum(mean_adj)
                        rs_seg.append((cum.max() - cum.min()) / seg.std())
                if rs_seg:
                    rs_vals.append(np.log(np.mean(rs_seg)))
        if len(rs_vals) >= 2:
            log_lags = np.log([8, 16, 32])[:len(rs_vals)]
            features["hurst"] = float(np.polyfit(log_lags, rs_vals, 1)[0])
        else:
            features["hurst"] = 0.5
    else:
        features["hurst"] = 0.5

    features["mean_rev"] = float(-features["autocorr_1"])

    # Tail Risk (7)
    features["var_95"] = float(-np.percentile(returns, 5)) if len(returns) > 0 else 0.0
    tail_returns = returns[returns <= np.percentile(returns, 5)]
    features["cvar_95"] = float(-tail_returns.mean()) if len(tail_returns) > 0 else features["var_95"]
    features["max_1d_loss"] = float(returns.min()) if len(returns) > 0 else 0.0
    features["max_1d_gain"] = float(returns.max()) if len(returns) > 0 else 0.0
    features["skewness"] = float(stats.skew(returns)) if len(returns) > 3 and returns.std() > 0 else 0.0
    features["kurtosis"] = float(stats.kurtosis(returns, fisher=True)) if len(returns) > 4 and returns.std() > 0 else 0.0
    s = features["skewness"]; k = features["kurtosis"]
    features["jarque_bera"] = float(T / 6.0 * (s**2 + k**2 / 4.0))

    # Price Structure (5)
    features["trend_slope"] = float(np.polyfit(np.arange(len(close)), close, 1)[0])
    features["trend_r2"] = float(np.corrcoef(np.arange(len(close)), close)[0, 1])**2
    cummax = np.maximum.accumulate(close)
    features["max_drawdown"] = float(np.min(close / (cummax + 1e-5) - 1))
    features["price_range"] = float((close.max() - close.min()) / close.mean())
    features["close_to_close"] = float(close[-1] / (close[0] + 1e-5) - 1)

    # Volume (4)
    vol_ret = np.diff(volume) / (volume[:-1] + 1e-5)
    features["volume_trend"] = float(np.polyfit(np.arange(len(volume)), volume, 1)[0]) / (volume.mean() + 1e-5)
    features["volume_volatility"] = float(vol_ret.std()) if len(vol_ret) > 0 else 0.0
    features["volume_price_corr"] = float(np.corrcoef(returns, vol_ret)[0, 1]) if T > 1 and returns.std() > 0 and vol_ret.std() > 0 else 0.0
    features["volume_ratio"] = float(volume[-10:].mean() / (volume[:-10].mean() + 1e-5)) if T > 10 else 1.0

    # Clean up
    for k in features:
        if np.isnan(features[k]) or np.isinf(features[k]):
            features[k] = 0.0

    return features


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_stock_windows(ticker, max_windows=MAX_WINDOWS):
    """Load and normalize OHLCV windows for a given stock ticker.

    Returns (windows, labels_list) where windows is (N, L, 6) and labels is list of dicts.
    """
    if ticker == "HK_ali_09988":
        csv_path = ALI_CSV
    else:
        csv_path = DATA_DIR / f"{ticker}.csv"

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(str(csv_path))
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        if col not in df.columns:
            df[col] = 0.0
    data = df[["open", "close", "high", "low", "volume", "amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]

    if len(data) < 200:
        raise ValueError(f"Too few rows: {len(data)}")

    mn, st = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)

    n_windows = min(max_windows, (len(data_norm) - LOOKBACK) // STRIDE)
    if n_windows < 10:
        raise ValueError(f"Too few windows: {n_windows}")

    windows = np.stack([data_norm[i:i+LOOKBACK] for i in range(0, n_windows * STRIDE, STRIDE)])

    labels_list = []
    for i in range(0, n_windows * STRIDE, STRIDE):
        win = data_norm[i:i+LOOKBACK]
        labels_list.append(compute_financial_stats(win))

    return windows, labels_list


# ═══════════════════════════════════════════════════════════════
# Model Loading & Extraction
# ═══════════════════════════════════════════════════════════════

def load_kronos(device):
    """Load Kronos tokenizer + model."""
    tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER_PATH).to(device).eval()

    with open(f"{KRONOS_MODEL_PATH}/config.json") as f:
        cfg = json.load(f)
    model = Kronos(
        s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"],
        n_layers=cfg["n_layers"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
        ffn_dropout_p=cfg["ffn_dropout_p"], attn_dropout_p=cfg["attn_dropout_p"],
        resid_dropout_p=cfg["resid_dropout_p"], token_dropout_p=cfg["token_dropout_p"],
        learn_te=cfg["learn_te"],
    )
    sd = load_file(f"{KRONOS_MODEL_PATH}/model.safetensors")
    model.load_state_dict(sd, strict=False)
    model = model.to(device).half().eval()
    d_model = cfg["d_model"]
    d_hidden = d_model * EXPANSION
    return model, tokenizer, d_model, d_hidden


def extract_kronos_activations(windows, tokenizer, model, layer, device, batch_size=64):
    """Extract Kronos layer activations from OHLCV windows."""
    acts_list = []

    def hook_fn(m, i, o):
        a = o[0] if isinstance(o, tuple) else o
        acts_list.append(a[:, -1, :].detach().cpu().float().numpy())

    hook = model.transformer[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        for b in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[b:b+batch_size]).float().to(device)
            s1, s2 = tokenizer.encode(batch, half=True)
            model(s1, s2)

    hook.remove()
    return np.concatenate(acts_list, axis=0)


# ═══════════════════════════════════════════════════════════════
# SAE Training & Eval
# ═══════════════════════════════════════════════════════════════

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

import random

def train_sae(activations, d_model, d_hidden, k, device, steps=STEPS, batch_size=BATCH, lr=LR):
    """Train a TopK SAE on activations."""
    sae = TopKSAE(d_model, d_hidden, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    acts_t = torch.from_numpy(activations).float().to(device)

    for step in range(steps):
        idx = torch.randint(0, len(acts_t), (batch_size,), device=device)
        xr, _ = sae(acts_t[idx])
        loss = torch.nn.functional.mse_loss(xr, acts_t[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        xt = acts_t[:min(500, len(acts_t))]
        recon, lat = sae(xt)
        ve = 1 - torch.nn.functional.mse_loss(recon, xt).item() / (xt.var().item() + 1e-10)
        l0 = (lat != 0).float().sum(-1).mean().item()
        alive = (lat.abs().sum(0) > 1e-6).float().mean().item()

    return sae, {"var_exp": float(ve), "l0": float(l0), "alive": float(alive)}


def label_features(full_lat, label_dicts, alive_thresh=10, act_thresh=5):
    """Label each alive SAE feature by strongest-correlated financial statistic."""
    label_keys = sorted(label_dicts[0].keys())
    label_names = [k.replace("_", " ").title() for k in label_keys]
    n = min(len(full_lat), len(label_dicts))
    label_arr = np.array([[l[k] for k in label_keys] for l in label_dicts[:n]])

    alive_mask = (full_lat != 0).sum(0) > alive_thresh
    type_dist = {}
    n_strong = 0
    n_alive = int(alive_mask.sum())
    feature_labels = []

    for j in np.where(alive_mask)[0]:
        act = full_lat[:, j]
        a = act != 0
        if a.sum() < act_thresh:
            continue
        corrs = []
        for k in range(len(label_keys)):
            try:
                c = np.corrcoef(act[a], label_arr[a, k])[0, 1]
                c = 0.0 if np.isnan(c) else c
            except Exception:
                c = 0.0
            corrs.append(c)
        best = np.argmax(np.abs(corrs))
        best_corr = corrs[best]
        type_dist[label_names[best]] = type_dist.get(label_names[best], 0) + 1
        if abs(best_corr) > 0.3:
            n_strong += 1
        feature_labels.append((int(j), label_names[best], float(best_corr)))

    return type_dist, n_strong, n_alive, feature_labels


def compute_concept_distribution_entropy(type_dist):
    """Compute entropy of the concept distribution."""
    total = sum(type_dist.values())
    if total == 0:
        return 0.0, 0.0
    probs = np.array([v / total for v in type_dist.values()])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    largest_pct = float(probs.max() * 100)
    return float(entropy), largest_pct


def compute_ablation_effect(model, tokenizer, sae, layer, test_windows, feature_ids, device):
    """Ablate SAE features and measure output change (1 - cos_sim)."""
    test_t = torch.from_numpy(test_windows).float().to(device)

    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_t, half=True)
        base = model(s1, s2)
    base_s1 = base[0].float()

    def make_intervene(ab_ids):
        def intervene(m, i, o):
            orig = o[0] if isinstance(o, tuple) else o
            B, T_val, D = orig.shape
            ablated = sae.ablate_reconstruct(orig.reshape(-1, D).float(), ab_ids).reshape(B, T_val, D).half()
            return (ablated,) + o[1:] if isinstance(o, tuple) else ablated
        return intervene

    hk = model.transformer[layer].register_forward_hook(make_intervene(feature_ids))
    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_t, half=True)
        ab = model(s1, s2)
    hk.remove()

    cs = torch.nn.functional.cosine_similarity(
        base_s1.reshape(-1, base_s1.shape[-1]),
        ab[0].float().reshape(-1, base_s1.shape[-1]), dim=-1).mean()
    return float(1.0 - cs.item())


def full_evaluation(model, tokenizer, d_model, d_hidden, windows, labels_list, layer, device, seed=None):
    """Train SAE and compute all evaluation metrics."""
    if seed is not None:
        set_seed(seed)

    acts = extract_kronos_activations(windows, tokenizer, model, layer, device)
    n = min(len(acts), len(labels_list))
    acts = acts[:n]

    t0 = time.time()
    sae, info = train_sae(acts, d_model, d_hidden, K, device)

    # Get full latent activations
    acts_t = torch.from_numpy(acts).float().to(device)
    with torch.no_grad():
        _, full_lat = sae(acts_t)
    lat_np = full_lat.cpu().numpy()

    # Label features
    type_dist, n_strong, n_alive, feat_labels = label_features(lat_np, labels_list[:n])

    # Entropy and largest family
    entropy, largest_family_pct = compute_concept_distribution_entropy(type_dist)

    # Ablation effect
    _, test_lat = sae(acts_t[:min(1000, len(acts_t))])
    freq = (test_lat != 0).float().sum(0)
    top50 = freq.argsort(descending=True)[:50].tolist()
    test_start = max(0, len(windows) - min(30, len(windows)))
    test_wins = windows[test_start:]
    if len(test_wins) >= 5:
        ab_effect = compute_ablation_effect(model, tokenizer, sae, layer, test_wins, top50, device)
    else:
        ab_effect = None

    train_time = time.time() - t0

    # Clean up
    del sae, acts_t
    torch.cuda.empty_cache()

    return {
        "ve": info["var_exp"],
        "l0": info["l0"],
        "alive": info["alive"],
        "n_alive": n_alive,
        "n_strong": n_strong,
        "entropy": entropy,
        "largest_family_pct": largest_family_pct,
        "ablation_effect": ab_effect,
        "type_distribution": {k: int(v) for k, v in sorted(type_dist.items(), key=lambda x: -x[1])},
        "train_time": train_time,
    }


# ═══════════════════════════════════════════════════════════════
# Bootstrap
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(values, n_bootstrap=10000, ci=95):
    """Compute bootstrap confidence interval for an array of values."""
    values = np.array(values, dtype=float)
    # Remove None/NaN
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return {"mean": float(np.mean(values)) if len(values) > 0 else 0.0,
                "ci_lower": None, "ci_upper": None, "n": len(values)}

    lo = (100 - ci) / 2
    hi = 100 - lo
    boot_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(float(np.mean(sample)))

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci_lower": float(np.percentile(boot_means, lo)),
        "ci_upper": float(np.percentile(boot_means, hi)),
        "n": len(values),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("STABILITY EXPERIMENTS: SEED + LAYER ROBUSTNESS")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Stocks: {list(STOCK_SELECTION.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"Layers: {LAYERS}")
    print()

    # ─── Load Kronos once ───
    print("Loading Kronos model...")
    model, tokenizer, d_model, d_hidden = load_kronos(DEVICE)
    n_model_layers = len(model.transformer)
    print(f"  Kronos loaded: {n_model_layers} layers, d_model={d_model}, d_hidden={d_hidden}")
    print()

    results = {
        "experiment": "stability_analysis",
        "config": {
            "k": K,
            "expansion": EXPANSION,
            "steps": STEPS,
            "batch_size": BATCH,
            "lr": LR,
            "d_model": d_model,
            "d_hidden": d_hidden,
            "lookback": LOOKBACK,
            "stride": STRIDE,
            "seeds": SEEDS,
            "layers": LAYERS,
        },
        "stocks_used": STOCK_SELECTION,
    }

    # ═══════════════════════════════════════════════════════
    # Experiment 1: Seed stability (layer 6, 3 seeds)
    # ═══════════════════════════════════════════════════════
    print("=" * 70)
    print("EXPERIMENT 1: Seed Stability (Layer 6)")
    print("=" * 70)

    seed_results = {}  # ticker -> {seed: metrics}

    for label, ticker in STOCK_SELECTION.items():
        print(f"\n{'─' * 60}")
        print(f"  {label}: {ticker}")
        try:
            windows, labels_list = load_stock_windows(ticker)
            print(f"    Loaded {len(windows)} windows")
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        seed_results[ticker] = {}
        for seed in SEEDS:
            print(f"    Seed {seed}...", end=" ", flush=True)
            t0 = time.time()
            metrics = full_evaluation(model, tokenizer, d_model, d_hidden,
                                       windows, labels_list, layer=6,
                                       device=DEVICE, seed=seed)
            seed_results[ticker][str(seed)] = metrics
            dt = time.time() - t0
            print(f"VE={metrics['ve']:.4f} alive={metrics['alive']:.1%} "
                  f"entropy={metrics['entropy']:.3f} largest={metrics['largest_family_pct']:.1f}% "
                  f"ab_effect={metrics['ablation_effect']:.4f} ({dt:.0f}s)")

    # Aggregate across seeds per stock
    seed_aggregates = {}
    for ticker, sd in seed_results.items():
        metrics_keys = ["ve", "alive", "n_alive", "n_strong", "entropy", "largest_family_pct", "ablation_effect"]
        agg = {}
        for key in metrics_keys:
            vals = [sd[s][key] for s in sd if sd[s].get(key) is not None]
            if vals:
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}
        seed_aggregates[ticker] = agg

    results["seed_stability"] = {
        "layer": 6,
        "seeds": SEEDS,
        "per_stock_per_seed": seed_results,
        "per_stock_aggregates": seed_aggregates,
    }

    # Global seed stability summary
    all_seed_ve = []
    all_seed_alive = []
    all_seed_entropy = []
    all_seed_largest = []
    all_seed_ab = []
    for ticker, agg in seed_aggregates.items():
        for key, vals_list in [
            ("ve", all_seed_ve),
            ("alive", all_seed_alive),
            ("entropy", all_seed_entropy),
            ("largest_family_pct", all_seed_largest),
            ("ablation_effect", all_seed_ab),
        ]:
            n_vals = len(agg[key].get("values", []))
            std_val = agg[key]["std"] if n_vals > 1 else 0.0
            mean_val = agg[key]["mean"]
            vals_list.append(std_val / (mean_val + 1e-10))  # CV

    print(f"\n  Seed Stability Summary (CV across 3 seeds):")
    print(f"    VE CV: {np.mean(all_seed_ve)*100:.3f}%")
    print(f"    Alive % CV: {np.mean(all_seed_alive)*100:.2f}%")
    print(f"    Entropy CV: {np.mean(all_seed_entropy)*100:.2f}%")
    print(f"    Largest Family CV: {np.mean(all_seed_largest)*100:.2f}%")
    print(f"    Ablation Effect CV: {np.mean(all_seed_ab)*100:.2f}%")

    # ═══════════════════════════════════════════════════════
    # Experiment 2: Layer stability (5 layers, seed=42)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 2: Layer Stability (seed=42)")
    print("=" * 70)

    layer_results = {}  # ticker -> {layer: metrics}

    for label, ticker in STOCK_SELECTION.items():
        print(f"\n{'─' * 60}")
        print(f"  {label}: {ticker}")
        try:
            windows, labels_list = load_stock_windows(ticker)
            print(f"    Loaded {len(windows)} windows")
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        layer_results[ticker] = {}
        for layer in LAYERS:
            if layer >= n_model_layers:
                print(f"    Layer {layer}: SKIP (model has {n_model_layers} layers)")
                continue
            print(f"    Layer {layer}...", end=" ", flush=True)
            t0 = time.time()
            metrics = full_evaluation(model, tokenizer, d_model, d_hidden,
                                       windows, labels_list, layer=layer,
                                       device=DEVICE, seed=42)
            layer_results[ticker][str(layer)] = metrics
            dt = time.time() - t0
            print(f"VE={metrics['ve']:.4f} alive={metrics['alive']:.1%} "
                  f"entropy={metrics['entropy']:.3f} largest={metrics['largest_family_pct']:.1f}% "
                  f"ab_effect={metrics.get('ablation_effect','N/A')} ({dt:.0f}s)")

    # Aggregate across layers per stock
    layer_aggregates = {}
    for ticker, ld in layer_results.items():
        metrics_keys = ["ve", "alive", "n_alive", "n_strong", "entropy", "largest_family_pct", "ablation_effect"]
        agg = {}
        for key in metrics_keys:
            vals = [ld[l][key] for l in ld if ld[l].get(key) is not None]
            if vals:
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}
        layer_aggregates[ticker] = agg

    results["layer_stability"] = {
        "seed": 42,
        "layers": LAYERS,
        "per_stock_per_layer": layer_results,
        "per_stock_aggregates": layer_aggregates,
    }

    # Global layer stability summary
    all_layer_ve = []
    all_layer_alive = []
    all_layer_entropy = []
    all_layer_largest = []
    all_layer_ab = []
    for ticker, agg in layer_aggregates.items():
        for key, vals_list in [
            ("ve", all_layer_ve),
            ("alive", all_layer_alive),
            ("entropy", all_layer_entropy),
            ("largest_family_pct", all_layer_largest),
            ("ablation_effect", all_layer_ab),
        ]:
            n_vals = len(agg[key].get("values", []))
            mean_val = agg[key]["mean"]
            std_val = agg[key]["std"] if n_vals > 1 else 0.0
            vals_list.append(std_val / (mean_val + 1e-10))

    print(f"\n  Layer Stability Summary (CV across {len(LAYERS)} layers):")
    print(f"    VE CV: {np.mean(all_layer_ve)*100:.3f}%")
    print(f"    Alive % CV: {np.mean(all_layer_alive)*100:.2f}%")
    print(f"    Entropy CV: {np.mean(all_layer_entropy)*100:.2f}%")
    print(f"    Largest Family CV: {np.mean(all_layer_largest)*100:.2f}%")
    print(f"    Ablation Effect CV: {np.mean(all_layer_ab)*100:.2f}%")

    # ═══════════════════════════════════════════════════════
    # Experiment 3: Bootstrap CIs for three-way comparison
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 3: Bootstrap CIs for Kronos / Chronos / FinText")
    print("=" * 70)

    bootstrap_results = {"models": {}}

    # ── Load FinText results ──
    fintext_path = "/data/houwanlong/finllm-mi/outputs/sae/fintext_chronos_results.json"
    if Path(fintext_path).exists():
        print("\n  Loading FinText results...")
        with open(fintext_path) as f:
            ft_data = json.load(f)
        ft_stocks = ft_data.get("per_stock", [])
        ft_metrics = {
            "largest_family_pct": [],
            "entropy": [],
            "n_strong": [],
            "alive_pct": [],
            "ablation_effect": [],
        }
        for ps in ft_stocks:
            for k in ft_metrics:
                v = ps.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    # Normalize: FinText stores fractions, convert to percentages
                    val = float(v)
                    if k == "largest_family_pct":
                        val *= 100
                    elif k == "alive_pct":
                        val *= 100
                    ft_metrics[k].append(val)

        ft_cis = {}
        for k, vals in ft_metrics.items():
            ft_cis[k] = bootstrap_ci(vals)
            print(f"    FinText {k}: {ft_cis[k]['mean']:.4f} [{ft_cis[k]['ci_lower']:.4f}, {ft_cis[k]['ci_upper']:.4f}] (n={ft_cis[k]['n']})")
        bootstrap_results["models"]["FinText"] = {
            "model_path": FINTEXT_MODEL_PATH,
            "n_stocks": len(ft_stocks),
            "bootstrap_95ci": ft_cis,
        }
    else:
        print(f"    FinText results not found at {fintext_path}")

    # ── Load Kronos results (scale120) ──
    kronos_path = "/data/houwanlong/finllm-mi/outputs/sae/scale120_results.json"
    if Path(kronos_path).exists():
        print("\n  Loading Kronos results (scale120)...")
        with open(kronos_path) as f:
            kr_data = json.load(f)
        kr_stocks = kr_data.get("per_stock", [])
        kr_metrics = {
            "largest_family_pct": [],
            "entropy": [],
            "n_strong": [],
            "alive_pct": [],
            "ablation_effect": [],
        }
        for ps in kr_stocks:
            # Compute largest_family_pct and entropy from type_dist
            td = ps.get("type_dist", {})
            total = sum(td.values())
            if total > 0:
                probs = np.array([v / total for v in td.values()])
                entropy_val = float(-np.sum(probs * np.log(probs + 1e-10)))
                largest_pct = float(probs.max() * 100)
            else:
                entropy_val = 0.0
                largest_pct = 0.0
            kr_metrics["largest_family_pct"].append(largest_pct)
            kr_metrics["entropy"].append(entropy_val)
            kr_metrics["n_strong"].append(float(ps.get("n_strong", 0)))
            alive_val = float(ps.get("alive", 0)) if ps.get("alive") is not None else 0.0
            kr_metrics["alive_pct"].append(alive_val * 100)
            ie = ps.get("intervention_effect")
            if ie is not None:
                kr_metrics["ablation_effect"].append(float(ie))

        kr_cis = {}
        for k, vals in kr_metrics.items():
            kr_cis[k] = bootstrap_ci(vals)
            print(f"    Kronos {k}: {kr_cis[k]['mean']:.4f} [{kr_cis[k]['ci_lower']:.4f}, {kr_cis[k]['ci_upper']:.4f}] (n={kr_cis[k]['n']})")
        bootstrap_results["models"]["Kronos"] = {
            "model_path": KRONOS_MODEL_PATH,
            "n_stocks": len(kr_stocks),
            "bootstrap_95ci": kr_cis,
        }
    else:
        print(f"    Kronos results not found at {kronos_path}")

    # ── Load Chronos results ──
    chronos_path = "/data/houwanlong/finllm-mi/outputs/sae/chronos_sae_results.json"
    if Path(chronos_path).exists():
        print("\n  Loading Chronos results...")
        with open(chronos_path) as f:
            ch_data = json.load(f)
        ch_per_stock = ch_data.get("per_stock_chronos", {})
        ch_metrics = {
            "largest_family_pct": [],
            "entropy": [],
            "n_strong": [],
            "alive_pct": [],
            "ablation_effect": [],
        }
        for ticker, ps in ch_per_stock.items():
            td = ps.get("type_distribution", {})
            total = sum(td.values())
            if total > 0:
                probs = np.array([v / total for v in td.values()])
                entropy_val = float(-np.sum(probs * np.log(probs + 1e-10)))
                largest_pct = float(probs.max() * 100)
            else:
                entropy_val = 0.0
                largest_pct = 0.0
            ch_metrics["largest_family_pct"].append(largest_pct)
            ch_metrics["entropy"].append(entropy_val)
            ch_metrics["n_strong"].append(float(ps.get("n_strong", 0)))
            ch_metrics["alive_pct"].append(float(ps.get("alive", 0)) * 100)
            ae = ps.get("ablation_effect")
            if ae is not None:
                ch_metrics["ablation_effect"].append(float(ae))

        ch_cis = {}
        for k, vals in ch_metrics.items():
            ch_cis[k] = bootstrap_ci(vals)
            print(f"    Chronos {k}: {ch_cis[k]['mean']:.4f} [{ch_cis[k]['ci_lower']:.4f}, {ch_cis[k]['ci_upper']:.4f}] (n={ch_cis[k]['n']})")
        bootstrap_results["models"]["Chronos"] = {
            "model_path": CHRONOS_MODEL_PATH,
            "n_stocks": len(ch_per_stock),
            "bootstrap_95ci": ch_cis,
        }
    else:
        print(f"    Chronos results not found at {chronos_path}")

    # ── Three-way comparison table ──
    print(f"\n{'=' * 70}")
    print("THREE-WAY COMPARISON TABLE (95% Bootstrap CI)")
    print("=" * 70)

    metrics_table = ["largest_family_pct", "entropy", "n_strong", "alive_pct", "ablation_effect"]
    metric_names = ["Largest Family %", "Entropy", "N Strong", "Alive %", "Ablation Effect"]
    model_names = ["Kronos", "Chronos", "FinText"]

    header = f"{'Metric':<22}"
    for m in model_names:
        header += f" {m:>28}"
    print(header)
    print("-" * len(header))

    for i, mk in enumerate(metrics_table):
        row = f"{metric_names[i]:<22}"
        for mn in model_names:
            if mn in bootstrap_results["models"]:
                ci = bootstrap_results["models"][mn]["bootstrap_95ci"].get(mk, {})
                if ci and ci.get("ci_lower") is not None:
                    row += f" {ci['mean']:>10.4f} [{ci['ci_lower']:>7.4f}, {ci['ci_upper']:<7.4f}]"
                else:
                    row += f" {ci.get('mean', float('nan')):>10.4f} {'(n/a)':>18}"
            else:
                row += f" {'N/A':>28}"
        print(row)

    results["three_way_bootstrap"] = bootstrap_results

    # ═══════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {OUTPUT}")
    print(f"Total time: {time.time() - t_total:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
