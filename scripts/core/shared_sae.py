"""Experiment 1: Shared SAE across 20 stocks (instead of per-stock SAEs).

Key questions:
- Does the shared SAE produce stronger signals than per-stock SAEs?
- Is the concept distribution more consistent?
- Are ablation effects larger?
"""
import torch, numpy as np, json, sys, time, os, warnings
from pathlib import Path
import pandas as pd
from scipy import stats
from collections import defaultdict

warnings.filterwarnings("ignore")

sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file


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
        out = self.decoder(lat * mask) + self.b_pre
        return out, lat * mask

    def ablate_reconstruct(self, x, ids):
        xc = x - self.b_pre
        lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat)
        mask.scatter_(-1, idx, 1.0)
        mask[:, ids] = 0
        return self.decoder(lat * mask) + self.b_pre


# ─── 30+ Financial Statistics ───
def compute_financial_stats(data_window, close_prices=None):
    """Compute 30+ financial statistics over a 64-period OHLCV window.

    Args:
        data_window: (64, 6) OHLCV array [open, close, high, low, volume, amount]
        close_prices: optional raw close prices

    Returns:
        dict of 33 financial statistics
    """
    close = data_window[:, 1]
    open_p = data_window[:, 0]
    high = data_window[:, 2]
    low = data_window[:, 3]
    volume = data_window[:, 4]
    amount = data_window[:, 5]

    returns = np.diff(close) / (close[:-1] + 1e-5)
    log_returns = np.log(close[1:] / (close[:-1] + 1e-5))
    T = len(returns)

    features = {}

    # ── Momentum (7) ──
    features["momentum_5"] = float(close[-1] / (close[-6] + 1e-5) - 1) if T >= 5 else 0.0
    features["momentum_10"] = float(close[-1] / (close[-11] + 1e-5) - 1) if T >= 10 else 0.0
    features["momentum_20"] = float(close[-1] / (close[-21] + 1e-5) - 1) if T >= 20 else 0.0
    features["momentum_64"] = float(close[-1] / (close[0] + 1e-5) - 1)

    features["ma_cross_5_20"] = float(np.mean(close[-5:]) - np.mean(close[-20:])) / (close.mean() + 1e-5) if T >= 20 else 0.0
    features["ma_cross_5_60"] = float(np.mean(close[-5:]) - np.mean(close[-60:])) / (close.mean() + 1e-5) if T >= 60 else 0.0

    # RSI-like: average gain / average loss
    gains = np.maximum(returns[-14:], 0)
    losses = np.abs(np.minimum(returns[-14:], 0))
    avg_gain = gains.mean() if len(gains) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 1e-5
    features["rsi_14"] = float(100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-5)))

    # ── Volatility (8) ──
    features["vol_realized"] = float(returns.std() * np.sqrt(T))  # annualized
    features["vol_parkinson"] = float(np.sqrt(np.mean(np.log(high[:-1] / (low[:-1] + 1e-5))**2)))
    features["vol_gk"] = float(np.sqrt(0.5 * np.mean(np.log(high[:-1] / (low[:-1] + 1e-5))**2)
                                       - (2 * np.log(2) - 1) * np.mean(np.log(close[1:] / (open_p[1:] + 1e-5))**2)))

    # Volatility of volatility
    rolling_vols = np.array([returns[max(0, i-5):i+5].std() for i in range(0, T, 5)])
    features["vol_of_vol"] = float(rolling_vols.std()) if len(rolling_vols) > 1 else 0.0

    # Volatility persistence: AR(1) on rolling volatility
    if len(rolling_vols) >= 4:
        lag = rolling_vols[1:]
        cur = rolling_vols[:-1]
        features["vol_persistence"] = float(np.corrcoef(lag, cur)[0, 1]) if np.std(cur) > 0 else 0.0
    else:
        features["vol_persistence"] = 0.0

    # Volatility clustering: autocorrelation of squared returns
    sq_ret = returns**2
    features["vol_clustering"] = float(np.corrcoef(sq_ret[1:], sq_ret[:-1])[0, 1]) if len(sq_ret) > 1 and sq_ret.std() > 0 else 0.0

    # High-low volatility ratio
    features["hl_vol_ratio"] = float((high[-1] / (low[-1] + 1e-5) - 1) / (features["vol_realized"] / np.sqrt(T) + 1e-5))

    # Close-to-close range
    features["close_range"] = float((close.max() - close.min()) / (close.mean() + 1e-5))

    # ── Autocorrelation (4) ──
    features["autocorr_1"] = float(np.corrcoef(returns[1:], returns[:-1])[0, 1]) if T > 1 and returns.std() > 0 else 0.0
    features["autocorr_5"] = float(np.corrcoef(returns[5:], returns[:-5])[0, 1]) if T > 5 and returns.std() > 0 else 0.0

    # Hurst exponent (simple R/S estimator)
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

    # Mean reversion: negative return autocorrelation
    features["mean_rev"] = float(-features["autocorr_1"])

    # ── Tail Risk (7) ──
    features["var_95"] = float(-np.percentile(returns, 5)) if len(returns) > 0 else 0.0
    tail_returns = returns[returns <= np.percentile(returns, 5)]
    features["cvar_95"] = float(-tail_returns.mean()) if len(tail_returns) > 0 else features["var_95"]

    features["max_1d_loss"] = float(returns.min()) if len(returns) > 0 else 0.0
    features["max_1d_gain"] = float(returns.max()) if len(returns) > 0 else 0.0

    features["skewness"] = float(stats.skew(returns)) if len(returns) > 3 and returns.std() > 0 else 0.0
    features["kurtosis"] = float(stats.kurtosis(returns, fisher=True)) if len(returns) > 4 and returns.std() > 0 else 0.0

    # Jarque-Bera
    s = features["skewness"]
    k = features["kurtosis"]
    jb = T / 6.0 * (s**2 + k**2 / 4.0)
    features["jarque_bera"] = float(jb)

    # ── Price Structure (5) ──
    features["trend_slope"] = float(np.polyfit(np.arange(len(close)), close, 1)[0])
    features["trend_r2"] = float(np.corrcoef(np.arange(len(close)), close)[0, 1])**2

    cummax = np.maximum.accumulate(close)
    features["max_drawdown"] = float(np.min(close / (cummax + 1e-5) - 1))

    features["price_range"] = float((close.max() - close.min()) / close.mean())
    features["close_to_close"] = float(close[-1] / (close[0] + 1e-5) - 1)

    # ── Volume (5) ──
    vol_ret = np.diff(volume) / (volume[:-1] + 1e-5)
    features["volume_trend"] = float(np.polyfit(np.arange(len(volume)), volume, 1)[0]) / (volume.mean() + 1e-5)
    features["volume_volatility"] = float(vol_ret.std()) if len(vol_ret) > 0 else 0.0
    features["volume_price_corr"] = float(np.corrcoef(returns, vol_ret)[0, 1]) if T > 1 and returns.std() > 0 and vol_ret.std() > 0 else 0.0

    # Volume ratio (recent / historical)
    features["volume_ratio"] = float(volume[-10:].mean() / (volume[:-10].mean() + 1e-5)) if T > 10 else 1.0

    # Amount volatility
    amt_ret = np.diff(amount) / (amount[:-1] + 1e-5)
    features["amount_volatility"] = float(amt_ret.std()) if len(amt_ret) > 0 else 0.0

    # Replace NaN/Inf with 0
    for k in features:
        if np.isnan(features[k]) or np.isinf(features[k]):
            features[k] = 0.0

    return features


def extract_activations(windows, tokenizer, model, layer, device, batch_size=64):
    """Extract layer activations from Kronos for given windows."""
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


def train_sae(activations, d_model, d_hidden, k, device, steps=3000, batch_size=512, lr=1e-4):
    """Train a TopK SAE on activations."""
    sae = TopKSAE(d_model, d_hidden, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    acts_t = torch.from_numpy(activations).float().to(device)

    for step in range(steps):
        idx = torch.randint(0, len(acts_t), (batch_size,))
        xr, _ = sae(acts_t[idx])
        loss = torch.nn.functional.mse_loss(xr, acts_t[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()

    # Evaluate
    with torch.no_grad():
        xt = acts_t[:min(500, len(acts_t))]
        recon, lat = sae(xt)
        ve = 1 - torch.nn.functional.mse_loss(recon, xt).item() / (xt.var().item() + 1e-10)
        l0 = (lat != 0).float().sum(-1).mean().item()
        alive = (lat.abs().sum(0) > 1e-6).float().mean().item()

    return sae, {"var_exp": float(ve), "l0": float(l0), "alive": float(alive)}


def label_features(full_lat, label_dicts, alive_thresh=10, act_thresh=5):
    """Label each alive SAE feature by strongest-correlated financial statistic.

    Returns:
        type_dist: dict of {stat_name: count}
        n_strong: number of features with |r| > 0.3
        feature_labels: list of (feature_idx, label_name, corr, corr_null_calibrated)
    """
    label_keys = sorted(label_dicts[0].keys())
    label_names = [k.replace("_", " ").title() for k in label_keys]
    n_labels = min(len(full_lat), len(label_dicts))
    label_arr = np.array([[l[k] for k in label_keys] for l in label_dicts[:n_labels]])

    alive_mask = (full_lat != 0).sum(0) > alive_thresh
    type_dist = {}
    n_strong = 0
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
        feature_labels.append((int(j), label_names[best], float(best_corr), label_keys[best]))

    return type_dist, n_strong, int(alive_mask.sum()), feature_labels


def null_calibrated_threshold(full_lat, label_dicts, n_shuffle=100):
    """Calibrate correlation threshold via label shuffling."""
    label_keys = sorted(label_dicts[0].keys())
    n = min(len(full_lat), len(label_dicts))
    label_arr = np.array([[l[k] for k in label_keys] for l in label_dicts[:n]])

    max_corrs = []
    alive_mask = (full_lat != 0).sum(0) > 10
    for _ in range(n_shuffle):
        perm_idx = np.random.permutation(n)
        batch_max = []
        for j in np.where(alive_mask)[0]:
            act = full_lat[:, j]
            a = act != 0
            if a.sum() < 5:
                continue
            try:
                corrs = [abs(np.corrcoef(act[a], label_arr[perm_idx][a, k])[0, 1]) for k in range(len(label_keys))]
                batch_max.append(np.nanmax([c for c in corrs if not np.isnan(c)]))
            except Exception:
                pass
        if batch_max:
            max_corrs.append(np.percentile(batch_max, 95))
    return float(np.median(max_corrs)) if max_corrs else 0.3


def compute_ablation_effect(model, tokenizer, sae, layer, test_windows, feature_ids, device):
    """Compute model output change when ablating specific SAE features."""
    test_t = torch.from_numpy(test_windows).float().to(device)

    # Baseline forward pass
    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_t, half=True)
        base = model(s1, s2)
    base_s1 = base[0].float()

    # Ablation forward pass
    def make_intervene(ab_ids):
        def intervene(m, i, o):
            orig = o[0] if isinstance(o, tuple) else o
            B, T, D = orig.shape
            ablated = sae.ablate_reconstruct(orig.reshape(-1, D).float(), ab_ids).reshape(B, T, D).half()
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
    return 1.0 - cs.item()


def main():
    t_total = time.time()
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES remaps, so use cuda:0
    layer = 6
    K = 64
    EXPANSION = 4
    N_STOCKS = 20
    STEPS = 3000
    BATCH = 512

    DATA_DIR = Path("/data/houwanlong/finllm-mi/data/scale120")
    OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/shared_sae_results.json"

    print("=" * 60)
    print("SHARED SAE EXPERIMENT")
    print("=" * 60)
    print(f"Device: {device}, Layer: {layer}, K: {K}, Stocks: {N_STOCKS}")
    print()

    # ─── Load Kronos ───
    print("Loading Kronos...")
    tokenizer = KronosTokenizer.from_pretrained(
        "/data/houwanlong/models/Kronos-Tokenizer-base"
    ).to(device).eval()

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
    sd = load_file("/data/houwanlong/models/Kronos-base/model.safetensors")
    model.load_state_dict(sd, strict=False)
    model = model.to(device).half().eval()
    d_model = cfg["d_model"]
    print(f"  d_model={d_model}, d_hidden={d_model * EXPANSION}")

    # ─── Load sector mapping, select 20 stocks ───
    with open("/tmp/sectors120.json") as f:
        sector_map = json.load(f)

    selected_stocks = []
    for sector in ["Bank", "Energy", "Tech", "Consumer"]:
        tickers = sector_map[sector][:5]
        for t in tickers:
            selected_stocks.append((t, sector))

    print(f"\nSelected {len(selected_stocks)} stocks: 5 per sector")
    for t, s in selected_stocks:
        print(f"  {t} ({s})")

    # ─── Phase 1: Extract activations from all stocks ───
    print(f"\n{'=' * 60}")
    print("Phase 1: Extracting activations from all stocks")
    print(f"{'=' * 60}")

    all_acts = []  # will concat across stocks
    stock_meta = {}  # stock_ticker -> {start_idx, end_idx, sector, acts, labels, windows}
    lookback, stride = 64, 32

    for ticker, sector in selected_stocks:
        csv_path = DATA_DIR / f"{ticker}.csv"
        print(f"\n  {ticker} ({sector})...", end=" ", flush=True)

        if not csv_path.exists():
            print("SKIP (file not found)")
            continue

        df = pd.read_csv(str(csv_path))
        for col in ["open", "close", "high", "low", "volume", "amount"]:
            if col not in df.columns:
                df[col] = 0.0
        data = df[["open", "close", "high", "low", "volume", "amount"]].values.astype(np.float32)
        data = data[~np.isnan(data).any(axis=1)]

        if len(data) < 200:
            print("SKIP (too few rows)")
            continue

        mn, st = data.mean(0), data.std(0)
        data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)

        n_windows = min(2000, (len(data_norm) - lookback) // stride)
        windows = np.stack([data_norm[i:i+lookback] for i in range(0, n_windows * stride, stride)])

        acts = extract_activations(windows, tokenizer, model, layer, device)

        # Compute financial labels
        labels_list = []
        for i in range(0, n_windows * stride, stride):
            win = data_norm[i:i+lookback]
            labels_list.append(compute_financial_stats(win))

        n = min(len(acts), len(labels_list))
        acts = acts[:n]
        labels_list = labels_list[:n]

        start_idx = len(all_acts)
        all_acts.append(acts)
        stock_meta[ticker] = {
            "sector": sector,
            "n_windows": n,
            "start_idx": start_idx,
            "acts": acts,
            "labels": labels_list,
            "windows": windows[:n],
        }
        print(f"{n} windows, acts shape={acts.shape}")

    if len(all_acts) == 0:
        print("ERROR: No valid stocks processed")
        return

    all_acts = np.concatenate(all_acts, axis=0)

    # Update start_idx after concat
    offset = 0
    for ticker in stock_meta:
        stock_meta[ticker]["start_idx"] = offset
        offset += len(stock_meta[ticker]["acts"])

    print(f"\n  Total activations: {all_acts.shape} ({all_acts.shape[0]} samples x {all_acts.shape[1]} dims)")
    print(f"  Stocks successfully processed: {len(stock_meta)}")

    # ─── Phase 2: Train SHARED SAE ───
    print(f"\n{'=' * 60}")
    print("Phase 2: Training SHARED SAE on all stocks")
    print(f"{'=' * 60}")

    d_hidden = d_model * EXPANSION
    shared_sae, train_info = train_sae(all_acts, d_model, d_hidden, K, device, steps=STEPS, batch_size=BATCH)
    print(f"  Variance explained: {train_info['var_exp']:.4f}")
    print(f"  Per-sample L0: {train_info['l0']:.1f}")
    print(f"  Alive features: {train_info['alive']:.1%}")

    # ─── Phase 3: Per-stock feature labeling ───
    print(f"\n{'=' * 60}")
    print("Phase 3: Per-stock feature labeling")
    print(f"{'=' * 60}")

    per_stock_results = {}
    acts_t = torch.from_numpy(all_acts).float().to(device)

    for ticker in stock_meta:
        meta = stock_meta[ticker]
        sect = meta["sector"]
        start = meta["start_idx"]
        end = start + len(meta["acts"])

        # Get latents for this stock
        stock_acts_t = acts_t[start:end]
        with torch.no_grad():
            _, lat = shared_sae(stock_acts_t)
        stock_lat = lat.cpu().numpy()

        # Label features
        type_dist, n_strong, n_alive, feature_labels = label_features(stock_lat, meta["labels"])

        # Null-calibrated threshold
        null_thresh = null_calibrated_threshold(stock_lat, meta["labels"])

        # Count features above null threshold
        n_above_null = sum(1 for fl in feature_labels if abs(fl[2]) > null_thresh)

        # Top feature frequencies
        _, all_l = shared_sae(stock_acts_t)
        freq = (all_l != 0).float().sum(0)
        top50_idx = freq.argsort(descending=True)[:50].tolist()

        # Ablation: ablate top 50 features
        test_start = max(0, len(meta["windows"]) - min(30, len(meta["windows"])))
        test_windows = meta["windows"][test_start:]
        if len(test_windows) >= 5:
            ab_effect = compute_ablation_effect(model, tokenizer, shared_sae, layer, test_windows, top50_idx, device)
            # Random baseline: ablate 50 random features
            all_ids = list(range(d_hidden))
            rand_effects = []
            for _ in range(10):
                rids = np.random.choice(all_ids, 50, replace=False).tolist()
                re = compute_ablation_effect(model, tokenizer, shared_sae, layer, test_windows, rids, device)
                rand_effects.append(re)
            rand_mean = np.mean(rand_effects)
            rand_std = np.std(rand_effects)
            z_score = (ab_effect - rand_mean) / (rand_std + 1e-10)
            p_val = 2 * stats.norm.sf(abs(z_score))
        else:
            ab_effect = None
            rand_mean, rand_std, z_score, p_val = None, None, None, None

        per_stock_results[ticker] = {
            "sector": sect,
            "n_windows": meta["n_windows"],
            "type_distribution": {k: int(v) for k, v in sorted(type_dist.items(), key=lambda x: -x[1])},
            "n_strong_features": n_strong,
            "n_alive_features": n_alive,
            "n_above_null_threshold": n_above_null,
            "null_calibrated_threshold": float(null_thresh),
            "ablation_effect": float(ab_effect) if ab_effect is not None else None,
            "random_ablation_mean": float(rand_mean) if rand_mean is not None else None,
            "random_ablation_std": float(rand_std) if rand_std is not None else None,
            "z_vs_random": float(z_score) if z_score is not None else None,
            "p_vs_random": float(p_val) if p_val is not None else None,
        }

        ie_str = f"abl_effect={ab_effect:.4f}" if ab_effect is not None else "abl_effect=NA"
        print(f"  {ticker} ({sect}): alive={n_alive}, strong={n_strong}, "
              f"above_null={n_above_null}, null_thr={null_thresh:.3f}, {ie_str}")

    # ─── Phase 4: Compare with per-stock SAE baseline ───
    print(f"\n{'=' * 60}")
    print("Phase 4: Comparison with per-stock SAE")
    print(f"{'=' * 60}")

    # Load existing per-stock results for comparison
    per_stock_ref_path = "/data/houwanlong/finllm-mi/outputs/sae/scale120_results.json"
    per_stock_ref = None
    if os.path.exists(per_stock_ref_path):
        with open(per_stock_ref_path) as f:
            per_stock_ref = json.load(f)

    # Aggregate concept distribution across all stocks (shared SAE)
    shared_total_dist = defaultdict(int)
    shared_total_strong = 0
    shared_total_alive = 0
    shared_abl_effects = []

    for ticker, r in per_stock_results.items():
        for concept, count in r["type_distribution"].items():
            shared_total_dist[concept] += count
        shared_total_strong += r["n_strong_features"]
        shared_total_alive += r["n_alive_features"]
        if r["ablation_effect"] is not None:
            shared_abl_effects.append(r["ablation_effect"])

    shared_total_concepts = sum(shared_total_dist.values())
    shared_dist_sorted = sorted(shared_total_dist.items(), key=lambda x: -x[1])

    print("\n  Shared SAE — Aggregate concept distribution:")
    for concept, count in shared_dist_sorted[:15]:
        pct = count / shared_total_concepts * 100
        print(f"    {concept}: {count} ({pct:.1f}%)")

    print(f"\n  Total labeled features: {shared_total_concepts}")
    print(f"  Total strongly correlated (|r|>0.3): {shared_total_strong}")
    print(f"  Total alive features: {shared_total_alive}")
    print(f"  Effective labeling rate: {shared_total_strong/shared_total_alive*100:.1f}%" if shared_total_alive > 0 else "")

    # Concept consistency: measure std of concept percentages across stocks
    print(f"\n  Concept consistency (cross-stock std of top concepts):")
    top_concepts = [c for c, _ in shared_dist_sorted[:8]]
    for concept in top_concepts:
        pcts = []
        for r in per_stock_results.values():
            total = sum(r["type_distribution"].values())
            pcts.append(r["type_distribution"].get(concept, 0) / total * 100 if total > 0 else 0)
        print(f"    {concept}: mean={np.mean(pcts):.1f}%, std={np.std(pcts):.1f}%")

    # Ablation comparison
    if shared_abl_effects:
        print(f"\n  Shared SAE ablation effects: mean={np.mean(shared_abl_effects):.4f}, std={np.std(shared_abl_effects):.4f}")
        if per_stock_ref and "per_stock" in per_stock_ref:
            per_stock_abl = [s.get("intervention_effect") for s in per_stock_ref["per_stock"]
                             if s.get("intervention_effect") is not None]
            if per_stock_abl:
                print(f"  Per-stock SAE ablation effects: mean={np.mean(per_stock_abl):.4f}, std={np.std(per_stock_abl):.4f}")
                # Paired comparison for overlapping stocks
                shared_abl_dict = {t: r["ablation_effect"] for t, r in per_stock_results.items() if r["ablation_effect"] is not None}
                per_stock_abl_dict = {s["ticker"]: s["intervention_effect"] for s in per_stock_ref["per_stock"] if s.get("intervention_effect") is not None}
                overlap = set(shared_abl_dict.keys()) & set(per_stock_abl_dict.keys())
                if len(overlap) >= 5:
                    shared_vals = [shared_abl_dict[t] for t in overlap]
                    per_stock_vals = [per_stock_abl_dict[t] for t in overlap]
                    t_stat, p_val = stats.ttest_rel(shared_vals, per_stock_vals)
                    print(f"  Paired comparison (n={len(overlap)} overlapping stocks):")
                    print(f"    Shared mean: {np.mean(shared_vals):.4f}")
                    print(f"    Per-stock mean: {np.mean(per_stock_vals):.4f}")
                    print(f"    Difference: {np.mean(shared_vals) - np.mean(per_stock_vals):+.4f}")
                    print(f"    Paired t-test: t={t_stat:.3f}, p={p_val:.4f} {'SIG' if p_val < 0.05 else 'ns'}")

    # ─── Phase 5: Save results ───
    print(f"\n{'=' * 60}")
    print("Phase 5: Saving results")
    print(f"{'=' * 60}")

    # Compute concept distribution entropy (higher = more distributed)
    dist_pcts = np.array([c / shared_total_concepts for _, c in shared_dist_sorted])
    entropy = -np.sum(dist_pcts * np.log(dist_pcts + 1e-10))

    final_results = {
        "experiment": "shared_sae",
        "n_stocks": len(per_stock_results),
        "layer": layer,
        "k": K,
        "expansion": EXPANSION,
        "d_model": d_model,
        "d_hidden": d_hidden,
        "training_samples": int(all_acts.shape[0]),
        "steps": STEPS,
        "batch_size": BATCH,
        "training_info": {
            "variance_explained": train_info["var_exp"],
            "per_sample_l0": train_info["l0"],
            "alive_features_pct": train_info["alive"],
            "alive_features_count": int(train_info["alive"] * d_hidden),
        },
        "aggregate_concept_distribution": [
            {"concept": c, "count": int(cnt), "pct": float(cnt / shared_total_concepts * 100)}
            for c, cnt in shared_dist_sorted
        ],
        "total_labeled_features": shared_total_concepts,
        "total_strongly_correlated": shared_total_strong,
        "concept_distribution_entropy": float(entropy),
        "per_stock": {t: per_stock_results[t] for t in sorted(per_stock_results.keys())},
        "comparison_with_per_stock_sae": {
            "note": "See scale120_results.json for per-stock SAE baseline",
            "overlapping_stocks": len(set(per_stock_results.keys()) & set(
                s["ticker"] for s in per_stock_ref["per_stock"])) if per_stock_ref else 0
        }
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    # Also save the SAE model
    sae_path = "/data/houwanlong/finllm-mi/outputs/sae/shared_sae_layer6.pt"
    torch.save(shared_sae.state_dict(), sae_path)
    print(f"  SAE model saved to {sae_path}")

    print(f"  Results saved to {OUTPUT}")
    print(f"\n  Total time: {time.time() - t_total:.0f}s")
    print(f"\n{'=' * 60}")
    print("SHARED SAE EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
