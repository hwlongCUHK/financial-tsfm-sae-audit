"""30+ financial features + steering validation on SAE features."""
import torch, numpy as np, json, sys, time, argparse
from pathlib import Path
import pandas as pd
from scipy import stats
from collections import defaultdict

sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

# ─── 30+ financial features ───
def compute_financial_features(close, high, low, volume, window=64):
    """Compute a comprehensive set of financial statistics."""
    returns = np.diff(close) / (close[:-1] + 1e-5)
    log_returns = np.diff(np.log(close + 1e-5))
    features = {}

    # Volatility family
    features["volatility"] = np.std(returns)
    features["parkinson_vol"] = np.sqrt(np.mean(np.log(high[1:]/low[1:])**2) / (4*np.log(2)))
    gk = 0.5*np.mean(np.log(high[1:]/(low[1:]+1e-5))**2) - (2*np.log(2)-1)*np.mean(np.log(close[1:]/(close[:-1]+1e-5))**2)
    features["garman_klass_vol"] = np.sqrt(max(0, gk))
    features["realized_vol"] = np.sqrt(np.sum(returns**2))
    features["vol_of_vol"] = np.std(np.abs(returns))

    # Trend / momentum family
    features["trend"] = np.polyfit(np.arange(window), close, 1)[0]
    features["momentum_5"] = close[-1] / (close[-6] + 1e-5) - 1
    features["momentum_10"] = close[-1] / (close[-11] + 1e-5) - 1
    features["momentum_20"] = close[-1] / (close[-21] + 1e-5) - 1
    features["ma_cross"] = np.mean(close[-5:]) / (np.mean(close[-20:]) + 1e-5) - 1
    features["rsi_like"] = np.mean(returns[-14:] > 0) / (np.mean(returns[-14:] < 0) + 1e-5)  # simplified RSI

    # Distribution shape family
    features["skewness"] = stats.skew(returns)
    features["kurtosis"] = stats.kurtosis(returns, fisher=True)
    features["jarque_bera"] = (len(returns)/6) * (stats.skew(returns)**2 + stats.kurtosis(returns, fisher=True)**2/4)

    # Tail risk family
    features["max_drawdown"] = np.min(close / np.maximum.accumulate(close) - 1)
    features["var_95"] = np.percentile(returns, 5)
    features["cvar_95"] = np.mean(returns[returns <= np.percentile(returns, 5)])
    features["max_1day_loss"] = np.min(returns)
    features["max_1day_gain"] = np.max(returns)

    # Range / dispersion family
    features["price_range"] = (close.max() - close.min()) / close.mean()
    features["high_low_range"] = np.mean(high[1:] - low[1:]) / close.mean()
    features["close_to_close_range"] = (close[-1] - close[0]) / close.mean()

    # Volatility dynamics
    features["vol_clustering"] = np.mean(returns**2) / (np.var(returns) + 1e-10)
    features["vol_persistence"] = np.corrcoef(np.abs(returns[1:]), np.abs(returns[:-1]))[0,1] if len(returns) > 2 else 0
    features["leverage_effect"] = np.corrcoef(returns, np.abs(returns))[0,1] if len(returns) > 2 else 0

    # Volume features
    vol_chg = np.diff(volume) / (volume[:-1] + 1e-5)
    features["volume_trend"] = np.mean(vol_chg) if len(vol_chg) > 0 else 0
    features["volume_volatility"] = np.std(vol_chg) if len(vol_chg) > 0 else 0
    features["volume_price_corr"] = np.corrcoef(returns, vol_chg[:len(returns)])[0,1] if len(returns) > 2 else 0

    # Autocorrelation
    features["autocorr_lag1"] = np.corrcoef(returns[1:], returns[:-1])[0,1] if len(returns) > 2 else 0
    features["autocorr_lag5"] = np.corrcoef(returns[5:], returns[:-5])[0,1] if len(returns) > 6 else 0

    # Hurst-like (simple rescaled range)
    cum_dev = np.cumsum(returns - returns.mean())
    features["hurst_like"] = np.log((cum_dev.max() - cum_dev.min()) / (np.std(returns) + 1e-10)) / np.log(len(returns))

    # Handle NaN/inf
    for k in list(features.keys()):
        if not np.isfinite(features[k]):
            features[k] = 0.0

    return features


class TopKSAE(torch.nn.Module):
    def __init__(self, d_model, d_hidden, k=64):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = torch.nn.Linear(d_hidden, d_model, bias=False)
        self.b_pre = torch.nn.Parameter(torch.zeros(d_model))
        self.k = k
    def forward(self, x):
        xc = x - self.b_pre; lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        return self.decoder(lat * mask) + self.b_pre, lat * mask
    def steer(self, x, steer_ids, multiplier=5.0):
        """Steer specific features by multiplying their activation."""
        xc = x - self.b_pre; lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        # Steer: amplify specified features
        lat[:, steer_ids] = lat[:, steer_ids] * multiplier
        return self.decoder(lat * mask) + self.b_pre
    def ablate_reconstruct(self, x, ids):
        xc = x - self.b_pre; lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        mask[:, ids] = 0
        return self.decoder(lat * mask) + self.b_pre


def process_one(csv_path, tokenizer, model, layer, device, args):
    t0 = time.time()
    df = pd.read_csv(str(csv_path))
    for col in ["open","close","high","low","volume","amount"]:
        if col not in df.columns: df[col] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 200: return None
    mn, st_d = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st_d + 1e-5), -5, 5)

    lb, stride = 64, 32
    nw = min(2000, (len(data_norm) - lb) // stride)
    windows = np.stack([data_norm[i:i+lb] for i in range(0, nw*stride, stride)])
    raw_windows = [data[i:i+lb] for i in range(0, nw*stride, stride)]
    n_train = int(len(windows) * 0.8)

    # Activations
    acts_list = []
    def hook_fn(m, i, o):
        a = o[0] if isinstance(o, tuple) else o
        acts_list.append(a[:, -1, :].detach().cpu().float().numpy())
    hook = model.transformer[layer].register_forward_hook(hook_fn)
    bs = 64
    with torch.no_grad():
        for b in range(0, n_train, bs):
            batch = torch.from_numpy(windows[b:b+bs]).float().to(device)
            s1, s2 = tokenizer.encode(batch, half=True)
            model(s1, s2)
    hook.remove()
    acts = np.concatenate(acts_list, axis=0)

    # Train SAE
    d_model, d_hidden = acts.shape[1], acts.shape[1] * 4
    sae = TopKSAE(d_model, d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
    at = torch.from_numpy(acts).float().to(device)
    for step in range(args.steps):
        idx = torch.randint(0, len(at), (args.batch,))
        xr, _ = sae(at[idx])
        loss = torch.nn.functional.mse_loss(xr, at[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

    # Compute 30+ financial features per window
    feature_names = []
    all_labels = []
    for rw in raw_windows:
        feats = compute_financial_features(rw[:, 1], rw[:, 2], rw[:, 3], rw[:, 4])
        if not feature_names:
            feature_names = sorted(feats.keys())
        all_labels.append([feats[k] for k in feature_names])

    label_arr = np.array(all_labels)  # (n_windows, n_features)

    # Get SAE latents
    n = min(len(acts), len(label_arr))
    full_lat = []
    with torch.no_grad():
        for i in range(0, n, 256):
            _, lat2 = sae(at[i:i+256])
            full_lat.append(lat2.cpu().numpy())
    full_lat = np.concatenate(full_lat)[:n]
    label_arr = label_arr[:n]

    # Assign each alive feature to best-correlated financial stat
    alive_mask = (full_lat != 0).sum(0) > 10
    concept_features = defaultdict(list)
    feature_best_corr = {}

    for j in np.where(alive_mask)[0]:
        act_j = full_lat[:, j]; a = act_j != 0
        if a.sum() < 5: continue
        corrs = [np.corrcoef(act_j[a], label_arr[a, k])[0,1] for k in range(len(feature_names))]
        corrs = [0 if np.isnan(c) else abs(c) for c in corrs]
        best_k = np.argmax(corrs)
        if corrs[best_k] > 0.15:
            concept_features[feature_names[best_k]].append(j)
            feature_best_corr[j] = (feature_names[best_k], corrs[best_k])

    # ─── STEERING VALIDATION ───
    test_start = n_train
    test_wins = windows[test_start:test_start+min(50, len(windows)-test_start)]
    if len(test_wins) < 10:
        del sae; torch.cuda.empty_cache()
        return None

    test_t = torch.from_numpy(test_wins).float().to(device)
    raw_test = raw_windows[test_start:test_start+len(test_wins)]

    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
        base_out = model(s1_ids, s2_ids)
    base_s1 = base_out[0].float()

    # Steer each concept's features and measure output change
    steer_results = {}
    concepts_tested = [c for c, feats in concept_features.items() if len(feats) >= 5]
    for concept in concepts_tested:
        feat_ids = concept_features[concept][:10]  # top 10 features per concept

        def make_intervention(steer_ids, mode="ablate"):
            def intervene(m, i, o):
                orig = o[0] if isinstance(o, tuple) else o
                B, T, D = orig.shape
                flat = orig.reshape(-1, D).float()
                if mode == "ablate":
                    modified = sae.ablate_reconstruct(flat, steer_ids)
                else:
                    modified = sae.steer(flat, steer_ids, multiplier=5.0)
                return (modified.reshape(B, T, D).half(),) + o[1:] if isinstance(o, tuple) else modified.reshape(B, T, D).half()
            return intervene

        # Ablation
        hk = model.transformer[layer].register_forward_hook(make_intervention(feat_ids, "ablate"))
        with torch.no_grad():
            ab_s1, ab_s2 = tokenizer.encode(test_t, half=True)
            ab_out = model(ab_s1, ab_s2)
        hk.remove()
        ab_s1_out = ab_out[0].float()

        # Amplification (steering)
        hk2 = model.transformer[layer].register_forward_hook(make_intervention(feat_ids, "steer"))
        with torch.no_grad():
            st_s1, st_s2 = tokenizer.encode(test_t, half=True)
            st_out = model(st_s1, st_s2)
        hk2.remove()
        st_s1_out = st_out[0].float()

        # Metrics
        cos_ab = torch.nn.functional.cosine_similarity(
            base_s1.reshape(-1, base_s1.shape[-1]), ab_s1_out.reshape(-1, ab_s1_out.shape[-1]), dim=-1).mean()
        cos_st = torch.nn.functional.cosine_similarity(
            base_s1.reshape(-1, base_s1.shape[-1]), st_s1_out.reshape(-1, st_s1_out.shape[-1]), dim=-1).mean()

        # Random baseline
        rand_cos_ab = []; rand_cos_st = []
        for _ in range(10):
            rids = np.random.choice(d_hidden, len(feat_ids), replace=False).tolist()
            hkr = model.transformer[layer].register_forward_hook(make_intervention(rids, "ablate"))
            with torch.no_grad():
                ra_s1, ra_s2 = tokenizer.encode(test_t, half=True)
                ra_out = model(ra_s1, ra_s2)
            hkr.remove()
            rand_cos_ab.append(torch.nn.functional.cosine_similarity(
                base_s1.reshape(-1, base_s1.shape[-1]), ra_out[0].float().reshape(-1, base_s1.shape[-1]), dim=-1).mean().item())

        rand_mean = np.mean(rand_cos_ab); rand_std = np.std(rand_cos_ab)
        z_ab = (cos_ab.item() - rand_mean) / (rand_std + 1e-10)
        p_ab = 2 * stats.norm.sf(abs(z_ab))

        steer_results[concept] = {
            "n_features": len(feat_ids),
            "cos_ablation": float(cos_ab.item()),
            "cos_steer": float(cos_st.item()),
            "rand_cos_mean": float(rand_mean),
            "z_vs_random": float(z_ab),
            "p_vs_random": float(p_ab),
            "significant": bool(p_ab < 0.05),
        }

    # Aggregate concept distribution
    type_dist = defaultdict(int)
    for j, (name, corr) in feature_best_corr.items():
        type_dist[name] += 1

    total = sum(type_dist.values())
    top_concepts = sorted(type_dist.items(), key=lambda x: -x[1])[:10]

    del sae; torch.cuda.empty_cache()
    return {
        "top_concepts": [(name, count, float(count/total*100)) for name, count in top_concepts],
        "n_features_total": len(feature_names),
        "steering": steer_results,
        "time": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/houwanlong/finllm-mi/data/scale120")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/rich_steering.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-stocks", type=int, default=30)
    args = parser.parse_args()

    device = args.device; t_total = time.time()
    print("Loading Kronos...")
    tokenizer = KronosTokenizer.from_pretrained("/data/houwanlong/models/Kronos-Tokenizer-base").to(device).eval()
    with open("/data/houwanlong/models/Kronos-base/config.json") as f: cfg = json.load(f)
    model = Kronos(s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"], n_layers=cfg["n_layers"],
                   d_model=cfg["d_model"], n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
                   ffn_dropout_p=cfg["ffn_dropout_p"], attn_dropout_p=cfg["attn_dropout_p"],
                   resid_dropout_p=cfg["resid_dropout_p"], token_dropout_p=cfg["token_dropout_p"],
                   learn_te=cfg["learn_te"])
    sd = load_file("/data/houwanlong/models/Kronos-base/model.safetensors")
    model.load_state_dict(sd, strict=False); model = model.to(device).half().eval()

    with open("/tmp/sectors120.json") as f:
        sector_map = json.load(f)
    ticker_sector = {}
    for sname, tickers in sector_map.items():
        for t in tickers: ticker_sector[t] = sname

    csv_files = sorted(Path(args.data_dir).glob("*.csv"))[:args.max_stocks]
    print(f"Processing {len(csv_files)} stocks with 30+ features + steering...")

    results = []
    for i, f in enumerate(csv_files):
        ticker = f.stem; sector = ticker_sector.get(ticker, "Other")
        print(f"[{i+1}/{len(csv_files)}] {ticker} ({sector})...", end=" ", flush=True)
        r = process_one(f, tokenizer, model, args.layer, device, args)
        if r:
            r["ticker"] = ticker; r["sector"] = sector
            results.append(r)
            sig_count = sum(1 for v in r["steering"].values() if v["significant"])
            print(f"concepts={len(r['steering'])} sig_steer={sig_count} top={r['top_concepts'][:3]}")
        else:
            print("SKIP")

    # ─── Aggregate concept distribution ───
    print(f"\n{'='*70}")
    print(f"AGGREGATE CONCEPT DISTRIBUTION ({len(results)} stocks, 30+ features)")
    print(f"{'='*70}")

    merged = defaultdict(float)
    all_concept_names = set()
    for r in results:
        for name, count, pct in r["top_concepts"]:
            merged[name] += count
            all_concept_names.add(name)

    total_all = sum(merged.values())
    print(f"\nTop 15 concepts (out of {len(all_concept_names)} total):")
    for name, count in sorted(merged.items(), key=lambda x: -x[1])[:15]:
        print(f"  {name:<25}: {count/total_all*100:5.1f}% ({int(count)} features)")

    # ─── Steering aggregation ───
    print(f"\n{'='*70}")
    print("STEERING VALIDATION (ablation + amplification)")
    print(f"{'='*70}")

    steer_agg = defaultdict(list)
    for r in results:
        for concept, data in r["steering"].items():
            steer_agg[concept].append(data)

    print(f"\n{'Concept':<25} {'n_stocks':<10} {'Ablate cos':<12} {'Steer cos':<12} {'vs Random':<12} {'% Sig':<10}")
    print("-" * 80)

    steer_summary = {}
    for concept in sorted(steer_agg.keys(), key=lambda x: -len(steer_agg[x])):
        items = steer_agg[concept]
        if len(items) < 3: continue
        ab_cos = [it["cos_ablation"] for it in items]
        st_cos = [it["cos_steer"] for it in items]
        sig_pct = np.mean([it["significant"] for it in items])
        p_vals = [it["p_vs_random"] for it in items]

        # Bonferroni across concepts
        steer_summary[concept] = {
            "n_stocks": len(items),
            "mean_ablation_cos": float(np.mean(ab_cos)),
            "mean_steer_cos": float(np.mean(st_cos)),
            "significant_pct": float(sig_pct),
            "mean_p_value": float(np.mean(p_vals)),
        }
        print(f"{concept:<25} {len(items):<10} {np.mean(ab_cos):.4f}       {np.mean(st_cos):.4f}       {'SIG' if sig_pct > 0.5 else 'ns':<12} {sig_pct:.0%}")

    # Bonferroni
    all_p = [np.mean([it["p_vs_random"] for it in steer_agg[c]]) for c in sorted(steer_agg.keys()) if len(steer_agg[c]) >= 3]
    from statsmodels.stats.multitest import multipletests
    if all_p:
        rej, p_corr, _, _ = multipletests(all_p, method='bonferroni')
        print(f"\nBonferroni correction ({len(all_p)} concepts):")
        sig_concepts = []
        for i, concept in enumerate(sorted([c for c in steer_agg.keys() if len(steer_agg[c]) >= 3])):
            if rej[i]:
                sig_concepts.append(concept)
                print(f"  {concept}: SIG (corrected p={p_corr[i]:.4f})")
        if not sig_concepts:
            print(f"  No concept survives Bonferroni")
    else:
        sig_concepts = []

    final = {
        "n_stocks": len(results),
        "n_features_used": len(set().union(*[set(dict(r['top_concepts']).keys()) for r in results])),
        "concept_distribution": {name: int(count) for name, count in sorted(merged.items(), key=lambda x: -x[1])[:30]},
        "steering_summary": steer_summary,
        "bonferroni_significant": sig_concepts,
    }
    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved. {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
