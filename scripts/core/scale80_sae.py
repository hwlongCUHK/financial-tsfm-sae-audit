"""80-stock SAE with fixed causal interpretation + bootstrap CIs."""
import torch, numpy as np, json, sys, time, argparse, os
from pathlib import Path
import pandas as pd
from scipy import stats
from collections import defaultdict

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
        xc = x - self.b_pre; lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        return self.decoder(lat * mask) + self.b_pre, lat * mask
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
    mn, st = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)

    lb, stride = 64, 32
    nw = min(2000, (len(data_norm) - lb) // stride)
    windows = np.stack([data_norm[i:i+lb] for i in range(0, nw*stride, stride)])
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

    # SAE train
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

    with torch.no_grad():
        xt = at[:min(500, len(at))]
        recon, lat = sae(xt)
        ve = 1 - torch.nn.functional.mse_loss(recon, xt).item() / (xt.var().item() + 1e-10)
        per_sample_l0 = (lat != 0).float().sum(-1).mean().item()
        alive = (lat.abs().sum(0) > 1e-6).float().mean().item()

    # Interpret
    label_keys = ["vol","trend","max_dd","range","vol_cluster","skew","kurt"]
    label_names = ["Volatility","Trend","Max Drawdown","Price Range","Vol Clustering","Skewness","Kurtosis"]
    labels = []
    for i in range(0, len(data_norm) - lb * 2, stride):
        c = data_norm[i:i+lb, 1]; r = np.diff(c) / (c[:-1] + 1e-5)
        labels.append({"vol": np.std(r), "trend": np.polyfit(np.arange(lb), c, 1)[0],
            "max_dd": float(np.min(c / np.maximum.accumulate(c) - 1)),
            "range": float((c.max() - c.min()) / c.mean()),
            "vol_cluster": float(np.mean(r**2) / (r.std()**2 + 1e-5)),
            "skew": float(np.mean((r - r.mean())**3) / (r.std()**3 + 1e-5)),
            "kurt": float(np.mean((r - r.mean())**4) / (r.std()**4 + 1e-5))})

    n = min(len(acts), len(labels))
    full_lat = []
    with torch.no_grad():
        for i in range(0, n, 256):
            _, lat2 = sae(at[i:i+256])
            full_lat.append(lat2.cpu().numpy())
    full_lat = np.concatenate(full_lat)[:n]
    label_arr = np.array([[l[k] for k in label_keys] for l in labels[:n]])

    type_dist = {}
    n_strong = 0
    alive_mask = (full_lat != 0).sum(0) > 10
    for j in np.where(alive_mask)[0]:
        act = full_lat[:, j]; a = act != 0
        if a.sum() < 5: continue
        corrs = [np.corrcoef(act[a], label_arr[a, k])[0,1] for k in range(len(label_keys))]
        corrs = [0 if np.isnan(c) else c for c in corrs]
        best = np.argmax(np.abs(corrs))
        type_dist[label_names[best]] = type_dist.get(label_names[best], 0) + 1
        if abs(corrs[best]) > 0.3: n_strong += 1

    # FIXED: Causal — higher intervention effect = LOWER cos = model output changes MORE
    test_wins_t = torch.from_numpy(windows[n_train:n_train+min(30, len(windows)-n_train)]).float().to(device)
    if len(test_wins_t) < 5:
        del sae; torch.cuda.empty_cache()
        return {"type_dist": type_dist, "n_strong": n_strong, "n_alive": int(alive_mask.sum()),
                "ve": float(ve), "l0": float(per_sample_l0), "alive": float(alive),
                "intervention_effect": None}

    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_wins_t, half=True)
        base = model(s1, s2)
    base_s1 = base[0].float()

    with torch.no_grad():
        _, all_l = sae(at[:min(1000, len(at))])
    freq = (all_l != 0).float().sum(0)
    top50 = freq.argsort(descending=True)[:50].tolist()

    def make_int(ab_ids):
        def intervene(m, i, o):
            orig = o[0] if isinstance(o, tuple) else o
            B, T, D = orig.shape
            ablated = sae.ablate_reconstruct(orig.reshape(-1, D).float(), ab_ids).reshape(B, T, D).half()
            return (ablated,) + o[1:] if isinstance(o, tuple) else ablated
        return intervene

    hk = model.transformer[layer].register_forward_hook(make_int(top50))
    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_wins_t, half=True)
        ab = model(s1, s2)
    hk.remove()

    # intervention_effect = 1 - cos_sim (higher = more impact)
    cs = torch.nn.functional.cosine_similarity(
        base_s1.reshape(-1, base_s1.shape[-1]),
        ab[0].float().reshape(-1, base_s1.shape[-1]), dim=-1).mean()
    intervention_effect = 1.0 - cs.item()  # FIXED: higher = more model output change

    # Random baseline
    rand_effects = []
    for _ in range(10):
        rids = np.random.choice(d_hidden, 50, replace=False).tolist()
        hk = model.transformer[layer].register_forward_hook(make_int(rids))
        with torch.no_grad():
            s1, s2 = tokenizer.encode(test_wins_t, half=True)
            ro = model(s1, s2)
        hk.remove()
        rcs = torch.nn.functional.cosine_similarity(
            base_s1.reshape(-1, base_s1.shape[-1]),
            ro[0].float().reshape(-1, base_s1.shape[-1]), dim=-1).mean()
        rand_effects.append(1.0 - rcs.item())

    rand_mean = np.mean(rand_effects)
    rand_std = np.std(rand_effects)
    z_score = (intervention_effect - rand_mean) / (rand_std + 1e-10)
    p_val = 2 * stats.norm.sf(abs(z_score))

    del sae; torch.cuda.empty_cache()
    return {"type_dist": type_dist, "n_strong": n_strong, "n_alive": int(alive_mask.sum()),
            "ve": float(ve), "l0": float(per_sample_l0), "alive": float(alive),
            "intervention_effect": float(intervention_effect),
            "random_effect_mean": float(rand_mean),
            "random_effect_std": float(rand_std),
            "z_vs_random": float(z_score), "p_vs_random": float(p_val),
            "time": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/houwanlong/finllm-mi/data/scale80")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/scale80_results.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
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
    model.load_state_dict(sd, strict=False)
    model = model.to(device).half().eval()

    # Load sector mapping
    with open("/tmp/sectors_filtered.json") as f:
        sector_map = json.load(f)
    # Build reverse map
    ticker_sector = {}
    for sector, tickers in sector_map.items():
        for t in tickers:
            ticker_sector[t] = sector

    # Process all CSVs
    csv_files = sorted(Path(args.data_dir).glob("sh*.csv"))
    print(f"Processing {len(csv_files)} stocks...")

    results = []
    for i, f in enumerate(csv_files):
        ticker = f.stem
        sector = ticker_sector.get(ticker, "Other")
        print(f"[{i+1}/{len(csv_files)}] {ticker} ({sector})...", end=" ", flush=True)
        r = process_one(f, tokenizer, model, args.layer, device, args)
        if r:
            r["ticker"] = ticker; r["sector"] = sector
            results.append(r)
            ie = r.get("intervention_effect")
            print(f"VE={r['ve']:.3f} alive={r['alive']:.1%} interv={'NA' if ie is None else f'{ie:.4f}'} p={r.get('p_vs_random',1):.4f}")
        else:
            print("SKIP")

    # ─── SECTOR AGGREGATION ───
    sectors = defaultdict(list)
    for r in results:
        sectors[r["sector"]].append(r)

    print(f"\n{'='*70}")
    print(f"SECTOR ANALYSIS — {len(results)} stocks")
    print(f"{'='*70}")

    # Bootstrap function
    def bootstrap_mean(vals, n=10000):
        means = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(n)]
        return np.mean(vals), np.percentile(means, 2.5), np.percentile(means, 97.5)

    sector_stats = {}
    for sector, stocks in sectors.items():
        print(f"\n## {sector} ({len(stocks)} stocks)")

        # Merge type distributions
        merged = {}
        for s in stocks:
            for t, c in s["type_dist"].items():
                merged[t] = merged.get(t, 0) + c
        total = sum(merged.values())
        top3 = sorted(merged.items(), key=lambda x: -x[1])[:3]

        # Intervention effects (fixed: higher = more impact)
        ies = [s["intervention_effect"] for s in stocks if s.get("intervention_effect") is not None]
        pvs = [s.get("p_vs_random", 1) for s in stocks if s.get("p_vs_random") is not None]
        n_sig = sum(1 for p in pvs if p < 0.05)

        mean_ie, ci_lo, ci_hi = bootstrap_mean(ies) if ies else (0, 0, 0)

        print(f"  Top concepts: {', '.join(f'{t} ({c/total*100:.1f}%)' for t, c in top3)}")
        print(f"  Intervention effect: {mean_ie:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  Significant in {n_sig}/{len(ies)} stocks")

        sector_stats[sector] = {
            "n_stocks": len(stocks),
            "top_concepts": [(t, int(c), float(c/total*100)) for t, c in top3],
            "intervention_mean": float(mean_ie),
            "intervention_ci_lo": float(ci_lo),
            "intervention_ci_hi": float(ci_hi),
            "n_significant": n_sig,
            "n_with_intervention": len(ies),
        }

    # Sector pairwise tests (bootstrap differences)
    print(f"\n{'='*70}")
    print("SECTOR DIFFERENCES (Bootstrap)")
    print(f"{'='*70}")

    sector_names = sorted(sectors.keys())
    for i, s1 in enumerate(sector_names):
        for s2 in sector_names[i+1:]:
            ie1 = [r["intervention_effect"] for r in sectors[s1] if r.get("intervention_effect") is not None]
            ie2 = [r["intervention_effect"] for r in sectors[s2] if r.get("intervention_effect") is not None]
            if len(ie1) < 3 or len(ie2) < 3: continue
            diff = np.mean(ie2) - np.mean(ie1)
            # Bootstrap the difference
            diffs = []
            for _ in range(10000):
                b1 = np.random.choice(ie1, len(ie1), replace=True)
                b2 = np.random.choice(ie2, len(ie2), replace=True)
                diffs.append(np.mean(b2) - np.mean(b1))
            ci_lo, ci_hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
            sig = ci_lo > 0 or ci_hi < 0
            print(f"  {s2} - {s1}: {diff:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] {'SIG' if sig else 'ns'}")

    # Overall
    ies_all = [r["intervention_effect"] for r in results if r.get("intervention_effect") is not None]
    alive_all = [r["alive"] for r in results]
    ves_all = [r["ve"] for r in results]

    print(f"\n{'='*70}")
    print(f"OVERALL ({len(results)} stocks)")
    print(f"{'='*70}")
    print(f"Mean intervention effect: {np.mean(ies_all):.4f} ± {np.std(ies_all):.4f}")
    print(f"Mean alive features: {np.mean(alive_all):.1%} ± {np.std(alive_all):.1%}")
    print(f"Mean VE: {np.mean(ves_all):.4f}")

    final = {"n_stocks": len(results), "layer": args.layer,
             "sector_stats": sector_stats, "per_stock": results}
    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved. Total time: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
