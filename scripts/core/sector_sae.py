"""Sector-level SAE: discover different financial concept signatures per sector."""
import torch, numpy as np, json, sys, time, argparse
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
        xc = x - self.b_pre
        lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        return self.decoder(lat * mask) + self.b_pre, lat * mask
    def ablate_reconstruct(self, x, ids):
        xc = x - self.b_pre
        lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        mask[:, ids] = 0
        return self.decoder(lat * mask) + self.b_pre


STOCKS = {
    # Banking (stable, regulated, mean-reverting)
    "Bank_ICBC": ("sh601398", "Bank"),
    "Bank_CMB": ("sh600036", "Bank"),
    "Bank_BOC": ("sh601988", "Bank"),
    # Energy (commodity-driven, volatile, cyclical)
    "Energy_PetroChina": ("sh601857", "Energy"),
    "Energy_Sinopec": ("sh600028", "Energy"),
    "Energy_Coal": ("sh601088", "Energy"),
    # Tech (high-growth, volatile, momentum-driven)
    "Tech_BOE": ("sz000725", "Tech"),
    "Tech_ChinaUnicom": ("sh600050", "Tech"),
    "Tech_Hengsheng": ("sh600570", "Tech"),
    # Consumer/Pharma (stable growth, defensive)
    "Cons_Moutai": ("sh600519", "Consumer"),
    "Cons_Hengrui": ("sh600276", "Consumer"),
    "Cons_SAIC": ("sh600104", "Consumer"),
}


def process_stock(csv_path, stock_name, sector, tokenizer, model, device, args):
    """SAE train + interpret + causal for one stock."""
    csv_path = Path(args.data_dir) / f"{csv_path}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(str(csv_path))
    for col in ["open","close","high","low","volume","amount"]:
        if col not in df.columns: df[col] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 200: return None
    mn, st = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)

    lookback, stride = 64, 32
    n_windows = min(2000, (len(data_norm) - lookback) // stride)
    windows = np.stack([data_norm[i:i+lookback] for i in range(0, n_windows * stride, stride)])

    n_train = int(len(windows) * 0.8)
    train_wins, test_wins = windows[:n_train], windows[n_train:]

    # Collect activations
    layer = args.layer
    acts_list = []
    def hook_fn(m, i, o):
        a = o[0] if isinstance(o, tuple) else o
        acts_list.append(a[:, -1, :].detach().cpu().float().numpy())
    hook = model.transformer[layer].register_forward_hook(hook_fn)

    bs = 64
    with torch.no_grad():
        for b in range(0, len(train_wins), bs):
            batch = torch.from_numpy(train_wins[b:b+bs]).float().to(device)
            s1, s2 = tokenizer.encode(batch, half=True)
            model(s1, s2)
    hook.remove()
    acts = np.concatenate(acts_list, axis=0)

    # Train SAE
    d_model, d_hidden = acts.shape[1], acts.shape[1] * 4
    sae = TopKSAE(d_model, d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
    acts_t = torch.from_numpy(acts).float().to(device)

    for step in range(args.steps):
        idx = torch.randint(0, len(acts_t), (args.batch,))
        xr, _ = sae(acts_t[idx])
        loss = torch.nn.functional.mse_loss(xr, acts_t[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

    # Evaluate
    with torch.no_grad():
        xt = acts_t[:min(500, len(acts_t))]
        recon, lat = sae(xt)
        ve = 1 - torch.nn.functional.mse_loss(recon, xt).item() / xt.var().item()
        l0 = (lat != 0).float().sum(-1).mean().item()
        alive = (lat.abs().sum(0) > 1e-6).float().mean().item()

    # Interpret: financial labels
    label_keys = ["vol", "trend", "max_dd", "range", "vol_cluster", "skew", "kurt"]
    label_names = ["Volatility", "Trend", "Max Drawdown", "Price Range", "Vol Clustering", "Skewness", "Kurtosis"]
    labels = []
    for i in range(0, len(data_norm) - lookback * 2, stride):
        c = data_norm[i:i+lookback, 1]; r = np.diff(c) / (c[:-1] + 1e-5)
        labels.append({"vol": np.std(r), "trend": np.polyfit(np.arange(lookback), c, 1)[0],
                       "max_dd": float(np.min(c / np.maximum.accumulate(c) - 1)),
                       "range": float((c.max() - c.min()) / c.mean()),
                       "vol_cluster": float(np.mean(r**2) / (r.std()**2 + 1e-5)),
                       "skew": float(np.mean((r - r.mean())**3) / (r.std()**3 + 1e-5)),
                       "kurt": float(np.mean((r - r.mean())**4) / (r.std()**4 + 1e-5))})

    n = min(len(acts), len(labels))
    full_lat = []
    with torch.no_grad():
        for i in range(0, n, 256):
            _, lat2 = sae(acts_t[i:i+256])
            full_lat.append(lat2.cpu().numpy())
    full_lat = np.concatenate(full_lat)[:n]
    label_arr = np.array([[l[k] for k in label_keys] for l in labels[:n]])

    # Per-feature type
    alive_mask = (full_lat != 0).sum(0) > 10
    type_dist = {}
    n_strong = 0
    for j in np.where(alive_mask)[0]:
        act = full_lat[:, j]; a = act != 0
        if a.sum() < 5: continue
        corrs = [np.corrcoef(act[a], label_arr[a, k])[0,1] for k in range(len(label_keys))]
        corrs = [0 if np.isnan(c) else c for c in corrs]
        best = np.argmax(np.abs(corrs))
        type_dist[label_names[best]] = type_dist.get(label_names[best], 0) + 1
        if abs(corrs[best]) > 0.3: n_strong += 1

    # Find "signature" — top 2 dominant concepts for this stock
    sig = sorted(type_dist.items(), key=lambda x: -x[1])[:2]
    signature = f"{sig[0][0]} ({sig[0][1]/sum(type_dist.values())*100:.0f}%) + {sig[1][0]} ({sig[1][1]/sum(type_dist.values())*100:.0f}%)" if len(sig) >= 2 else sig[0][0]

    # Causal: model-level intervention
    test_wins_t = torch.from_numpy(test_wins[:min(30, len(test_wins))]).float().to(device)
    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_wins_t, half=True)
        base = model(s1, s2)
    base_s1 = base[0].float()

    with torch.no_grad():
        _, all_l = sae(acts_t[:min(1000, len(acts_t))])
    freq = (all_l != 0).float().sum(0)
    top100 = freq.argsort(descending=True)[:100]

    ablations = []
    for n_ab in [20, 50, 100]:
        ids = top100[:n_ab].tolist()
        def make_int(ab_ids):
            def intervene(m, i, o):
                orig = o[0] if isinstance(o, tuple) else o
                B, T, D = orig.shape
                ablated = sae.ablate_reconstruct(orig.reshape(-1, D).float(), ab_ids).reshape(B, T, D).half()
                return (ablated,) + o[1:] if isinstance(o, tuple) else ablated
            return intervene
        hk = model.transformer[layer].register_forward_hook(make_int(ids))
        with torch.no_grad():
            s1, s2 = tokenizer.encode(test_wins_t, half=True)
            ab = model(s1, s2)
        hk.remove()
        cs = torch.nn.functional.cosine_similarity(base_s1.reshape(-1, base_s1.shape[-1]), ab[0].float().reshape(-1, base_s1.shape[-1]), dim=-1).mean()
        ablations.append({"n": n_ab, "cos": float(cs.item())})

    del sae; torch.cuda.empty_cache()
    return {"stock": stock_name, "sector": sector, "n_windows": len(windows),
            "ve": float(ve), "l0": float(l0), "alive": float(alive),
            "type_dist": type_dist, "signature": signature, "n_strong": n_strong,
            "ablations": ablations, "n_alive": int(alive_mask.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/houwanlong/finllm-mi/data/scale")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/sector_results.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device; t0 = time.time()
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

    results = []
    for name, (ticker, sector) in STOCKS.items():
        print(f"\n{name} ({sector})...", end=" ", flush=True)
        r = process_stock(ticker, name, sector, tokenizer, model, device, args)
        if r:
            results.append(r)
            print(f"VE={r['ve']:.3f} alive={r['alive']:.1%} sig={r['signature']}")
        else:
            print("SKIP")

    # ─── SECTOR STORIES ───
    print(f"\n{'='*70}")
    print(f"SECTOR STORIES — {len(results)} stocks across 4 sectors")
    print(f"{'='*70}")

    sectors = defaultdict(list)
    for r in results:
        sectors[r["sector"]].append(r)

    all_sector_data = {}
    for sector, stocks in sectors.items():
        print(f"\n## {sector.upper()} ({len(stocks)} stocks)")
        # Aggregate type distribution
        merged = {}
        for s in stocks:
            for t, c in s["type_dist"].items():
                merged[t] = merged.get(t, 0) + c
        total = sum(merged.values())
        top3 = sorted(merged.items(), key=lambda x: -x[1])[:3]
        print(f"  Dominant concepts: {', '.join(f'{t} ({c/total*100:.0f}%)' for t, c in top3)}")

        # Per-stock signatures
        print(f"  Stock signatures:")
        ves = []; alives = []; strgs = []; cos50 = []
        for s in stocks:
            print(f"    {s['stock']}: {s['signature']} (strong: {s['n_strong']}/{s['n_alive']})")
            ves.append(s['ve']); alives.append(s['alive'])
            strgs.append(s['n_strong']); cos50.append(s['ablations'][1]['cos'] if len(s['ablations']) > 1 else 0)

        # Sector narrative
        avg_strong = np.mean(strgs)
        avg_cos = np.mean(cos50)
        print(f"  Sector narrative: {sector} stocks average {avg_strong:.0f} strongly correlated features. "
              f"Ablating top 50 features: cos={avg_cos:.3f}. "
              f"Key insight: {top3[0][0]} dominates in {sector}.")

        all_sector_data[sector] = {
            "top_concepts": [(t, int(c), float(c/total*100)) for t, c in top3],
            "mean_strong_features": float(avg_strong),
            "mean_ve": float(np.mean(ves)),
            "mean_alive": float(np.mean(alives)),
            "mean_cos50": float(avg_cos),
            "stocks": [s["stock"] for s in stocks],
        }

    # ─── OVERALL ───
    print(f"\n{'='*70}")
    print("OVERALL")
    print(f"{'='*70}")
    ve_all = [r["ve"] for r in results]
    alive_all = [r["alive"] for r in results]
    cos50_all = [r["ablations"][1]["cos"] if len(r["ablations"]) > 1 else 0 for r in results]
    print(f"Stocks: {len(results)}, Mean VE: {np.mean(ve_all):.4f}, Mean alive: {np.mean(alive_all):.1%}")
    print(f"Mean cos(top50): {np.mean(cos50_all):.4f}")

    final = {"n_stocks": len(results), "layer": args.layer,
             "sectors": all_sector_data, "per_stock": results}
    with open(args.output, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nSaved. {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
