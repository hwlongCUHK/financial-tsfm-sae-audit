"""Financial task impact: ablate concept features, measure downstream forecast change."""
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
    mn, st_d = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st_d + 1e-5), -5, 5)

    lb, stride = 64, 32
    nw = min(2000, (len(data_norm) - lb) // stride)
    windows = np.stack([data_norm[i:i+lb] for i in range(0, nw*stride, stride)])
    n_train = int(len(windows) * 0.8)

    # Activations for SAE training
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

    # Label features via correlation + LLM labels from global analysis
    label_keys = ["vol","trend","max_dd","range","vol_cluster","skew","kurt"]
    label_names = ["Volatility","Trend","Max Drawdown","Price Range","Vol Clustering","Skewness","Kurtosis"]
    labels = []
    for i in range(0, len(data_norm) - lb * 2, stride):
        c = data_norm[i:i+lb, 1]; r = np.diff(c) / (c[:-1] + 1e-5)
        labels.append({k: (np.std(r) if k=="vol" else
            np.polyfit(np.arange(lb), c, 1)[0] if k=="trend" else
            float(np.min(c / np.maximum.accumulate(c) - 1)) if k=="max_dd" else
            float((c.max() - c.min()) / c.mean()) if k=="range" else
            float(np.mean(r**2) / (r.std()**2 + 1e-5)) if k=="vol_cluster" else
            float(np.mean((r - r.mean())**3) / (r.std()**3 + 1e-5)) if k=="skew" else
            float(np.mean((r - r.mean())**4) / (r.std()**4 + 1e-5))) for k in label_keys})

    n = min(len(acts), len(labels))
    full_lat = []
    with torch.no_grad():
        for i in range(0, n, 256):
            _, lat2 = sae(at[i:i+256])
            full_lat.append(lat2.cpu().numpy())
    full_lat = np.concatenate(full_lat)[:n]
    label_arr = np.array([[l[k] for k in label_keys] for l in labels[:n]])

    # Assign each alive feature to a concept
    alive_mask = (full_lat != 0).sum(0) > 10
    concept_features = defaultdict(list)
    for j in np.where(alive_mask)[0]:
        act = full_lat[:, j]; a = act != 0
        if a.sum() < 5: continue
        corrs = [np.corrcoef(act[a], label_arr[a, k])[0,1] for k in range(len(label_keys))]
        corrs = [0 if np.isnan(c) else c for c in corrs]
        best = np.argmax(np.abs(corrs))
        if abs(corrs[best]) > 0.2:  # Only features with meaningful correlation
            concept_features[label_names[best]].append(j)

    # ─── FINANCIAL TASK IMPACT ───
    test_start = n_train
    test_wins = windows[test_start:test_start+min(50, len(windows)-test_start)]
    if len(test_wins) < 10:
        del sae; torch.cuda.empty_cache()
        return None

    test_t = torch.from_numpy(test_wins).float().to(device)

    # Baseline predictions: run Kronos on test windows, get output tokens
    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
        base_out = model(s1_ids, s2_ids)
    base_s1 = base_out[0].float()  # (B, T, vocab)

    # Ground truth: next-token at each position
    # For autoregressive model, output[t] predicts input[t+1]
    # Financial task: how well does the model predict the NEXT token?
    def compute_forecast_quality(output_logits, s1_targets):
        """Measure financial prediction quality."""
        # Top-1 accuracy of next-token prediction
        preds = output_logits.argmax(dim=-1)  # (B, T)
        targets = s1_targets  # (B, T)
        # Shift: output[:, t] predicts target[:, t+1]
        # Use last 10 positions for financial forecast evaluation
        pred_last = preds[:, -10:]  # last 10 predicted tokens
        target_last = targets[:, -10:]

        # Top-1 accuracy
        acc = (pred_last == target_last).float().mean().item()

        # Rank correlation: how well does the model predict the ORDERING of tokens?
        # Compare top-5 predictions between baseline and ablated
        top5_base = output_logits.topk(5, dim=-1).indices[:, -10:, :]

        return acc, top5_base

    base_acc, base_top5 = compute_forecast_quality(base_s1, s1_ids)

    # ─── CONCEPT-SPECIFIC ABLATION ───
    concept_impacts = {}
    concepts_to_test = [c for c, feats in concept_features.items() if len(feats) >= 5]

    for concept in concepts_to_test:
        feat_ids = concept_features[concept][:20]  # Top 20 features per concept

        def make_int(ab_ids):
            def intervene(m, i, o):
                orig = o[0] if isinstance(o, tuple) else o
                B, T, D = orig.shape
                ablated = sae.ablate_reconstruct(orig.reshape(-1, D).float(), ab_ids).reshape(B, T, D).half()
                return (ablated,) + o[1:] if isinstance(o, tuple) else ablated
            return intervene

        hk = model.transformer[layer].register_forward_hook(make_int(feat_ids))
        with torch.no_grad():
            s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
            ab_out = model(s1_ids, s2_ids)
        hk.remove()

        ab_s1 = ab_out[0].float()
        ab_acc, ab_top5 = compute_forecast_quality(ab_s1, s1_ids)

        # Top-5 agreement change
        top5_agree = (base_top5 == ab_top5).float().mean().item()

        # For random baseline: same number of features, frequency-matched
        rand_accs = []
        rand_agrees = []
        for _ in range(10):
            rids = np.random.choice(d_hidden, len(feat_ids), replace=False).tolist()
            hk2 = model.transformer[layer].register_forward_hook(make_int(rids))
            with torch.no_grad():
                s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
                rand_out = model(s1_ids, s2_ids)
            hk2.remove()
            rand_s1 = rand_out[0].float()
            r_acc, r_top5 = compute_forecast_quality(rand_s1, s1_ids)
            rand_accs.append(r_acc)
            rand_agrees.append((base_top5 == r_top5).float().mean().item())

        rand_acc_mean = np.mean(rand_accs)
        rand_acc_std = np.std(rand_accs)
        acc_drop = base_acc - ab_acc  # Positive = ablation hurts accuracy
        z_acc = (acc_drop - 0) / (rand_acc_std + 1e-10)  # How many std above zero
        p_acc = 1 - stats.norm.cdf(abs(acc_drop) / (rand_acc_std + 1e-10))

        concept_impacts[concept] = {
            "n_features": len(feat_ids),
            "base_acc": float(base_acc),
            "ablated_acc": float(ab_acc),
            "acc_drop": float(acc_drop),
            "rand_acc_mean": float(rand_acc_mean),
            "top5_agree": float(top5_agree),
            "rand_top5_agree": float(np.mean(rand_agrees)),
            "significant": bool(acc_drop > rand_acc_std * 2),
        }

    # Also: top-50 features (replicating earlier experiment)
    with torch.no_grad():
        _, all_l = sae(at[:min(1000, len(at))])
    freq = (all_l != 0).float().sum(0)
    top50 = freq.argsort(descending=True)[:50].tolist()

    hk = model.transformer[layer].register_forward_hook(make_int(top50))
    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
        top50_out = model(s1_ids, s2_ids)
    hk.remove()
    top50_acc, _ = compute_forecast_quality(top50_out[0].float(), s1_ids)
    top50_drop = base_acc - top50_acc

    del sae; torch.cuda.empty_cache()
    return {
        "base_acc": float(base_acc),
        "top50_acc_drop": float(top50_drop),
        "concept_impacts": concept_impacts,
        "time": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/houwanlong/finllm-mi/data/scale120")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/financial_impact.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-stocks", type=int, default=0)
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
    with open("/tmp/sectors120.json") as f:
        sector_map = json.load(f)
    ticker_sector = {}
    for sector, tickers in sector_map.items():
        for t in tickers:
            ticker_sector[t] = sector

    csv_files = sorted(Path(args.data_dir).glob("*.csv"))
    if args.max_stocks > 0:
        csv_files = csv_files[:args.max_stocks]

    print(f"Processing {len(csv_files)} stocks with financial task impact...")

    results = []
    for i, f in enumerate(csv_files):
        ticker = f.stem
        sector = ticker_sector.get(ticker, "Other")
        print(f"[{i+1}/{len(csv_files)}] {ticker} ({sector})...", end=" ", flush=True)
        r = process_one(f, tokenizer, model, args.layer, device, args)
        if r:
            r["ticker"] = ticker; r["sector"] = sector
            results.append(r)
            ci = r.get("concept_impacts", {})
            sig_concepts = [c for c, v in ci.items() if v.get("significant")]
            print(f"acc={r['base_acc']:.3f} top50_drop={r['top50_acc_drop']:.4f} sig_concepts={sig_concepts}")
        else:
            print("SKIP")

    # ─── Aggregate ───
    print(f"\n{'='*70}")
    print(f"FINANCIAL TASK IMPACT — {len(results)} stocks")
    print(f"{'='*70}")

    # Per-concept aggregate
    all_concepts = set()
    for r in results:
        all_concepts.update(r["concept_impacts"].keys())

    print("\nPer-concept financial impact (acc drop, higher = more impact):")
    concept_agg = {}
    for concept in sorted(all_concepts):
        drops = []
        sigs = []
        for r in results:
            if concept in r["concept_impacts"]:
                ci = r["concept_impacts"][concept]
                drops.append(ci["acc_drop"])
                sigs.append(ci["significant"])
        if drops:
            mean_drop = np.mean(drops)
            # Bootstrap CI
            boots = [np.mean(np.random.choice(drops, len(drops), replace=True)) for _ in range(10000)]
            ci_lo, ci_hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
            sig_pct = np.mean(sigs)
            concept_agg[concept] = {
                "mean_acc_drop": float(mean_drop),
                "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                "significant_pct": float(sig_pct),
                "n_stocks": len(drops),
            }
            sig_marker = "SIG" if ci_lo > 0 else "ns"
            print(f"  {concept:<20}: drop={mean_drop:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] {sig_marker} ({sig_pct:.0%} sig)")

    # Sector-level top50 impact
    sectors = defaultdict(list)
    for r in results:
        sectors[r["sector"]].append(r["top50_acc_drop"])

    print(f"\nTop-50 feature ablation → forecast accuracy drop by sector:")
    for sector, drops in sorted(sectors.items()):
        md = np.mean(drops)
        boots = [np.mean(np.random.choice(drops, len(drops), replace=True)) for _ in range(10000)]
        ci_lo, ci_hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        sig = "SIG" if ci_lo > 0 else "ns"
        print(f"  {sector:<12}: drop={md:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] {sig} (n={len(drops)})")

    # Overall
    all_drops = [r["top50_acc_drop"] for r in results]
    mean_d = np.mean(all_drops)
    boots = [np.mean(np.random.choice(all_drops, len(all_drops), replace=True)) for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    print(f"\nOverall top-50 acc drop: {mean_d:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] {'SIG' if ci_lo > 0 else 'ns'}")

    final = {
        "n_stocks": len(results), "layer": args.layer,
        "concept_aggregate": concept_agg,
        "per_stock": results,
        "overall_top50_drop": {"mean": float(mean_d), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)},
    }
    with open(args.output, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nSaved. {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
