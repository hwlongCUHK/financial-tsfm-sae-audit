"""Multi-stock SAE: train, interpret, and causally validate on 6 stocks."""
import torch
import numpy as np
import json, sys, time, argparse
from pathlib import Path
import pandas as pd
from scipy import stats

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
        x_centered = x - self.b_pre
        latents = self.encoder(x_centered)
        topk_vals, topk_idx = torch.topk(latents, self.k, dim=-1)
        mask = torch.zeros_like(latents)
        mask.scatter_(-1, topk_idx, 1.0)
        latents = latents * mask
        return self.decoder(latents) + self.b_pre, latents

    def ablate_reconstruct(self, x, ablate_ids):
        x_centered = x - self.b_pre
        latents = self.encoder(x_centered)
        topk_vals, topk_idx = torch.topk(latents, self.k, dim=-1)
        mask = torch.zeros_like(latents)
        mask.scatter_(-1, topk_idx, 1.0)
        mask[:, ablate_ids] = 0
        return self.decoder(latents * mask) + self.b_pre


def process_stock(csv_path, stock_name, tokenizer, model, device, args):
    """Full pipeline for one stock: load data, train SAE, interpret, causal validate."""
    print(f"\n{'='*60}")
    print(f"STOCK: {stock_name}")
    print(f"{'='*60}")

    # Load data
    df = pd.read_csv(str(csv_path))
    for col in ["open","close","high","low","volume","amount"]:
        if col not in df.columns: df[col] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 200:
        print("  SKIP: too few rows")
        return None

    mn, st = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)

    lookback = 64
    stride = 32
    n_windows = (len(data_norm) - lookback) // stride
    windows = np.stack([data_norm[i:i+lookback] for i in range(0, min(n_windows, 2000) * stride, stride)])
    print(f"  Windows: {windows.shape}")

    # Split: 80% for SAE training, 20% for causal test
    n_train = int(len(windows) * 0.8)
    train_wins = windows[:n_train]
    test_wins = windows[n_train:]

    # ─── 1. Collect activations ───
    n_layers = len(model.transformer)
    layer = args.layer
    acts_list = []

    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        acts_list.append(act[:, -1, :].detach().cpu().float().numpy())

    hook = model.transformer[layer].register_forward_hook(hook_fn)
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(train_wins), batch_size):
            batch = torch.from_numpy(train_wins[i:i+batch_size]).float().to(device)
            s1_ids, s2_ids = tokenizer.encode(batch, half=True)
            model(s1_ids, s2_ids)
    hook.remove()

    train_acts = np.concatenate(acts_list, axis=0)
    print(f"  Train activations: {train_acts.shape}")

    # ─── 2. Train SAE ───
    d_model = train_acts.shape[1]
    d_hidden = d_model * 4
    sae = TopKSAE(d_model, d_hidden, k=64).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
    acts_t = torch.from_numpy(train_acts).float().to(device)

    for step in range(args.steps):
        idx = torch.randint(0, len(acts_t), (args.batch,))
        x = acts_t[idx]
        x_recon, latents = sae(x)
        loss = torch.nn.functional.mse_loss(x_recon, x)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()

    # Evaluate SAE
    with torch.no_grad():
        x_test = acts_t[:min(500, len(acts_t))]
        recon, latents = sae(x_test)
        var_exp = 1 - torch.nn.functional.mse_loss(recon, x_test).item() / x_test.var().item()
        per_sample_sparsity = (latents != 0).float().sum(dim=-1).mean().item()
        ever_active = (latents.abs().sum(dim=0) > 1e-6).float().mean().item()
    print(f"  SAE: var_exp={var_exp:.4f}, per-sample-L0={per_sample_sparsity:.1f}, ever-active={ever_active:.1%}")

    # ─── 3. Interpretability ───
    # Financial labels
    labels = []
    stride_l = 32
    for i in range(0, len(data_norm) - lookback * 2, stride_l):
        close = data_norm[i:i+lookback, 1]
        rets = np.diff(close) / (close[:-1] + 1e-5)
        labels.append({
            "vol": np.std(rets),
            "trend": np.polyfit(np.arange(lookback), close, 1)[0],
            "max_dd": float(np.min(close / np.maximum.accumulate(close) - 1)),
            "range": float((close.max() - close.min()) / close.mean()),
            "vol_cluster": float(np.mean(rets**2) / (rets.std()**2 + 1e-5)),
            "skew": float(np.mean((rets - rets.mean())**3) / (rets.std()**3 + 1e-5)),
            "kurt": float(np.mean((rets - rets.mean())**4) / (rets.std()**4 + 1e-5)),
        })

    n_match = min(len(train_acts), len(labels))
    train_latents_full = []
    with torch.no_grad():
        for i in range(0, n_match, 256):
            batch = torch.from_numpy(train_acts[i:i+256]).float().to(device)
            _, lat = sae(batch)
            train_latents_full.append(lat.cpu().numpy())
    train_latents_full = np.concatenate(train_latents_full, axis=0)[:n_match]
    labels = labels[:n_match]

    # Per-feature correlations
    label_keys = ["vol", "trend", "max_dd", "range", "vol_cluster", "skew", "kurt"]
    label_names = ["Volatility", "Trend", "Max Drawdown", "Price Range", "Vol Clustering", "Skewness", "Kurtosis"]
    label_arr = np.array([[l[k] for k in label_keys] for l in labels])

    feat_active = (train_latents_full != 0).sum(axis=0) > 10
    alive_idx = np.where(feat_active)[0]
    alive_latents = train_latents_full[:, alive_idx]

    type_dist = {}
    n_high_corr = 0
    for j in range(len(alive_idx)):
        feat_vals = alive_latents[:, j]
        active = feat_vals != 0
        if active.sum() < 5:
            continue
        corrs = [np.corrcoef(feat_vals[active], label_arr[active, k])[0, 1] for k in range(len(label_keys))]
        corrs = [0 if np.isnan(c) else c for c in corrs]
        best_k = np.argmax(np.abs(corrs))
        type_dist[label_names[best_k]] = type_dist.get(label_names[best_k], 0) + 1
        if abs(corrs[best_k]) > 0.3:
            n_high_corr += 1

    # ─── 4. Model-level causal validation ───
    test_wins_t = torch.from_numpy(test_wins[:min(30, len(test_wins))]).float().to(device)

    # Baseline
    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(test_wins_t, half=True)
        baseline_out = model(s1_ids, s2_ids)
    baseline_s1 = baseline_out[0].float()

    # Get top features from training activations
    with torch.no_grad():
        _, all_lat = sae(acts_t[:min(1000, len(acts_t))])
        feat_usage = (all_lat != 0).float().sum(dim=0)
    top100 = feat_usage.argsort(descending=True)[:100]

    # Ablation experiments
    ab_results = []
    for n_ab in [20, 50, 100]:
        ablate_ids = top100[:n_ab].tolist()

        def make_intervention(ids):
            def intervene(module, input, output):
                orig = output[0] if isinstance(output, tuple) else output
                B, T, D = orig.shape
                flat = orig.reshape(-1, D).float()
                ablated = sae.ablate_reconstruct(flat, ids).reshape(B, T, D).half()
                return (ablated,) + output[1:] if isinstance(output, tuple) else ablated
            return intervene

        hook = model.transformer[layer].register_forward_hook(make_intervention(ablate_ids))
        with torch.no_grad():
            s1_ids, s2_ids = tokenizer.encode(test_wins_t, half=True)
            ab_out = model(s1_ids, s2_ids)
        hook.remove()

        cos_sim = torch.nn.functional.cosine_similarity(
            baseline_s1.reshape(-1, baseline_s1.shape[-1]),
            ab_out[0].float().reshape(-1, baseline_s1.shape[-1]), dim=-1).mean()
        top1_agree = (baseline_s1.argmax(-1) == ab_out[0].float().argmax(-1)).float().mean()

        ab_results.append({"n_ablated": n_ab, "cos_sim": float(cos_sim.item()), "top1_agree": float(top1_agree.item())})

    # Random baseline matching activation frequency
    rand_cos = []
    for trial in range(20):
        # Match frequency: pick features with similar activation counts to top20
        top20_freq = feat_usage[top100[:20]].float().mean().item()
        candidates = torch.where((feat_usage > top20_freq * 0.5) & (feat_usage < top20_freq * 2.0))[0]
        if len(candidates) < 20:
            candidates = torch.arange(len(feat_usage))
        rand_ids = candidates[torch.randperm(len(candidates))[:20]].tolist()

        hook = model.transformer[layer].register_forward_hook(make_intervention(rand_ids))
        with torch.no_grad():
            s1_ids, s2_ids = tokenizer.encode(test_wins_t, half=True)
            rand_out = model(s1_ids, s2_ids)
        hook.remove()
        cos_r = torch.nn.functional.cosine_similarity(
            baseline_s1.reshape(-1, baseline_s1.shape[-1]),
            rand_out[0].float().reshape(-1, baseline_s1.shape[-1]), dim=-1).mean()
        rand_cos.append(float(cos_r.item()))

    rand_mean = np.mean(rand_cos)
    rand_std = np.std(rand_cos)
    top20_cos = ab_results[0]["cos_sim"]
    t_stat, p_val = stats.ttest_1samp(rand_cos, top20_cos)

    print(f"  Causal: top20_cos={top20_cos:.4f} rand_cos={rand_mean:.4f}±{rand_std:.4f} p={p_val:.4f}")

    del sae
    torch.cuda.empty_cache()

    return {
        "stock": stock_name,
        "n_windows": len(windows),
        "sae_var_explained": float(var_exp),
        "sae_per_sample_l0": float(per_sample_sparsity),
        "sae_ever_active": float(ever_active),
        "n_alive_features": int(len(alive_idx)),
        "n_high_corr": n_high_corr,
        "type_distribution": type_dist,
        "ablations": ab_results,
        "random_baseline": {"mean": rand_mean, "std": rand_std},
        "top20_vs_random_p": float(p_val),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/houwanlong/finllm-mi/data/scale")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/multi_stock_results.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stocks", type=str, default="")
    args = parser.parse_args()

    device = args.device
    t0 = time.time()

    print("Loading Kronos...")
    tokenizer = KronosTokenizer.from_pretrained("/data/houwanlong/models/Kronos-Tokenizer-base").to(device).eval()
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

    # Select stocks
    data_dir = Path(args.data_dir)
    all_csvs = sorted(data_dir.glob("sh*.csv"))
    if args.stocks:
        selected = [data_dir / f"{s}.csv" for s in args.stocks.split(",")]
    else:
        # 6 diverse stocks from different sectors
        tickers = ["sh600519", "sh601857", "sz000725", "sh601398", "sh600276", "sh600104"]
        selected = [data_dir / f"{t}.csv" for t in tickers]

    print(f"Processing {len(selected)} stocks on layer {args.layer}")

    results = []
    for csv_path in selected:
        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found")
            continue
        r = process_stock(csv_path, csv_path.stem, tokenizer, model, device, args)
        if r:
            results.append(r)

    # ─── Aggregate ───
    print(f"\n{'='*70}")
    print(f"MULTI-STOCK RESULTS: {len(results)} stocks")
    print(f"{'='*70}")

    var_exps = [r["sae_var_explained"] for r in results]
    l0s = [r["sae_per_sample_l0"] for r in results]
    evers = [r["sae_ever_active"] for r in results]
    top20_cos_list = [r["ablations"][0]["cos_sim"] for r in results]
    ps = [r["top20_vs_random_p"] for r in results]

    # Aggregate type distribution
    total_type = {}
    for r in results:
        for t, c in r["type_distribution"].items():
            total_type[t] = total_type.get(t, 0) + c
    total_features = sum(total_type.values())
    print(f"\nAggregate feature types ({total_features} features):")
    for t, c in sorted(total_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c/total_features*100:.1f}%")

    print(f"\nPer-stock summary:")
    print(f"{'Stock':<15} {'VarExp':<10} {'L0':<8} {'Alive':<8} {'Top20cos':<10} {'p(rand)':<10}")
    for r in results:
        print(f"{r['stock']:<15} {r['sae_var_explained']:.4f}    {r['sae_per_sample_l0']:.1f}    "
              f"{r['sae_ever_active']:.1%}   {r['ablations'][0]['cos_sim']:.4f}    {r['top20_vs_random_p']:.4f}")

    print(f"\nMean var_exp: {np.mean(var_exps):.4f} ± {np.std(var_exps):.4f}")
    print(f"Mean L0: {np.mean(l0s):.1f} ± {np.std(l0s):.1f}")
    print(f"Mean ever-active: {np.mean(evers):.1%} ± {np.std(evers):.1%}")
    print(f"Top20 cos: {np.mean(top20_cos_list):.4f} ± {np.std(top20_cos_list):.4f}")
    n_sig = sum(1 for p in ps if p < 0.05)
    print(f"Significant (p<0.05) in {n_sig}/{len(results)} stocks")

    # Overall paired test (top20 vs random mean)
    all_deltas = [r["ablations"][0]["cos_sim"] - r["random_baseline"]["mean"] for r in results]
    t_all, p_all = stats.ttest_1samp(all_deltas, 0)
    print(f"\nOverall delta (top20 - random): {np.mean(all_deltas):.6f} ± {np.std(all_deltas):.6f}")
    print(f"Paired test: t={t_all:.3f}, p={p_all:.6f}")

    final = {
        "n_stocks": len(results),
        "layer": args.layer,
        "aggregate": {
            "var_exp_mean": float(np.mean(var_exps)),
            "per_sample_l0_mean": float(np.mean(l0s)),
            "ever_active_mean": float(np.mean(evers)),
            "top20_cos_mean": float(np.mean(top20_cos_list)),
            "n_significant": n_sig,
            "overall_delta": float(np.mean(all_deltas)),
            "overall_p": float(p_all),
        },
        "type_distribution": total_type,
        "per_stock": results,
    }
    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nSaved to {args.output}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
