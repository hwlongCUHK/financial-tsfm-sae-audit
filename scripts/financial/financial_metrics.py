"""RankIC + Directional Accuracy from SAE feature ablation — financially meaningful metrics."""
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

def compute_rankic(predictions, actuals):
    """Rank Information Coefficient: Spearman correlation between predicted and actual returns."""
    if len(predictions) < 5:
        return 0.0
    rho, _ = stats.spearmanr(predictions, actuals)
    return rho if not np.isnan(rho) else 0.0

def compute_directional_accuracy(predictions, actuals):
    """Fraction of predictions with correct sign."""
    correct = (np.sign(predictions) == np.sign(actuals)).sum()
    return correct / len(predictions) if len(predictions) > 0 else 0.0

def compute_volatility_error(pred_vol, actual_vol):
    """MSE between predicted and actual volatility."""
    return np.mean((pred_vol - actual_vol)**2)

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

    # Train SAE
    acts_list = []
    def make_hook(storage):
        def h(m, i, o):
            a = o[0] if isinstance(o, tuple) else o
            storage.append(a[:, -1, :].detach().cpu().float().numpy())
        return h
    acts_train = []
    hook = model.transformer[layer].register_forward_hook(make_hook(acts_train))
    bs = 64
    with torch.no_grad():
        for b in range(0, n_train, bs):
            batch = torch.from_numpy(windows[b:b+bs]).float().to(device)
            s1, s2 = tokenizer.encode(batch, half=True)
            model(s1, s2)
    hook.remove()
    acts = np.concatenate(acts_train, axis=0)

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
        _, all_lat = sae(at[:min(1000, len(at))])
    freq = (all_lat != 0).float().sum(0)
    top50 = freq.argsort(descending=True)[:50].tolist()

    # Financial metrics on test windows
    test_start = n_train
    n_test = min(30, len(windows) - test_start)
    if n_test < 5:
        del sae; torch.cuda.empty_cache(); return None
    test_wins = windows[test_start:test_start+n_test]
    test_t = torch.from_numpy(test_wins).float().to(device)

    # Baseline forward pass
    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
        base_out = model(s1_ids, s2_ids)
    base_s1 = base_out[0].float()  # (B, T, vocab_s1)

    # Decode s1 logits back to price predictions
    # For autoregressive model, output[t] predicts input[t+1]
    # Use last position prediction as the forecast
    base_preds = base_s1[:, -1, :].argmax(dim=-1).float().cpu().numpy()  # (B,)

    # Ablate top 50 features
    def make_int(ab_ids):
        def intervene(m, i, o):
            orig = o[0] if isinstance(o, tuple) else o
            B, T, D = orig.shape
            ablated = sae.ablate_reconstruct(orig.reshape(-1, D).float(), ab_ids).reshape(B, T, D).half()
            return (ablated,) + o[1:] if isinstance(o, tuple) else ablated
        return intervene

    hk = model.transformer[layer].register_forward_hook(make_int(top50))
    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
        ab_out = model(s1_ids, s2_ids)
    hk.remove()
    ab_s1 = ab_out[0].float()
    ab_preds = ab_s1[:, -1, :].argmax(dim=-1).float().cpu().numpy()

    # Compute financial metrics
    # Use close price changes as ground truth for direction and rank
    # Actual next-period return from the raw data
    actual_returns = []
    for i in range(test_start, test_start+n_test):
        if i + 1 < len(data):
            close_cur = data[i+lb-1, 1]
            close_next = data[i+lb, 1]
            actual_returns.append((close_next - close_cur) / (close_cur + 1e-5))
    actual_returns = np.array(actual_returns[:n_test])

    # Use token ID as a proxy for predicted return direction
    # (higher token ≈ higher predicted price)
    base_rank = stats.rankdata(base_preds)
    actual_rank = stats.rankdata(actual_returns)

    rankic_base = compute_rankic(base_preds, actual_returns)
    rankic_ablated = compute_rankic(ab_preds, actual_returns)
    dir_base = compute_directional_accuracy(base_preds - np.median(base_preds), actual_returns)
    dir_ablated = compute_directional_accuracy(ab_preds - np.median(ab_preds), actual_returns)

    del sae; torch.cuda.empty_cache()

    return {
        "rankic_base": float(rankic_base),
        "rankic_ablated": float(rankic_ablated),
        "rankic_drop": float(rankic_base - rankic_ablated),
        "dir_base": float(dir_base),
        "dir_ablated": float(dir_ablated),
        "dir_drop": float(dir_base - dir_ablated),
        "n_test_windows": n_test,
        "time": time.time() - t0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/houwanlong/finllm-mi/data/scale120")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/financial_metrics.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-stocks", type=int, default=111)
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

    csv_files = sorted(Path(args.data_dir).glob("*.csv"))[:args.max_stocks]
    print(f"Processing {len(csv_files)} stocks...")

    results = []
    for i, f in enumerate(csv_files):
        if (i+1) % 20 == 0 or i == 0:
            print(f"[{i+1}/{len(csv_files)}]...")
        r = process_one(f, tokenizer, model, args.layer, device, args)
        if r:
            r["ticker"] = f.stem
            results.append(r)

    n = len(results)
    print(f"\nProcessed: {n} stocks")

    if n == 0:
        print("No results!")
        return

    # Aggregate
    rankic_drops = [r["rankic_drop"] for r in results]
    dir_drops = [r["dir_drop"] for r in results]
    rankic_bases = [r["rankic_base"] for r in results]
    dir_bases = [r["dir_base"] for r in results]

    def bootstrap_ci(vals):
        means = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(10000)]
        return np.mean(vals), np.percentile(means, 2.5), np.percentile(means, 97.5)

    print(f"\nFinancial Metrics (n={n} stocks):")
    print(f"{'Metric':<30} {'Baseline':>10} {'Ablated':>10} {'Drop':>10} {'95% CI':>25} {'Sig':>6}")
    print("-" * 95)

    for name, base_vals, drop_vals in [
        ("RankIC", rankic_bases, rankic_drops),
        ("Directional Accuracy", dir_bases, dir_drops),
    ]:
        b_mean = np.mean(base_vals)
        d_mean, d_lo, d_hi = bootstrap_ci(drop_vals)
        t_stat, p_val = stats.ttest_1samp(drop_vals, 0)
        sig = "SIG" if p_val < 0.05 else "ns"
        print(f"{name:<30} {b_mean:>10.4f} {'—':>10} {d_mean:>+10.4f} [{d_lo:+.4f}, {d_hi:+.4f}] {sig:>6}")
        print(f"  p={p_val:.4f}, n_stocks_with_positive_drop={sum(1 for d in drop_vals if d > 0)}/{len(drop_vals)}")

    final = {
        "n_stocks": n,
        "layer": args.layer,
        "rankic": {
            "baseline_mean": float(np.mean(rankic_bases)),
            "drop_mean": float(np.mean(rankic_drops)),
            "drop_ci_lo": float(d_lo), "drop_ci_hi": float(d_hi),
            "p_value": float(p_val),
            "positive_drop_count": int(sum(1 for d in rankic_drops if d > 0)),
        },
        "directional": {
            "baseline_mean": float(np.mean(dir_bases)),
            "drop_mean": float(np.mean(dir_drops)),
            "drop_ci_lo": float(d_lo), "drop_ci_hi": float(d_hi),
            "p_value": float(p_val),
            "positive_drop_count": int(sum(1 for d in dir_drops if d > 0)),
        },
        "per_stock": results,
    }
    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved. {time.time()-t_total:.0f}s")

if __name__ == "__main__":
    main()
