"""Financially meaningful SAE ablation metrics v2.

Three metrics:
  a) Directional accuracy: fraction of correct up/down predictions (baseline vs ablated)
  b) Return sign agreement: fraction of test windows where ablation flips the sign of predicted return
  c) Volatility shift: change in std of predicted returns (baseline vs ablated)

Key difference from v1: properly decodes s1+s2 logits back to K-line prices via the tokenizer decoder,
so predicted returns are measured in price space rather than token-ID space.
"""
import torch
import numpy as np
import json
import sys
import time
import os
from pathlib import Path
import pandas as pd
from scipy import stats

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file


class TopKSAE(torch.nn.Module):
    """Top-K Sparse Autoencoder (k=64, expansion=2x)."""
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


def compute_directional_accuracy(predicted_returns, actual_returns):
    """Fraction of predictions with correct sign."""
    valid = (np.abs(predicted_returns) > 0) & (np.abs(actual_returns) > 0)
    if valid.sum() < 3:
        return 0.5
    correct = (np.sign(predicted_returns[valid]) == np.sign(actual_returns[valid])).sum()
    return correct / valid.sum()


def compute_return_sign_agreement(base_returns, ablated_returns):
    """Fraction of test windows where the predicted return sign does NOT change (agreement)."""
    valid = (np.abs(base_returns) > 1e-8) & (np.abs(ablated_returns) > 1e-8)
    if valid.sum() < 3:
        return 1.0, 0.0
    agree = (np.sign(base_returns[valid]) == np.sign(ablated_returns[valid]))
    flip_rate = 1.0 - agree.mean()
    return agree.mean(), flip_rate


def compute_volatility_shift(base_returns, ablated_returns):
    """Change in standard deviation of predicted returns."""
    base_vol = np.std(base_returns)
    ablated_vol = np.std(ablated_returns)
    if base_vol < 1e-8:
        return 0.0, 0.0, 1.0
    ratio = ablated_vol / base_vol
    return base_vol, ablated_vol, ratio


def decode_returns(s1_logits, s2_logits, tokenizer, device):
    """
    Decode model output (s1+s2 logits) back to predicted K-line prices and compute returns.

    s1_logits: (B, T, vocab_s1)  - the model's predicted next-token logits
    s2_logits: (B, T, vocab_s2)

    Returns:
        predicted_returns: (B,)  - predicted close-to-close return for the forecast step
    """
    B, T, _ = s1_logits.shape
    # Use last position prediction
    s1_pred = s1_logits[:, -1, :].argmax(dim=-1)  # (B,)
    s2_pred = s2_logits[:, -1, :].argmax(dim=-1)  # (B,)

    # Build composite full-sequence input: append predicted token to actual sequence
    # But we need actual input tokens to decode. Let's decode the predicted token
    # into K-line space directly using the tokenizer's decode method.
    # The decode method expects BSQ indices. The Kronos model's s1/s2 IDs range
    # [0, 2^s1_bits-1] and map to the same index space as the tokenizer.

    # Decode single predicted token: reshape to (B, 1) as long and decode
    s1_pred_expanded = s1_pred.unsqueeze(1).long()  # (B, 1) as long for bitwise ops
    s2_pred_expanded = s2_pred.unsqueeze(1).long()

    with torch.no_grad():
        # The tokenizer.decode expects indices in the format returned by encode(half=True)
        # Pass as list [s1, s2] with half=True
        reconstructed = tokenizer.decode([s1_pred_expanded, s2_pred_expanded], half=True)
        # reconstructed shape: (B, 1, 6)  [open, close, high, low, volume, amount]

    pred_close = reconstructed[:, 0, 1].cpu().float().numpy()  # close price at index 1

    # The predicted close is in the normalized space of the tokenizer's training distribution,
    # not in the original data space. We compute returns as relative changes:
    # predicted_return ≈ (pred_close - 0) / 1  since data is normalized ~N(0,1)
    # But more properly, we use the tokenizer's encoded space: the decode output is the
    # reconstruction of what was encoded. The original data goes through:
    #   data -> embed -> encoder -> BSQ -> decoder -> reconstruction
    # The reconstruction ≈ original normalized data.
    # So pred_close is approximately the predicted normalized close price.

    # For return prediction: we compare consecutive normalized close values.
    # Since we only have 1 predicted step, we compute:
    #   predicted_return = pred_close (as a z-score, the normalized close for the next step)
    # But for direction, we need the actual last close from the input window.

    return pred_close


def process_one(csv_path, tokenizer, model, layer, device):
    """Train SAE + evaluate ablation impact on financial metrics for one stock."""
    t0 = time.time()
    df = pd.read_csv(str(csv_path))
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        if col not in df.columns:
            df[col] = 0.0
    data_orig = df[["open", "close", "high", "low", "volume", "amount"]].values.astype(np.float32)
    data_orig = data_orig[~np.isnan(data_orig).any(axis=1)]
    if len(data_orig) < 200:
        print(f"  {csv_path.stem}: too few rows ({len(data_orig)}), skipping")
        return None

    # Normalize data for tokenizer (same as tokenizer expects)
    mn = data_orig.mean(0)
    st_d = data_orig.std(0)
    data_norm = np.clip((data_orig - mn) / (st_d + 1e-5), -5, 5)

    # Create sliding windows
    lb, stride = 64, 32
    nw = min(2000, (len(data_norm) - lb) // stride)
    if nw < 20:
        print(f"  {csv_path.stem}: only {nw} windows, skipping")
        return None
    windows = np.stack([data_norm[i:i + lb] for i in range(0, nw * stride, stride)])

    n_train = int(len(windows) * 0.8)
    if n_train < 10:
        return None

    # ---- Phase 1: Extract layer-6 activations from training windows ----
    acts_train = []

    def make_hook(storage):
        def h(m, i, o):
            a = o[0] if isinstance(o, tuple) else o
            storage.append(a[:, -1, :].detach().cpu().float().numpy())
        return h

    hook = model.transformer[layer].register_forward_hook(make_hook(acts_train))
    bs = 64
    with torch.no_grad():
        for b in range(0, n_train, bs):
            batch = torch.from_numpy(windows[b:b + bs]).float().to(device)
            s1, s2 = tokenizer.encode(batch, half=True)
            model(s1, s2)
    hook.remove()
    acts = np.concatenate(acts_train, axis=0)
    print(f"  {csv_path.stem}: extracted {acts.shape[0]} acts, d={acts.shape[1]}")

    # ---- Phase 2: Train SAE (k=64, expansion=2x, 3000 steps) ----
    d_model = acts.shape[1]
    d_hidden = d_model * 2  # expansion=2x
    sae = TopKSAE(d_model, d_hidden, k=64).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
    at = torch.from_numpy(acts).float().to(device)
    n_steps = 3000
    for step in range(n_steps):
        idx = torch.randint(0, len(at), (512,))
        xr, _ = sae(at[idx])
        loss = torch.nn.functional.mse_loss(xr, at[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"    SAE step {step}/{n_steps}, loss={loss.item():.6f}")

    # ---- Phase 3: Select top-50 most active features ----
    with torch.no_grad():
        _, all_lat = sae(at[:min(1000, len(at))])
    freq = (all_lat != 0).float().sum(0)
    top50 = freq.argsort(descending=True)[:50].tolist()

    # ---- Phase 4: Compute financial metrics on test windows ----
    test_start = n_train
    n_test = min(30, len(windows) - test_start)
    if n_test < 5:
        del sae
        torch.cuda.empty_cache()
        return None

    test_wins = windows[test_start:test_start + n_test]
    test_t = torch.from_numpy(test_wins).float().to(device)

    # Baseline forward pass
    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
        base_s1, base_s2 = model(s1_ids, s2_ids)
    base_s1 = base_s1.float()  # (B, T, 1024)
    base_s2 = base_s2.float()  # (B, T, 1024)

    # Intervene on layer-6 with SAE ablation
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
        ab_s1, ab_s2 = model(s1_ids, s2_ids)
    hk.remove()
    ab_s1 = ab_s1.float()
    ab_s2 = ab_s2.float()

    del sae
    torch.cuda.empty_cache()

    # ---- Phase 5: Decode logits to price predictions ----
    base_close = decode_returns(base_s1, base_s2, tokenizer, device)
    ab_close = decode_returns(ab_s1, ab_s2, tokenizer, device)

    # Actual next-period close prices from raw data
    actual_prices = []
    for i in range(test_start, test_start + n_test):
        idx_end = (i * stride) + lb
        if idx_end < len(data_norm):
            actual_prices.append(data_norm[idx_end, 1])  # normalized close
        else:
            actual_prices.append(data_norm[-1, 1])
    actual_prices = np.array(actual_prices)

    # The model predicts the NEXT close price.
    # Since data is normalized, direct comparison works.
    # The predicted close from decode is in normalized space.

    # Get last known close for computing actual returns
    last_known_closes = []
    for i in range(test_start, test_start + n_test):
        idx_last = (i * stride) + lb - 1
        if idx_last < len(data_norm):
            last_known_closes.append(data_norm[idx_last, 1])
        else:
            last_known_closes.append(data_norm[-2, 1])
    last_known_closes = np.array(last_known_closes)

    # Compute actual returns (next close - last known close)
    actual_returns = actual_prices - last_known_closes
    base_returns = base_close - last_known_closes
    ablated_returns = ab_close - last_known_closes

    # ---- Metrics ----
    dir_acc_base = compute_directional_accuracy(base_returns, actual_returns)
    dir_acc_ablated = compute_directional_accuracy(ablated_returns, actual_returns)

    sign_agree, sign_flip_rate = compute_return_sign_agreement(base_returns, ablated_returns)

    base_vol, ablated_vol, vol_ratio = compute_volatility_shift(base_returns, ablated_returns)

    elapsed = time.time() - t0
    print(f"  {csv_path.stem}: dir_acc(base={dir_acc_base:.3f}, ab={dir_acc_ablated:.3f}), "
          f"flip_rate={sign_flip_rate:.3f}, vol_ratio={vol_ratio:.3f} [{elapsed:.1f}s]")

    return {
        "ticker": csv_path.stem,
        "dir_acc_base": float(dir_acc_base),
        "dir_acc_ablated": float(dir_acc_ablated),
        "dir_acc_drop": float(dir_acc_base - dir_acc_ablated),
        "sign_agreement": float(sign_agree),
        "sign_flip_rate": float(sign_flip_rate),
        "volatility_base": float(base_vol),
        "volatility_ablated": float(ablated_vol),
        "volatility_ratio": float(vol_ratio),
        "volatility_shift": float(ablated_vol - base_vol),
        "n_test_windows": n_test,
        "top50_features": top50,
        "time": float(elapsed),
    }


def bootstrap_ci(vals, n_boot=10000):
    means = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    return np.mean(vals), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/houwanlong/finllm-mi/data/scale120")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/fin_metrics_v2.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--max-stocks", type=int, default=30)
    parser.add_argument("--start-idx", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    t_total = time.time()

    # Load Kronos Tokenizer
    print("Loading Kronos Tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(
        "/data/houwanlong/models/Kronos-Tokenizer-base"
    ).to(device).eval()

    # Load Kronos Model
    print("Loading Kronos Model...")
    with open("/data/houwanlong/models/Kronos-base/config.json") as f:
        cfg = json.load(f)
    model = Kronos(
        s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"],
        n_layers=cfg["n_layers"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
        ffn_dropout_p=cfg["ffn_dropout_p"],
        attn_dropout_p=cfg["attn_dropout_p"],
        resid_dropout_p=cfg["resid_dropout_p"],
        token_dropout_p=cfg["token_dropout_p"],
        learn_te=cfg["learn_te"],
    )
    sd = load_file("/data/houwanlong/models/Kronos-base/model.safetensors")
    model.load_state_dict(sd, strict=False)
    model = model.to(device).half().eval()
    print(f"  Model: {cfg['n_layers']} layers, d_model={cfg['d_model']}")

    # Get stock files
    csv_files = sorted(Path(args.data_dir).glob("*.csv"))
    if args.start_idx > 0:
        csv_files = csv_files[args.start_idx:]
    csv_files = csv_files[:args.max_stocks]
    print(f"Processing {len(csv_files)} stocks (idx {args.start_idx}-{args.start_idx + len(csv_files) - 1})...")
    print()

    results = []
    for i, f in enumerate(csv_files):
        print(f"[{i + 1}/{len(csv_files)}] {f.stem}")
        r = process_one(f, tokenizer, model, args.layer, device)
        if r:
            results.append(r)
        else:
            print(f"  SKIPPED (insufficient data)")

    n = len(results)
    print(f"\n{'=' * 70}")
    print(f"Processed: {n} stocks successfully")
    print(f"{'=' * 70}")

    if n == 0:
        print("No results!")
        return

    # Aggregate metrics
    dir_drops = [r["dir_acc_drop"] for r in results]
    flip_rates = [r["sign_flip_rate"] for r in results]
    vol_ratios = [r["volatility_ratio"] for r in results]
    vol_shifts = [r["volatility_shift"] for r in results]
    dir_bases = [r["dir_acc_base"] for r in results]
    dir_ablated_list = [r["dir_acc_ablated"] for r in results]

    print("\n" + "=" * 80)
    print("FINANCIAL METRICS V2 SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<35} {'Baseline':>10} {'Ablated':>10} {'Delta':>10} {'95% CI':>25} {'Sig':>6}")
    print("-" * 90)

    # 1. Directional Accuracy
    b_mean = np.mean(dir_bases)
    a_mean = np.mean(dir_ablated_list)
    d_mean, d_lo, d_hi = bootstrap_ci(dir_drops)
    t_stat, p_val = stats.ttest_1samp(dir_drops, 0)
    sig = "SIG" if p_val < 0.05 else "ns"
    print(f"{'Directional Accuracy':<35} {b_mean:>10.4f} {a_mean:>10.4f} "
          f"{d_mean:>+10.4f} [{d_lo:+.4f}, {d_hi:+.4f}] {sig:>6}")
    print(f"  p={p_val:.4f}, positive_drop={sum(1 for d in dir_drops if d > 0)}/{n}")

    # 2. Return Sign Flip Rate
    flip_mean, flip_lo, flip_hi = bootstrap_ci(flip_rates)
    print(f"{'Sign Flip Rate (ablation flips sign)':<35} {'—':>10} {'—':>10} "
          f"{flip_mean:>10.4f} [{flip_lo:.4f}, {flip_hi:.4f}] {'—':>6}")

    # 3. Volatility Ratio
    vr_mean, vr_lo, vr_hi = bootstrap_ci(vol_ratios)
    # Test if vol_ratio != 1 (stability of volatility)
    t_stat_vr, p_val_vr = stats.ttest_1samp(vol_ratios, 1.0)
    sig_vr = "SIG" if p_val_vr < 0.05 else "ns"
    print(f"{'Volatility Ratio (abl/baseline)':<35} {'—':>10} {'—':>10} "
          f"{vr_mean:>10.4f} [{vr_lo:.4f}, {vr_hi:.4f}] {sig_vr:>6}")
    print(f"  p={p_val_vr:.4f} (H0: ratio=1), mean_shift={np.mean(vol_shifts):+.6f}")

    # Save
    final = {
        "n_stocks": n,
        "layer": args.layer,
        "config": {
            "k": 64,
            "expansion": "2x",
            "steps": 3000,
            "top_ablated": 50,
            "max_stocks": args.max_stocks,
            "start_idx": args.start_idx,
        },
        "directional_accuracy": {
            "baseline_mean": float(b_mean),
            "ablated_mean": float(a_mean),
            "drop_mean": float(d_mean),
            "drop_ci_lo": float(d_lo),
            "drop_ci_hi": float(d_hi),
            "p_value": float(p_val),
            "positive_drop_count": int(sum(1 for d in dir_drops if d > 0)),
        },
        "sign_flip": {
            "mean_flip_rate": float(flip_mean),
            "flip_rate_ci_lo": float(flip_lo),
            "flip_rate_ci_hi": float(flip_hi),
        },
        "volatility": {
            "mean_ratio": float(vr_mean),
            "ratio_ci_lo": float(vr_lo),
            "ratio_ci_hi": float(vr_hi),
            "p_value_vs_1": float(p_val_vr),
            "mean_shift": float(np.mean(vol_shifts)),
        },
        "per_stock": results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {args.output}")
    print(f"Total time: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
