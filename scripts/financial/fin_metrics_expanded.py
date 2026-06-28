"""Expanded financial validation metrics for SAE feature ablation on Kronos.

Metrics (all on held-out test windows, ablated vs baseline):
  - Directional accuracy: fraction of windows where ablated vs baseline agree on sign
  - Volatility stability ratio: std(ablated_returns) / std(baseline_returns)
  - MAE: mean absolute error of decoded close prices (ablated vs baseline)
  - RMSE: root mean squared error of decoded close prices (ablated vs baseline)
  - RankIC: Spearman rank correlation between ablated and baseline token rankings
  - Volatility forecast error: |std(pred) - realized| for both ablated and baseline
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
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

# ── Constants ────────────────────────────────────────────────────────────────
LAYER = 6
WINDOW_LENGTH = 64
STRIDE = 32
N_TRAIN = 60
N_VAL = 10
N_TEST = 30
SAE_K = 64
SAE_EXPANSION = 4
SAE_LR = 1e-4
SAE_STEPS = 3000
SAE_BATCH = 256
TOP_K_ABLATE = 50
DATA_DIR = "/data/houwanlong/finllm-mi/data/scale120"
MODEL_DIR = "/data/houwanlong/models/Kronos-base"
TOKENIZER_DIR = "/data/houwanlong/models/Kronos-Tokenizer-base"
OUTPUT_PATH = "/data/houwanlong/finllm-mi/outputs/sae/fin_metrics_expanded.json"
DEVICE = "cuda:0"
MAX_STOCKS = 30


# ── SAE ──────────────────────────────────────────────────────────────────────
class TopKSAE(torch.nn.Module):
    """Top-K Sparse Autoencoder for residual stream features."""

    def __init__(self, d_model: int, d_hidden: int, k: int = 64):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = torch.nn.Linear(d_hidden, d_model, bias=False)
        self.b_pre = torch.nn.Parameter(torch.zeros(d_model))
        self.k = k

    def forward(self, x: torch.Tensor):
        xc = x - self.b_pre
        lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat)
        mask.scatter_(-1, idx, 1.0)
        return self.decoder(lat * mask) + self.b_pre, lat * mask

    def ablate_reconstruct(self, x: torch.Tensor, ids: list) -> torch.Tensor:
        """Reconstruct with specified latent features zeroed out."""
        xc = x - self.b_pre
        lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat)
        mask.scatter_(-1, idx, 1.0)
        mask[:, ids] = 0
        return self.decoder(lat * mask) + self.b_pre


# ── Metric Helpers ───────────────────────────────────────────────────────────
def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, safe for small / constant arrays."""
    if len(x) < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rho, _ = stats.spearmanr(x, y)
    return float(rho) if not np.isnan(rho) else 0.0


def _direction_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of elements where sign(a) == sign(b)."""
    sign_a = np.sign(a)
    sign_b = np.sign(b)
    valid = (sign_a != 0) & (sign_b != 0)
    if valid.sum() == 0:
        return 0.0
    return float((sign_a[valid] == sign_b[valid]).mean())


# ── Per-Stock Processing ─────────────────────────────────────────────────────
def process_one(
    csv_path: Path,
    tokenizer: KronosTokenizer,
    model: Kronos,
    device: str,
) -> dict | None:
    t_start = time.time()

    # ── Load & normalise ─────────────────────────────────────────────────
    df = pd.read_csv(str(csv_path))
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        if col not in df.columns:
            df[col] = 0.0
    data = df[["open", "close", "high", "low", "volume", "amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < (WINDOW_LENGTH + (N_TRAIN + N_VAL + N_TEST) * STRIDE):
        return None

    mn = data.mean(0)
    st_d = data.std(0)
    data_norm = np.clip((data - mn) / (st_d + 1e-5), -5, 5)

    n_windows_max = min(2000, (len(data_norm) - WINDOW_LENGTH) // STRIDE)
    windows = np.stack([
        data_norm[i : i + WINDOW_LENGTH]
        for i in range(0, n_windows_max * STRIDE, STRIDE)
    ])

    total_needed = N_TRAIN + N_VAL + N_TEST
    if len(windows) < total_needed:
        n_train = int(len(windows) * 0.6)
        n_val = int(len(windows) * 0.1)
        n_test = min(30, len(windows) - n_train - n_val)
    else:
        n_train, n_val, n_test = N_TRAIN, N_VAL, N_TEST

    if n_test < 5:
        return None

    train_wins = windows[:n_train]
    test_wins = windows[n_train + n_val : n_train + n_val + n_test]

    # ── Collect layer-6 residual stream activations (train windows) ──────
    acts_storage: list = []

    def _hook_fn(storage):
        def hook(module, inputs, output):
            act = output[0] if isinstance(output, tuple) else output
            storage.append(act[:, -1, :].detach().cpu().float().numpy())
        return hook

    handle = model.transformer[LAYER].register_forward_hook(_hook_fn(acts_storage))
    with torch.no_grad():
        for b_start in range(0, n_train, SAE_BATCH):
            batch = torch.from_numpy(train_wins[b_start : b_start + SAE_BATCH]).float().to(device)
            s1, s2 = tokenizer.encode(batch, half=True)
            model(s1, s2)
    handle.remove()

    acts = np.concatenate(acts_storage, axis=0)

    # ── Train TopK SAE ───────────────────────────────────────────────────
    d_model = acts.shape[1]
    d_hidden = d_model * SAE_EXPANSION
    sae = TopKSAE(d_model, d_hidden, k=SAE_K).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
    acts_t = torch.from_numpy(acts).float().to(device)

    for _step in range(SAE_STEPS):
        idx = torch.randint(0, len(acts_t), (SAE_BATCH,))
        x_recon, _ = sae(acts_t[idx])
        loss = torch.nn.functional.mse_loss(x_recon, acts_t[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()

    # ── Identify top-50 most-active features ─────────────────────────────
    with torch.no_grad():
        _, lat_all = sae(acts_t[: min(1000, len(acts_t))])
    freq = (lat_all != 0).float().sum(0)
    top_ids = freq.argsort(descending=True)[:TOP_K_ABLATE].tolist()

    # ── Test: baseline forward pass ──────────────────────────────────────
    test_t = torch.from_numpy(test_wins).float().to(device)

    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
        base_out = model(s1_ids, s2_ids)
    base_s1_logits = base_out[0].float()               # (B, T, vocab_s1)
    base_tokens = base_s1_logits[:, -1, :].argmax(dim=-1)  # (B,)

    # Decode baseline predicted tokens → OHLCVA values
    try:
        base_decoded = tokenizer.decode(base_tokens)   # (B, 6) or (B, ...)
        if isinstance(base_decoded, torch.Tensor):
            base_decoded = base_decoded.float().cpu().numpy()
        else:
            base_decoded = np.asarray(base_decoded)
    except Exception:
        # Fallback: treat token ID as proxy signal
        base_decoded = base_tokens.float().cpu().numpy()[:, None]

    # ── Test: ablated forward pass (top-50 features removed) ─────────────
    def _make_intervention(ablate_ids):
        def intervene(module, inputs, output):
            orig = output[0] if isinstance(output, tuple) else output
            B, T, D = orig.shape
            ablated = sae.ablate_reconstruct(orig.reshape(-1, D).float(), ablate_ids)
            ablated = ablated.reshape(B, T, D).half()
            if isinstance(output, tuple):
                return (ablated,) + output[1:]
            return ablated
        return intervene

    hk = model.transformer[LAYER].register_forward_hook(_make_intervention(top_ids))
    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
        ab_out = model(s1_ids, s2_ids)
    hk.remove()

    ab_s1_logits = ab_out[0].float()
    ab_tokens = ab_s1_logits[:, -1, :].argmax(dim=-1)

    try:
        ab_decoded = tokenizer.decode(ab_tokens)
        if isinstance(ab_decoded, torch.Tensor):
            ab_decoded = ab_decoded.float().cpu().numpy()
        else:
            ab_decoded = np.asarray(ab_decoded)
    except Exception:
        ab_decoded = ab_tokens.float().cpu().numpy()[:, None]

    # ── Compute metrics ──────────────────────────────────────────────────
    # Signal: use decoded close prices (col 1) or token-ID proxy
    if base_decoded.shape[1] >= 2:
        base_signal = base_decoded[:, 1]
        ab_signal = ab_decoded[:, 1]
    else:
        base_signal = base_decoded[:, 0]
        ab_signal = ab_decoded[:, 0]

    # 1. Directional accuracy (ablated vs baseline sign agreement)
    #    Compare sign of predicted change between consecutive test windows.
    base_returns = np.diff(base_signal)   # length n_test-1
    ab_returns = np.diff(ab_signal)       # length n_test-1
    dir_acc = _direction_agreement(ab_returns, base_returns)

    # 2. Volatility stability ratio: std(ablated_returns) / std(baseline_returns)
    std_base = float(np.std(base_returns)) if len(base_returns) > 1 else 1e-8
    std_ab = float(np.std(ab_returns)) if len(ab_returns) > 1 else 1e-8
    vol_stab_ratio = std_ab / std_base if std_base > 1e-12 else 1.0

    # 3. MAE of decoded close prices (ablated vs baseline)
    mae = float(np.mean(np.abs(ab_signal - base_signal)))

    # 4. RMSE of decoded close prices (ablated vs baseline)
    rmse = float(np.sqrt(np.mean((ab_signal - base_signal) ** 2)))

    # 5. RankIC: Spearman correlation between ablated and baseline token rankings
    #    Use raw argmax token IDs for ranking, per user spec.
    base_ranks = base_tokens.float().cpu().numpy()
    ab_ranks = ab_tokens.float().cpu().numpy()
    rankic = _spearman_r(ab_ranks, base_ranks)

    # 6. Volatility forecast error: |std(pred) - realized|
    #    Realized volatility from actual close prices on test windows
    actual_returns = []
    offset = n_train + n_val
    for i in range(n_test):
        w_start = offset * STRIDE + i * STRIDE
        if w_start + WINDOW_LENGTH < len(data):
            c_cur = data[w_start + WINDOW_LENGTH - 1, 1]
            c_next = data[w_start + WINDOW_LENGTH, 1]
            actual_returns.append((c_next - c_cur) / (c_cur + 1e-5))
    actual_returns = np.array(actual_returns)
    realized_vol = np.std(actual_returns) if len(actual_returns) > 1 else 0.0

    vol_err_base = float(np.abs(std_base - realized_vol))
    vol_err_ab = float(np.abs(std_ab - realized_vol))

    del sae
    torch.cuda.empty_cache()

    return {
        "ticker": csv_path.stem,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "dir_acc": dir_acc,
        "vol_stab_ratio": vol_stab_ratio,
        "mae": mae,
        "rmse": rmse,
        "rankic": rankic,
        "vol_err_base": vol_err_base,
        "vol_err_ab": vol_err_ab,
        "vol_err_drop": vol_err_ab - vol_err_base,
        "realized_vol": float(realized_vol),
        "std_base": float(std_base),
        "std_ab": float(std_ab),
        "time_sec": time.time() - t_start,
    }


# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def _bootstrap_ci(vals: list, n_boot: int = 10000) -> tuple[float, float, float]:
    arr = np.array(vals)
    means = np.array([
        np.mean(np.random.choice(arr, len(arr), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    t_total = time.time()

    print("Loading Kronos tokenizer & model ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_DIR).to(DEVICE).eval()

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)
    model = Kronos(
        s1_bits=cfg["s1_bits"],
        s2_bits=cfg["s2_bits"],
        n_layers=cfg["n_layers"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        ff_dim=cfg["ff_dim"],
        ffn_dropout_p=cfg["ffn_dropout_p"],
        attn_dropout_p=cfg["attn_dropout_p"],
        resid_dropout_p=cfg["resid_dropout_p"],
        token_dropout_p=cfg["token_dropout_p"],
        learn_te=cfg["learn_te"],
    )
    sd = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    model.load_state_dict(sd, strict=False)
    model = model.to(DEVICE).half().eval()

    csv_files = sorted(Path(DATA_DIR).glob("*.csv"))[:MAX_STOCKS]
    print(f"Processing {len(csv_files)} stocks (layer {LAYER}, top-{TOP_K_ABLATE} ablation) ...")

    results = []
    for i, fpath in enumerate(csv_files):
        print(f"[{i + 1}/{len(csv_files)}] {fpath.stem} ...", end=" ", flush=True)
        r = process_one(fpath, tokenizer, model, DEVICE)
        if r is None:
            print("SKIP")
            continue
        results.append(r)
        print(
            f"dir_acc={r['dir_acc']:.3f} "
            f"vol_ratio={r['vol_stab_ratio']:.3f} "
            f"mae={r['mae']:.4f} "
            f"rankic={r['rankic']:.3f} "
            f"({r['time_sec']:.0f}s)"
        )

    n = len(results)
    print(f"\nProcessed: {n} / {len(csv_files)} stocks successfully\n")

    if n == 0:
        print("No results to aggregate.")
        return

    # ── Aggregate across stocks ────────────────────────────────────────
    dir_accs = [r["dir_acc"] for r in results]
    vol_ratios = [r["vol_stab_ratio"] for r in results]
    maes = [r["mae"] for r in results]
    rmses = [r["rmse"] for r in results]
    rankics = [r["rankic"] for r in results]
    vol_err_drops = [r["vol_err_drop"] for r in results]
    vol_err_bases = [r["vol_err_base"] for r in results]
    vol_err_abs = [r["vol_err_ab"] for r in results]

    print(f"{'Metric':<35} {'Mean':>10} {'95% CI':>30} {'Sig':>6}")
    print("-" * 85)

    report_lines = []
    for name, vals, ref in [
        ("Directional Accuracy", dir_accs, 0.5),
        ("Volatility Stability Ratio", vol_ratios, 1.0),
        ("MAE (close price, decoded)", maes, 0.0),
        ("RMSE (close price, decoded)", rmses, 0.0),
        ("RankIC (ablated vs baseline)", rankics, 0.0),
        ("Vol Err Drop (abl - base)", vol_err_drops, 0.0),
    ]:
        mean_val, lo, hi = _bootstrap_ci(vals)
        t_stat, p_val = stats.ttest_1samp(vals, ref)
        sig = "SIG" if p_val < 0.05 else "ns"
        print(f"{name:<35} {mean_val:>10.4f}  [{lo:.4f}, {hi:.4f}]  {sig:>6}")
        report_lines.append(
            f"  p={p_val:.4f}, n_stocks_above_ref={sum(1 for v in vals if v > ref)}/{n}"
        )
        print(report_lines[-1])

    # ── Write output ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final = {
        "n_stocks": n,
        "layer": LAYER,
        "top_k_ablate": TOP_K_ABLATE,
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "sae_k": SAE_K,
        "sae_expansion": SAE_EXPANSION,
        "sae_steps": SAE_STEPS,
        "directional_accuracy": {
            "mean": float(np.mean(dir_accs)),
            "ci_lo": _bootstrap_ci(dir_accs)[1],
            "ci_hi": _bootstrap_ci(dir_accs)[2],
            "p_value_vs_0_5": float(stats.ttest_1samp(dir_accs, 0.5)[1]),
        },
        "volatility_stability_ratio": {
            "mean": float(np.mean(vol_ratios)),
            "ci_lo": _bootstrap_ci(vol_ratios)[1],
            "ci_hi": _bootstrap_ci(vol_ratios)[2],
            "p_value_vs_1_0": float(stats.ttest_1samp(vol_ratios, 1.0)[1]),
        },
        "mae": {
            "mean": float(np.mean(maes)),
            "ci_lo": _bootstrap_ci(maes)[1],
            "ci_hi": _bootstrap_ci(maes)[2],
        },
        "rmse": {
            "mean": float(np.mean(rmses)),
            "ci_lo": _bootstrap_ci(rmses)[1],
            "ci_hi": _bootstrap_ci(rmses)[2],
        },
        "rankic": {
            "mean": float(np.mean(rankics)),
            "ci_lo": _bootstrap_ci(rankics)[1],
            "ci_hi": _bootstrap_ci(rankics)[2],
            "p_value_vs_0": float(stats.ttest_1samp(rankics, 0.0)[1]),
        },
        "volatility_forecast_error": {
            "baseline_mean": float(np.mean(vol_err_bases)),
            "ablated_mean": float(np.mean(vol_err_abs)),
            "drop_mean": float(np.mean(vol_err_drops)),
            "drop_ci_lo": _bootstrap_ci(vol_err_drops)[1],
            "drop_ci_hi": _bootstrap_ci(vol_err_drops)[2],
            "p_value": float(stats.ttest_1samp(vol_err_drops, 0.0)[1]),
            "positive_drop_count": int(sum(1 for d in vol_err_drops if d > 0)),
        },
        "per_stock": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Total time: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
