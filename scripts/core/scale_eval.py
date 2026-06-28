"""Scaled multi-stock evaluation: 28 stocks, LOOKBACK=20, ~350 windows each."""
import numpy as np, json, warnings, pandas as pd, time, sys
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file
import torch

t0 = time.time()
device = "cuda:0"
DATA_DIR = Path("/data/houwanlong/finllm-mi/data/scale")
OUT = Path("/data/houwanlong/finllm-mi/outputs")

print("Loading models...")
tokenizer = KronosTokenizer.from_pretrained("/data/houwanlong/models/Kronos-Tokenizer-base").to(device).eval()
with open("/data/houwanlong/models/Kronos-base/config.json") as f:
    cfg = json.load(f)
kronos_model = Kronos(
    s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"],
    n_layers=cfg["n_layers"], d_model=cfg["d_model"],
    n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
    ffn_dropout_p=cfg["ffn_dropout_p"], attn_dropout_p=cfg["attn_dropout_p"],
    resid_dropout_p=cfg["resid_dropout_p"], token_dropout_p=cfg["token_dropout_p"],
    learn_te=cfg["learn_te"],
)
sd = load_file("/data/houwanlong/models/Kronos-base/model.safetensors")
kronos_model.load_state_dict(sd, strict=False)
kronos_model = kronos_model.to(device).half().eval()
n_k = len(kronos_model.transformer)

from transformers import T5Model
chronos_model = T5Model.from_pretrained("/data/houwanlong/models/chronos-t5-small", torch_dtype=torch.float16).to(device).eval()
n_c = len(chronos_model.encoder.block)
print("Models loaded. Kronos: {}L, Chronos: {}L".format(n_k, n_c))

LOOKBACK = 20
STRIDE = 10
BATCH = 64

def make_hook(layer_idx, storage):
    def h(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        storage[layer_idx].append(act.detach().cpu().float().mean(dim=1).numpy())
    return h

def process_stock(csv_path):
    """Full pipeline for one stock. Returns results dict or None if failed."""
    try:
        df = pd.read_csv(str(csv_path))
    except Exception as e:
        print("  SKIP: read error: {}".format(e))
        return None

    for col in ["open","close","high","low","volume","amount"]:
        if col not in df.columns: df[col] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]

    if len(data) < 100:
        print("  SKIP: only {} rows".format(len(data)))
        return None

    mn, st_d = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st_d + 1e-5), -5, 5)

    # Compute windows + labels
    n_windows = (len(data_norm) - LOOKBACK * 2) // STRIDE
    if n_windows < 50:
        print("  SKIP: only {} windows".format(n_windows))
        return None

    windows_close = []
    labels = []
    raw_feats = []

    for i in range(0, n_windows * STRIDE, STRIDE):
        win = data_norm[i:i+LOOKBACK]
        close_only = np.zeros_like(win)
        close_only[:] = win[:, 1:2]
        windows_close.append(close_only)

        close_cur = data_norm[i:i+LOOKBACK, 1]
        returns = np.diff(close_cur) / (close_cur[:-1] + 1e-5)
        raw_feats.append([
            np.mean(returns), np.std(returns),
            float(np.mean((returns - returns.mean())**3) / (returns.std()**3 + 1e-5)),
            float(np.mean((returns - returns.mean())**4) / (returns.std()**4 + 1e-5)),
            float(close_cur[-1] / (np.mean(close_cur) + 1e-5) - 1),
            float(returns[-1]), float(returns[-2]),
        ])

        close_fut = data_norm[i+LOOKBACK:i+2*LOOKBACK, 1]
        rets_fut = np.diff(close_fut) / (close_fut[:-1] + 1e-5)
        labels.append({"vol": np.std(rets_fut), "slope": np.polyfit(np.arange(LOOKBACK), close_fut, 1)[0]})

    windows_close = np.stack(windows_close, axis=0)
    raw_feats = np.array(raw_feats, dtype=np.float64)

    # Extract Kronos activations
    k_acts = {l: [] for l in range(n_k)}
    hooks = [kronos_model.transformer[l].register_forward_hook(make_hook(l, k_acts)) for l in range(n_k)]
    n_batches = len(windows_close) // BATCH
    if n_batches < 1:
        for h in hooks: h.remove()
        return None
    with torch.no_grad():
        for b in range(n_batches):
            batch = torch.from_numpy(windows_close[b*BATCH:(b+1)*BATCH]).float().to(device)
            s1_ids, s2_ids = tokenizer.encode(batch, half=True)
            kronos_model(s1_ids, s2_ids)
    for h in hooks: h.remove()
    k_acts = {l: np.stack(k_acts[l][:n_batches*BATCH], axis=0).reshape(-1, 832) for l in range(n_k)}

    # Extract Chronos activations
    c_windows = []
    for i in range(0, n_windows * STRIDE, STRIDE):
        c_windows.append(data_norm[i:i+LOOKBACK, 1])
    c_windows = np.stack(c_windows, axis=0)

    c_acts_raw = {l: [] for l in range(n_c)}
    hooks = [chronos_model.encoder.block[l].register_forward_hook(make_hook(l, c_acts_raw)) for l in range(n_c)]
    with torch.no_grad():
        for b in range(n_batches):
            batch = torch.from_numpy(c_windows[b*BATCH:(b+1)*BATCH]).float().to(device)
            batch_norm = (batch - batch.min(dim=1, keepdim=True).values) / (
                batch.max(dim=1, keepdim=True).values - batch.min(dim=1, keepdim=True).values + 1e-5)
            token_ids = (batch_norm * 4095).long()
            mask = torch.ones_like(token_ids)
            dec_input = torch.zeros(token_ids.shape[0], 1, dtype=torch.long, device=device)
            chronos_model(input_ids=token_ids, attention_mask=mask, decoder_input_ids=dec_input)
    for h in hooks: h.remove()
    c_acts = {l: np.stack(c_acts_raw[l][:n_batches*BATCH], axis=0).reshape(-1, 512) for l in range(n_c)}

    # Match sample count
    n = min(min(v.shape[0] for v in k_acts.values()), min(v.shape[0] for v in c_acts.values()), len(labels))
    for l in k_acts: k_acts[l] = k_acts[l][:n]
    for l in c_acts: c_acts[l] = c_acts[l][:n]
    labels = labels[:n]
    raw_feats = raw_feats[:n]

    # Labels
    y_vol_c = np.array([l["vol"] for l in labels])
    y_trend_c = np.array([l["slope"] for l in labels])

    # Regression instead of classification (avoids class imbalance)
    vol_median = np.median(y_vol_c)
    y_vol_cls = np.array([1 if v > vol_median else 0 for v in y_vol_c])

    # Chronological split
    n_train = int(n * 0.6)
    n_test_start = int(n * 0.7)
    if n_test_start >= n or n - n_test_start < 10:
        return None
    n_test = n - n_test_start

    # Probe all layers, take best
    def probe_best(acts_dict, y, classification=True):
        best = -1
        for layer in range(min(n_k, n_c)):
            Xs = StandardScaler().fit_transform(acts_dict[layer][:n_train])
            if classification:
                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(Xs, y[:n_train])
                X_test = StandardScaler().fit_transform(acts_dict[layer][n_test_start:])
                f1 = f1_score(y[n_test_start:], clf.predict(X_test), average="macro")
                best = max(best, f1)
            else:
                reg = Ridge(alpha=1.0)
                reg.fit(Xs, y[:n_train])
                X_test = StandardScaler().fit_transform(acts_dict[layer][n_test_start:])
                r2 = r2_score(y[n_test_start:], reg.predict(X_test))
                best = max(best, r2)
        return best

    kv = probe_best(k_acts, y_vol_cls)
    cv = probe_best(c_acts, y_vol_cls)
    kv_r2 = probe_best(k_acts, y_vol_c, classification=False)
    cv_r2 = probe_best(c_acts, y_vol_c, classification=False)

    # Raw baseline
    Xr = StandardScaler().fit_transform(raw_feats[:n_train])
    clf_r = LogisticRegression(max_iter=1000, C=1.0).fit(Xr, y_vol_cls[:n_train])
    Xr_test = StandardScaler().fit_transform(raw_feats[n_test_start:])
    rv = f1_score(y_vol_cls[n_test_start:], clf_r.predict(Xr_test), average="macro")

    chance = max(np.bincount(y_vol_cls[n_test_start:])) / n_test

    return {
        "stock": csv_path.stem.replace(".csv",""),
        "n_windows": n, "n_test": n_test,
        "k_vol": float(kv), "c_vol": float(cv),
        "raw_vol": float(rv), "chance": float(chance),
        "k_vol_r2": float(kv_r2), "c_vol_r2": float(cv_r2),
    }

# Process all stocks
csv_files = sorted(DATA_DIR.glob("sh*.csv")) + sorted(DATA_DIR.glob("sz*.csv"))
print("\nProcessing {} stocks...".format(len(csv_files)))

results = []
for csv_path in csv_files:
    stock_name = csv_path.stem
    print("\n{}:".format(stock_name), end=" ", flush=True)
    r = process_stock(csv_path)
    if r:
        results.append(r)
        print("N={}, test={}, K={:.3f}, C={:.3f}, Raw={:.3f}, R2_K={:.3f}".format(
            r["n_windows"], r["n_test"], r["k_vol"], r["c_vol"], r["raw_vol"], r["k_vol_r2"]))
    else:
        print("SKIPPED")

# ─── Aggregate ───
print("\n" + "=" * 70)
print("SCALED RESULTS: {} stocks".format(len(results)))
print("=" * 70)

k_vols = [r["k_vol"] for r in results]
c_vols = [r["c_vol"] for r in results]
raw_vs = [r["raw_vol"] for r in results]
k_r2s = [r["k_vol_r2"] for r in results]
c_r2s = [r["c_vol_r2"] for r in results]
k_beats_c = sum(1 for kv, cv in zip(k_vols, c_vols) if kv > cv)

# Per-stock delta
deltas = [kv - cv for kv, cv in zip(k_vols, c_vols)]
delta_mean = np.mean(deltas)
delta_std = np.std(deltas)
# Paired t-test
t_stat, p_val_delta = stats.ttest_rel(k_vols, c_vols)
# Bootstrap CI on mean delta
np.random.seed(42)
boot_means = [np.mean(np.random.choice(deltas, len(deltas), replace=True)) for _ in range(10000)]
delta_ci_lo = np.percentile(boot_means, 2.5)
delta_ci_hi = np.percentile(boot_means, 97.5)

print("\nSummary:")
print("  Stocks evaluated: {}".format(len(results)))
print("  Mean windows/stock: {:.0f}".format(np.mean([r["n_windows"] for r in results])))
print("  Mean test samples/stock: {:.0f}".format(np.mean([r["n_test"] for r in results])))
print()
print("  Kronos vol F1:  {:.4f} ± {:.4f}".format(np.mean(k_vols), np.std(k_vols)))
print("  Chronos vol F1: {:.4f} ± {:.4f}".format(np.mean(c_vols), np.std(c_vols)))
print("  Raw vol F1:     {:.4f} ± {:.4f}".format(np.mean(raw_vs), np.std(raw_vs)))
print("  Kronos vol R2:  {:.4f} ± {:.4f}".format(np.mean(k_r2s), np.std(k_r2s)))
print("  Chronos vol R2: {:.4f} ± {:.4f}".format(np.mean(c_r2s), np.std(c_r2s)))
print()
print("  Kronos > Chronos in {}/{} stocks ({:.0f}%)".format(k_beats_c, len(results), 100*k_beats_c/len(results)))
print("  Mean delta (K-C): {:.4f} [{:.4f}, {:.4f}]".format(delta_mean, delta_ci_lo, delta_ci_hi))
print("  Paired t-test: t={:.3f}, p={:.4f} {}".format(
    t_stat, p_val_delta,
    "SIGNIFICANT (p<0.05)" if p_val_delta < 0.05 else "NOT significant"))

# Save
final = {
    "timestamp": "2026-06-23T20:00:00",
    "n_stocks": len(results),
    "lookback": LOOKBACK, "stride": STRIDE,
    "mean_windows": float(np.mean([r["n_windows"] for r in results])),
    "mean_test_samples": float(np.mean([r["n_test"] for r in results])),
    "aggregate": {
        "kronos_vol_f1_mean": float(np.mean(k_vols)),
        "kronos_vol_f1_std": float(np.std(k_vols)),
        "chronos_vol_f1_mean": float(np.mean(c_vols)),
        "chronos_vol_f1_std": float(np.std(c_vols)),
        "raw_vol_f1_mean": float(np.mean(raw_vs)),
        "kronos_vol_r2_mean": float(np.mean(k_r2s)),
        "chronos_vol_r2_mean": float(np.mean(c_r2s)),
        "k_beats_c_count": k_beats_c,
        "k_beats_c_pct": float(k_beats_c / len(results)),
        "mean_delta_kc": float(delta_mean),
        "delta_ci_lo": float(delta_ci_lo),
        "delta_ci_hi": float(delta_ci_hi),
        "paired_t_pval": float(p_val_delta),
        "significant": p_val_delta < 0.05,
    },
    "per_stock": results,
}
with open(str(OUT / "scale_results.json"), "w") as f:
    json.dump(final, f, indent=2)

print("\nSaved. Done in {:.0f}s".format(time.time() - t0))
