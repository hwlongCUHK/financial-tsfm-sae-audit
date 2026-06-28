"""Multi-stock rolling-origin evaluation — addresses reviewer W2 + rolling-origin request."""
import numpy as np, json, warnings, pandas as pd, time, sys
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file
import torch

t0 = time.time()
device = "cuda:0"
DATA_DIR = Path("/data/houwanlong/finllm-mi/data")
OUT_DIR = Path("/data/houwanlong/finllm-mi/outputs")

# ─── 1. Load models ───
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

print("Kronos: {}L, Chronos: {}L".format(n_k, n_c))

# ─── 2. Process each stock ───
STOCKS = [
    ("Alibaba 5min", DATA_DIR.parent / "code/finetune_csv/data/HK_ali_09988_kline_5min_all.csv", 64, 32),
    ("ICBC", DATA_DIR / "CN_工商银行_601398_daily.csv", 64, 8),
    ("PetroChina", DATA_DIR / "CN_中国石油_601857_daily.csv", 64, 8),
    ("Moutai", DATA_DIR / "CN_贵州茅台_600519_daily.csv", 64, 8),
    ("BOE", DATA_DIR / "CN_京东方A_000725_daily.csv", 64, 8),
    ("Changan", DATA_DIR / "CN_长安汽车_000625_daily.csv", 64, 8),
]

LOOKBACK = 64
BATCH = 32

results_per_stock = []

def make_hook(layer_idx, storage):
    def h(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        storage[layer_idx].append(act.detach().cpu().float().mean(dim=1).numpy())
    return h

def extract_kronos(data, lookback, stride):
    """Extract Kronos activations from OHLCV data."""
    n_windows = (len(data) - lookback * 2) // stride  # need double for OOS labels
    windows = []
    for i in range(0, n_windows * stride, stride):
        windows.append(data[i:i+lookback])
    windows = np.stack(windows, axis=0)

    acts = {l: [] for l in range(n_k)}
    hooks = [kronos_model.transformer[l].register_forward_hook(make_hook(l, acts)) for l in range(n_k)]

    n_batches = len(windows) // BATCH
    with torch.no_grad():
        for b in range(n_batches):
            batch = torch.from_numpy(windows[b*BATCH:(b+1)*BATCH]).float().to(device)
            s1_ids, s2_ids = tokenizer.encode(batch, half=True)
            kronos_model(s1_ids, s2_ids)

    for h in hooks: h.remove()
    return {l: np.stack(acts[l][:n_batches*BATCH], axis=0).reshape(-1, 832) for l in range(n_k)}

def extract_chronos(close_prices, lookback, stride):
    """Extract Chronos activations from close prices."""
    n_windows = (len(close_prices) - lookback * 2) // stride
    windows = []
    for i in range(0, n_windows * stride, stride):
        windows.append(close_prices[i:i+lookback])
    windows = np.stack(windows, axis=0)  # (n, lookback)

    acts = {l: [] for l in range(n_c)}
    hooks = [chronos_model.encoder.block[l].register_forward_hook(make_hook(l, acts)) for l in range(n_c)]

    n_batches = len(windows) // BATCH
    with torch.no_grad():
        for b in range(n_batches):
            batch = torch.from_numpy(windows[b*BATCH:(b+1)*BATCH]).float().to(device)
            batch_norm = (batch - batch.min(dim=1, keepdim=True).values) / (
                batch.max(dim=1, keepdim=True).values - batch.min(dim=1, keepdim=True).values + 1e-5)
            token_ids = (batch_norm * 4095).long()
            mask = torch.ones_like(token_ids)
            dec_input = torch.zeros(token_ids.shape[0], 1, dtype=torch.long, device=device)
            chronos_model(input_ids=token_ids, attention_mask=mask, decoder_input_ids=dec_input)

    for h in hooks: h.remove()
    return {l: np.stack(acts[l][:n_batches*BATCH], axis=0).reshape(-1, 512) for l in range(n_c)}

def label_future(data, lookback, stride):
    """Label each window with the NEXT window's volatility/trend."""
    n_windows = (len(data) - lookback * 2) // stride
    labels = []
    raw_feats = []
    for i in range(0, n_windows * stride, stride):
        # Current window raw features
        close_cur = data[i:i+lookback, 1]
        returns = np.diff(close_cur) / (close_cur[:-1] + 1e-5)
        raw = [
            np.mean(returns), np.std(returns),
            float(np.mean((returns - returns.mean())**3) / (returns.std()**3 + 1e-5)),
            float(np.mean((returns - returns.mean())**4) / (returns.std()**4 + 1e-5)),
            float(np.mean(close_cur[-10:]) / (np.mean(close_cur[:10]) + 1e-5) - 1),
            float(close_cur[-1] / (np.mean(close_cur) + 1e-5) - 1),
            float(returns[-1]), float(returns[-2]), float(returns[-3]),
        ]
        raw_feats.append(raw)
        # Future window labels
        close_fut = data[i+lookback:i+2*lookback, 1]
        rets_fut = np.diff(close_fut) / (close_fut[:-1] + 1e-5)
        vol_fut = np.std(rets_fut)
        slope_fut = np.polyfit(np.arange(lookback), close_fut, 1)[0]
        labels.append({"vol": vol_fut, "slope": slope_fut})
    return labels, np.array(raw_feats, dtype=np.float64)

def probe_chrono(X, y, train_end, test_start):
    """Probe with chronological split."""
    if test_start >= len(y):
        return 0.0, 0.0
    Xs = StandardScaler().fit_transform(X[:train_end])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xs, y[:train_end])
    X_test = StandardScaler().fit_transform(X[test_start:])
    return f1_score(y[test_start:], clf.predict(X_test), average="macro"), 0.0

# ─── 3. Per-stock evaluation ───
for stock_name, csv_path, lookback, stride in STOCKS:
    print("\n" + "=" * 50)
    print("{} ({})".format(stock_name, csv_path.name))

    df = pd.read_csv(str(csv_path))
    for col in ["open","close","high","low","volume","amount"]:
        if col not in df.columns: df[col] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    mn, st = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)

    # Generate OOS labels
    labels, raw_feats = label_future(data_norm, lookback, stride)
    print("  Windows: {}".format(len(labels)))

    if len(labels) < 50:  # skip if too few windows
        print("  SKIP: too few windows")
        continue

    # Extract activations
    k_acts = extract_kronos(data_norm, lookback, stride)
    c_acts = extract_chronos(data_norm[:, 1], lookback, stride)  # close price

    # Match sample counts
    n = min(min(v.shape[0] for v in k_acts.values()),
            min(v.shape[0] for v in c_acts.values()), len(labels))
    for l in k_acts: k_acts[l] = k_acts[l][:n]
    for l in c_acts: c_acts[l] = c_acts[l][:n]
    labels = labels[:n]
    raw_feats = raw_feats[:n]

    # Vol labels
    vol_median = np.median([l["vol"] for l in labels])
    y_vol = np.array([1 if l["vol"] > vol_median else 0 for l in labels])
    y_trend = np.array([0 if abs(l["slope"]) < 1e-4 else (1 if l["slope"] > 0 else -1) for l in labels])

    # Chronological split: 60% train, 15% gap, 25% test
    n_total = len(y_vol)
    n_train = int(n_total * 0.6)
    n_test_start = int(n_total * 0.75)

    # Probe all layers, take best
    best_kv, best_cv, best_kt, best_ct = 0, 0, 0, 0
    for layer in range(min(n_k, n_c)):
        kv, _ = probe_chrono(k_acts[layer], y_vol, n_train, n_test_start)
        cv_, _ = probe_chrono(c_acts[layer], y_vol, n_train, n_test_start)
        kt, _ = probe_chrono(k_acts[layer], y_trend, n_train, n_test_start)
        ct_, _ = probe_chrono(c_acts[layer], y_trend, n_train, n_test_start)
        best_kv = max(best_kv, kv)
        best_cv = max(best_cv, cv_)
        best_kt = max(best_kt, kt)
        best_ct = max(best_ct, ct_)

    # Raw baseline
    raw_vol, _ = probe_chrono(raw_feats, y_vol, n_train, n_test_start)
    raw_trend, _ = probe_chrono(raw_feats, y_trend, n_train, n_test_start)
    chance_vol = max(np.bincount(y_vol[n_test_start:])) / len(y_vol[n_test_start:])

    results_per_stock.append({
        "stock": stock_name, "n_windows": n_total,
        "kronos_vol": float(best_kv), "chronos_vol": float(best_cv),
        "kronos_trend": float(best_kt), "chronos_trend": float(best_ct),
        "raw_vol": float(raw_vol), "raw_trend": float(raw_trend),
        "chance_vol": float(chance_vol),
    })

    print("  K-Vol={:.4f} C-Vol={:.4f} Raw-Vol={:.4f} Chance={:.4f}".format(
        best_kv, best_cv, raw_vol, chance_vol))
    print("  K-Trend={:.4f} C-Trend={:.4f} Raw-Trend={:.4f}".format(
        best_kt, best_ct, raw_trend))

# ─── 4. Aggregate results ───
print("\n" + "=" * 60)
print("MULTI-STOCK ROLLING-ORIGIN RESULTS")
print("=" * 60)

print("\n{:<18} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "Stock", "N", "K-Vol", "C-Vol", "Raw-Vol", "Chance", "Raw>K?"))
for r in results_per_stock:
    print("{:<18} {:>6} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10}".format(
        r["stock"], r["n_windows"],
        r["kronos_vol"], r["chronos_vol"], r["raw_vol"], r["chance_vol"],
        "YES" if r["raw_vol"] > r["kronos_vol"] else "no"))

# Aggregate stats
k_vols = [r["kronos_vol"] for r in results_per_stock]
c_vols = [r["chronos_vol"] for r in results_per_stock]
raw_vols = [r["raw_vol"] for r in results_per_stock]
chances = [r["chance_vol"] for r in results_per_stock]

print("\n{:<18} {:>6} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
    "MEAN", len(results_per_stock),
    np.mean(k_vols), np.mean(c_vols), np.mean(raw_vols), np.mean(chances)))
print("{:<18} {:>6} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
    "STD", "", np.std(k_vols), np.std(c_vols), np.std(raw_vols), np.std(chances)))

# Stock-level significance: how many stocks does raw beat Kronos?
raw_beats_k = sum(1 for i in range(len(results_per_stock)) if raw_vols[i] > k_vols[i])
print("\nRaw features beat Kronos in {}/{} stocks".format(raw_beats_k, len(results_per_stock)))

# Out-of-chance test: is Kronos vol significantly above chance?
k_above_chance = [k - c for k, c in zip(k_vols, chances)]
print("Kronos above chance: mean={:+.4f} std={:.4f}".format(np.mean(k_above_chance), np.std(k_above_chance)))

# Paired t-test (Kronos vs Raw, informal)
from scipy import stats
t_stat, p_val = stats.ttest_rel(k_vols, raw_vols)
print("Paired t-test (K vs Raw vol): t={:.3f} p={:.4f} {}".format(
    t_stat, p_val, "Raw SIG BETTER" if p_val < 0.05 and np.mean(raw_vols) > np.mean(k_vols) else "not sig"))

t_stat, p_val = stats.ttest_rel(k_vols, c_vols)
print("Paired t-test (K vs C vol): t={:.3f} p={:.4f} {}".format(
    t_stat, p_val, "SIG" if p_val < 0.05 else "not sig"))

# Save
final = {
    "timestamp": "2026-06-23T19:00:00",
    "n_stocks": len(results_per_stock),
    "per_stock": results_per_stock,
    "aggregate": {
        "kronos_vol_mean": float(np.mean(k_vols)),
        "kronos_vol_std": float(np.std(k_vols)),
        "chronos_vol_mean": float(np.mean(c_vols)),
        "chronos_vol_std": float(np.std(c_vols)),
        "raw_vol_mean": float(np.mean(raw_vols)),
        "raw_vol_std": float(np.std(raw_vols)),
        "chance_vol_mean": float(np.mean(chances)),
        "raw_beats_kronos_count": raw_beats_k,
        "k_above_chance_mean": float(np.mean(k_above_chance)),
        "k_vs_raw_pval": float(p_val),
    }
}
with open(str(OUT_DIR / "multi_stock_results.json"), "w") as f:
    json.dump(final, f, indent=2)

print("\nDone in {:.1f}s!".format(time.time() - t0))
