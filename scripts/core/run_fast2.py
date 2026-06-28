import numpy as np, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Load
kronos_dir = Path("/data/houwanlong/finllm-mi/outputs/kronos_kline")
chronos_dir = Path("/data/houwanlong/finllm-mi/outputs/chronos_kline")

def load_acts(d, pat):
    acts = {}
    for f in sorted(d.glob(pat)):
        arr = np.load(f)
        acts[int(f.stem.split("_")[-1])] = arr.reshape(-1, arr.shape[-1])
    return acts

k_acts = load_acts(kronos_dir, "layer_*.npy")
c_acts = load_acts(chronos_dir, "layer_*.npy")
n_k = max(k_acts.keys()) + 1  # 12
n_c = max(c_acts.keys()) + 1  # 6
print("Kronos layers:", sorted(k_acts.keys()), "d_model:", k_acts[0].shape[-1])
print("Chronos layers:", sorted(c_acts.keys()), "d_model:", c_acts[0].shape[-1])

# Labels
df = pd.read_csv("/data/houwanlong/finllm-mi/code/finetune_csv/data/HK_ali_09988_kline_5min_all.csv")
for col in ["open","close","high","low","volume","amount"]:
    if col not in df.columns: df[col] = 0.0
data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
data = data[~np.isnan(data).any(axis=1)]
mn, st = data.mean(0), data.std(0)
data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)

LOOKBACK, STRIDE = 64, 32
labels = []
all_vols = []
for i in range(0, len(data) - LOOKBACK, STRIDE):
    close = data[i:i+LOOKBACK, 1]
    rets = np.diff(close) / (close[:-1] + 1e-5)
    vol = np.std(rets)
    slope = np.polyfit(np.arange(LOOKBACK), close, 1)[0]
    all_vols.append(vol)
    labels.append({"vol_c": float(vol), "trend_c": float(slope)})

vol_median = np.median(all_vols)
print("Vol median:", vol_median)

for l in labels:
    l["vol"] = 1 if l["vol_c"] > vol_median else 0
    l["trend"] = 0 if abs(l["trend_c"]) < 1e-4 else (1 if l["trend_c"] > 0 else -1)

n = min(min(v.shape[0] for v in k_acts.values()), min(v.shape[0] for v in c_acts.values()), len(labels))
for layer in k_acts: k_acts[layer] = k_acts[layer][:n]
for layer in c_acts: c_acts[layer] = c_acts[layer][:n]
y_vol = np.array([l["vol"] for l in labels[:n]])
y_trend = np.array([l["trend"] for l in labels[:n]])
y_vol_c = np.array([l["vol_c"] for l in labels[:n]], dtype=np.float64)

print("Samples:", n)
print("Vol dist:", np.bincount(y_vol))
print("Trend dist:", np.bincount(y_trend + 1))

def probe_clf(X, y):
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0)
    return cross_val_score(clf, Xs, y, cv=5, scoring="f1_macro").mean()

chance_vol = max(np.bincount(y_vol)) / len(y_vol)
chance_trend = max(np.bincount(y_trend + 1)) / len(y_trend)
print("Chance vol: {:.4f}, Chance trend: {:.4f}".format(chance_vol, chance_trend))

# ─── EXP 1: Layer-wise Probing (only common layers 0-5) ───
print("\n=== EXP 1: Financial Property Probing (L0-L5) ===")
header = "{:<8} {:<12} {:<12} {:<12} {:<12}".format("Layer", "K-Vol", "C-Vol", "K-Trend", "C-Trend")
print(header)
print("-" * 56)
best = {"k_vol": (0,0), "c_vol": (0,0), "k_trend": (0,0), "c_trend": (0,0)}

for layer in range(min(n_k, n_c)):
    kv = probe_clf(k_acts[layer], y_vol)
    cv_val = probe_clf(c_acts[layer], y_vol)
    kt = probe_clf(k_acts[layer], y_trend)
    ct_val = probe_clf(c_acts[layer], y_trend)
    print("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        "L"+str(layer), kv, cv_val, kt, ct_val))
    if kv > best["k_vol"][0]: best["k_vol"] = (kv, layer)
    if cv_val > best["c_vol"][0]: best["c_vol"] = (cv_val, layer)
    if kt > best["k_trend"][0]: best["k_trend"] = (kt, layer)
    if ct_val > best["c_trend"][0]: best["c_trend"] = (ct_val, layer)

# Also show Kronos deep layers
print("\nKronos deep layers (no Chronos equivalent):")
for layer in range(n_c, n_k):
    kv = probe_clf(k_acts[layer], y_vol)
    kt = probe_clf(k_acts[layer], y_trend)
    print("  L{}: Vol={:.4f}, Trend={:.4f}".format(layer, kv, kt))
    if kv > best["k_vol"][0]: best["k_vol"] = (kv, layer)
    if kt > best["k_trend"][0]: best["k_trend"] = (kt, layer)

print("\nBest probes:")
print("  Kronos:  Vol L{} F1={:.4f}, Trend L{} F1={:.4f}".format(
    best["k_vol"][1], best["k_vol"][0], best["k_trend"][1], best["k_trend"][0]))
print("  Chronos: Vol L{} F1={:.4f}, Trend L{} F1={:.4f}".format(
    best["c_vol"][1], best["c_vol"][0], best["c_trend"][1], best["c_trend"][0]))
print("  Chance:  Vol={:.4f}, Trend={:.4f}".format(chance_vol, chance_trend))

# ─── EXP 2: Cross-Model Transfer (PCA to common 128d) ───
print("\n=== EXP 2: Cross-Model Transfer (128d PCA) ===")

def transfer_pca(X_train, y_train, X_test, y_test, n_dims=128):
    """Project both to common PCA space, train on one, test on other."""
    # Learn PCA on training data
    pca = PCA(n_components=min(n_dims, X_train.shape[0], X_train.shape[1]))
    Xt = pca.fit_transform(StandardScaler().fit_transform(X_train))
    Xe = pca.transform(StandardScaler().fit_transform(X_test))
    clf = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0)
    clf.fit(Xt, y_train)
    return f1_score(y_test, clf.predict(Xe), average="macro")

pairs = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)]
header = "{:<14} {:<12} {:<12}".format("Direction", "Vol F1", "Trend F1")
print(header)
print("-" * 38)
k2c_v, k2c_t, c2k_v, c2k_t = [], [], [], []

for kl, cl in pairs:
    a = transfer_pca(k_acts[kl], y_vol, c_acts[cl], y_vol); k2c_v.append(a)
    b = transfer_pca(k_acts[kl], y_trend, c_acts[cl], y_trend); k2c_t.append(b)
    d = transfer_pca(c_acts[cl], y_vol, k_acts[kl], y_vol); c2k_v.append(d)
    e = transfer_pca(c_acts[cl], y_trend, k_acts[kl], y_trend); c2k_t.append(e)
    print("{:<14} {:<12.4f} {:<12.4f}".format("K{} -> C{}".format(kl,cl), a, b))
    print("{:<14} {:<12.4f} {:<12.4f}".format("C{} -> K{}".format(cl,kl), d, e))

mv_k2c = np.mean(k2c_v); mv_c2k = np.mean(c2k_v)
mt_k2c = np.mean(k2c_t); mt_c2k = np.mean(c2k_t)

print("\nTransfer Summary (Volatility):")
print("  Mean K->C: {:.4f}".format(mv_k2c))
print("  Mean C->K: {:.4f}".format(mv_c2k))
print("  Asymmetry: {:.4f} ({})".format(
    mv_c2k - mv_k2c,
    "C->K transfers better" if mv_c2k > mv_k2c else "K->C transfers better"))

# ─── EXP 3: Feature Ablation ───
print("\n=== EXP 3: Feature Ablation (top 1% features, Volatility) ===")

def ablate(X, y):
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0)
    clf.fit(Xs, y)
    imp = np.abs(clf.coef_).flatten()
    full = f1_score(y, clf.predict(Xs), average="macro")

    n_ab = max(1, int(len(imp) * 0.01))
    top = np.argsort(imp)[-n_ab:]
    rnd = np.random.choice(len(imp), n_ab, replace=False)

    Xa = Xs.copy(); Xa[:, top] = 0
    ca = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0).fit(Xa, y)
    ab_f1 = f1_score(y, ca.predict(Xa), average="macro")

    Xr = Xs.copy(); Xr[:, rnd] = 0
    cr = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0).fit(Xr, y)
    rn_f1 = f1_score(y, cr.predict(Xr), average="macro")

    return full, ab_f1, rn_f1, n_ab

header = "{:<10} {:<10} {:<10} {:<10} {:<10} {:<8}".format(
    "Layer", "Full", "Ablated", "Random", "Drop", "#Abl")
print(header)
print("-" * 58)
k_drops, c_drops = [], []

for name, acts, n_layers in [("Kronos", k_acts, n_k), ("Chronos", c_acts, n_c)]:
    for layer in range(n_layers):
        f, a, r, n_ab = ablate(acts[layer], y_vol)
        drop = f - a
        label = "{}L{}".format(name[:1], layer)
        print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<8}".format(
            label, f, a, r, drop, n_ab))
        if name == "Kronos":
            k_drops.append(drop)
        else:
            c_drops.append(drop)

mean_kd = np.mean(k_drops)
mean_cd = np.mean(c_drops)
print("\nKronos mean ablation drop: {:.4f} ({} features)".format(mean_kd, "more important" if mean_kd > mean_cd else "less important"))
print("Chronos mean ablation drop: {:.4f}".format(mean_cd))
print("=> {} features more causally linked to output".format(
    "Kronos" if mean_kd > mean_cd else "Chronos"))

# ─── Save ───
results = {
    "n_samples": int(n), "n_kronos_layers": n_k, "n_chronos_layers": n_c,
    "kronos_d": int(k_acts[0].shape[-1]), "chronos_d": int(c_acts[0].shape[-1]),
    "chance_vol": float(chance_vol), "chance_trend": float(chance_trend),
    "probing": {
        "kronos_best_vol": {"layer": best["k_vol"][1], "f1": float(best["k_vol"][0])},
        "chronos_best_vol": {"layer": best["c_vol"][1], "f1": float(best["c_vol"][0])},
        "kronos_best_trend": {"layer": best["k_trend"][1], "f1": float(best["k_trend"][0])},
        "chronos_best_trend": {"layer": best["c_trend"][1], "f1": float(best["c_trend"][0])},
    },
    "transfer": {
        "mean_k2c_vol": float(mv_k2c), "mean_c2k_vol": float(mv_c2k),
        "asymmetry_vol": float(mv_c2k - mv_k2c),
    },
    "ablation": {
        "kronos_mean_drop": float(mean_kd),
        "chronos_mean_drop": float(mean_cd),
    },
}
with open("/data/houwanlong/finllm-mi/outputs/full_probing_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("Saved: /data/houwanlong/finllm-mi/outputs/full_probing_results.json")
