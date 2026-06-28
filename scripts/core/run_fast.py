import numpy as np, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load activations
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

# Generate financial labels
df = pd.read_csv("/data/houwanlong/finllm-mi/code/finetune_csv/data/HK_ali_09988_kline_5min_all.csv")
for col in ["open","close","high","low","volume","amount"]:
    if col not in df.columns:
        df[col] = 0.0
data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
data = data[~np.isnan(data).any(axis=1)]
mn, st = data.mean(0), data.std(0)
data_norm = np.clip((data - mn) / (st + 1e-5), -5, 5)

LOOKBACK, STRIDE = 64, 32
labels = []
for i in range(0, len(data) - LOOKBACK, STRIDE):
    close = data[i:i+LOOKBACK, 1]
    rets = np.diff(close) / (close[:-1] + 1e-5)
    vol = np.std(rets)
    slope = np.polyfit(np.arange(LOOKBACK), close, 1)[0]
    labels.append({
        "vol": 1 if vol > 0.001 else 0,
        "trend": 0 if abs(slope) < 1e-4 else (1 if slope > 0 else -1),
    })

n = min(min(v.shape[0] for v in k_acts.values()), min(v.shape[0] for v in c_acts.values()), len(labels))
for layer in k_acts: k_acts[layer] = k_acts[layer][:n]
for layer in c_acts: c_acts[layer] = c_acts[layer][:n]
y_vol = np.array([l["vol"] for l in labels[:n]])
y_trend = np.array([l["trend"] for l in labels[:n]])

print("Samples:", n)
print("Vol dist:", np.bincount(y_vol))
print("Trend dist:", np.bincount(y_trend + 1))

def probe(X, y):
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0)
    return cross_val_score(clf, Xs, y, cv=5, scoring="f1_macro").mean()

# EXP 1: Probing
print("\n=== EXP 1: Financial Property Probing ===")
header = "{:<8} {:<12} {:<12} {:<12} {:<12}".format("Layer", "K-Vol", "C-Vol", "K-Trend", "C-Trend")
print(header)
print("-" * 56)
best = {"k_vol": (0,0), "c_vol": (0,0), "k_trend": (0,0), "c_trend": (0,0)}

for layer in sorted(k_acts.keys()):
    kv = probe(k_acts[layer], y_vol)
    kt = probe(k_acts[layer], y_trend)
    cv_val = probe(c_acts[layer], y_vol) if layer in c_acts else 0
    ct_val = probe(c_acts[layer], y_trend) if layer in c_acts else 0
    print("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        "L"+str(layer), kv, cv_val, kt, ct_val))
    if kv > best["k_vol"][0]: best["k_vol"] = (kv, layer)
    if cv_val > best["c_vol"][0]: best["c_vol"] = (cv_val, layer)
    if kt > best["k_trend"][0]: best["k_trend"] = (kt, layer)
    if ct_val > best["c_trend"][0]: best["c_trend"] = (ct_val, layer)

print("\nBest volatility probe:")
print("  Kronos:  L{} F1={:.4f}".format(best["k_vol"][1], best["k_vol"][0]))
print("  Chronos: L{} F1={:.4f}".format(best["c_vol"][1], best["c_vol"][0]))
print("Best trend probe:")
print("  Kronos:  L{} F1={:.4f}".format(best["k_trend"][1], best["k_trend"][0]))
print("  Chronos: L{} F1={:.4f}".format(best["c_trend"][1], best["c_trend"][0]))

# EXP 2: Cross-Model Transfer
print("\n=== EXP 2: Cross-Model Transfer ===")
def transfer(X_train, y_train, X_test, y_test):
    Xts = StandardScaler().fit_transform(X_train)
    Xes = StandardScaler().fit_transform(X_test)
    clf = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0)
    clf.fit(Xts, y_train)
    return f1_score(y_test, clf.predict(Xes), average="macro")

pairs = [(0,0),(3,1),(5,2),(8,3),(9,4),(11,5)]
header = "{:<12} {:<12} {:<12}".format("Direction", "Vol F1", "Trend F1")
print(header)
print("-" * 36)
k2c_v, k2c_t, c2k_v, c2k_t = [], [], [], []

for kl, cl in pairs:
    if kl in k_acts and cl in c_acts:
        a = transfer(k_acts[kl], y_vol, c_acts[cl], y_vol); k2c_v.append(a)
        b = transfer(k_acts[kl], y_trend, c_acts[cl], y_trend); k2c_t.append(b)
        d = transfer(c_acts[cl], y_vol, k_acts[kl], y_vol); c2k_v.append(d)
        e = transfer(c_acts[cl], y_trend, k_acts[kl], y_trend); c2k_t.append(e)
        print("{:<12} {:<12.4f} {:<12.4f}".format(
            "K{}->C{}".format(kl,cl), a, b))
        print("{:<12} {:<12.4f} {:<12.4f}".format(
            "C{}->K{}".format(cl,kl), d, e))

mv_k2c = np.mean(k2c_v); mv_c2k = np.mean(c2k_v)
mt_k2c = np.mean(k2c_t); mt_c2k = np.mean(c2k_t)

print("\nTransfer Summary:")
print("  Mean K->C vol: {:.4f}, trend: {:.4f}".format(mv_k2c, mt_k2c))
print("  Mean C->K vol: {:.4f}, trend: {:.4f}".format(mv_c2k, mt_c2k))
print("  Asymmetry (vol): {:.4f} ({})".format(
    mv_c2k - mv_k2c,
    "C->K better" if mv_c2k > mv_k2c else "K->C better"))

# EXP 3: Feature Ablation
print("\n=== EXP 3: Feature Ablation (Volatility) ===")
def ablate(X, y):
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0)
    clf.fit(Xs, y)
    imp = np.abs(clf.coef_).flatten()
    full = f1_score(y, clf.predict(Xs), average="macro")

    n_ablate = max(1, int(len(imp) * 0.01))
    top = np.argsort(imp)[-n_ablate:]
    rnd = np.random.choice(len(imp), n_ablate, replace=False)

    Xa = Xs.copy(); Xa[:, top] = 0
    ca = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0).fit(Xa, y)
    ab_f1 = f1_score(y, ca.predict(Xa), average="macro")

    Xr = Xs.copy(); Xr[:, rnd] = 0
    cr = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0).fit(Xr, y)
    rn_f1 = f1_score(y, cr.predict(Xr), average="macro")

    return full, ab_f1, rn_f1

header = "{:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Layer", "Full", "Ablated", "Random", "Drop")
print(header)
print("-" * 50)
k_drops, c_drops = [], []

for name, acts in [("Kronos", k_acts), ("Chronos", c_acts)]:
    for layer in sorted(acts.keys()):
        f, a, r = ablate(acts[layer], y_vol)
        drop = f - a
        label = "{}L{}".format(name[:1], layer)
        print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            label, f, a, r, drop))
        if name == "Kronos":
            k_drops.append(drop)
        else:
            c_drops.append(drop)

print("\nKronos mean ablation drop: {:.4f}".format(np.mean(k_drops)))
print("Chronos mean ablation drop: {:.4f}".format(np.mean(c_drops)))
winner = "Kronos" if np.mean(k_drops) > np.mean(c_drops) else "Chronos"
print("=> {} features are more causally important for volatility prediction".format(winner))

# Save
results = {
    "probing": {
        "kronos_best_vol": {"layer": best["k_vol"][1], "f1": float(best["k_vol"][0])},
        "chronos_best_vol": {"layer": best["c_vol"][1], "f1": float(best["c_vol"][0])},
        "kronos_best_trend": {"layer": best["k_trend"][1], "f1": float(best["k_trend"][0])},
        "chronos_best_trend": {"layer": best["c_trend"][1], "f1": float(best["c_trend"][0])},
    },
    "transfer": {
        "mean_k2c_vol": float(mv_k2c), "mean_c2k_vol": float(mv_c2k),
        "mean_k2c_trend": float(mt_k2c), "mean_c2k_trend": float(mt_c2k),
        "asymmetry_vol": float(mv_c2k - mv_k2c),
    },
    "ablation": {
        "kronos_mean_drop": float(np.mean(k_drops)),
        "chronos_mean_drop": float(np.mean(c_drops)),
        "winner": winner,
    },
}
with open("/data/houwanlong/finllm-mi/outputs/full_probing_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("All results saved!")
print("=" * 60)
