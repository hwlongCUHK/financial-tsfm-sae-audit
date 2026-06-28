"""Shared SAE trained on all 120 stocks. Primary analysis: concept distribution,
feature labeling, ablation, and financial metrics on a single shared dictionary.
"""
import torch, numpy as np, json, time, os
from pathlib import Path; import pandas as pd
from collections import defaultdict
import sys
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/shared_sae_120.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH_SIZE, LR = 4, 64, 5000, 512, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1
N_STOCKS = 120
PRE_CAL_THRESHOLD = 0.15

LABEL_NAMES = ["momentum_5","trend","volatility","vol_persistence","autocorr_lag1",
               "autocorr_lag5","max_drawdown","var_95","max_1day_gain","max_1day_loss",
               "skewness","kurtosis","price_range","vol_clustering","volume_trend","volume_price_corr"]

tok = KronosTokenizer.from_pretrained("/data/houwanlong/models/Kronos-Tokenizer-base").to(device).eval()
with open("/data/houwanlong/models/Kronos-base/config.json") as f: cfg = json.load(f)
model = Kronos(s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"], n_layers=cfg["n_layers"],
               d_model=cfg["d_model"], n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
               ffn_dropout_p=cfg["ffn_dropout_p"], attn_dropout_p=cfg["attn_dropout_p"],
               resid_dropout_p=cfg["resid_dropout_p"], token_dropout_p=cfg["token_dropout_p"],
               learn_te=cfg["learn_te"])
sd = load_file("/data/houwanlong/models/Kronos-base/model.safetensors")
model.load_state_dict(sd, strict=False); model = model.to(device).half().eval()
d_model = cfg["d_model"]; d_hidden = d_model * EXPANSION

class TopKSAE(torch.nn.Module):
    def __init__(self, d, h, k):
        super().__init__()
        self.enc = torch.nn.Linear(d, h, bias=True)
        self.dec = torch.nn.Linear(h, d, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(d))
        self.k = k
    def encode(self, x):
        xc = x - self.b; lat = self.enc(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        m = torch.zeros_like(lat); m.scatter_(-1, idx, 1.0)
        return lat * m
    def decode(self, lat): return self.dec(lat) + self.b

def load_stock(fname):
    df = pd.read_csv(DATA / fname)
    for c in ["open","close","high","low","volume","amount"]:
        if c not in df.columns: df[c] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 100: return None
    mn, st = data.mean(0), data.std(0)
    dn = np.clip((data-mn)/(st+1e-5), -5, 5)
    nw = min(2000, (len(dn)-WINDOW)//STRIDE)
    if nw < 25: return None
    wins = np.stack([dn[i:i+WINDOW] for i in range(0, nw*STRIDE, STRIDE)])
    return wins, dn

def compute_labels(dn, n_wins):
    all_labels = []
    for i in range(n_wins):
        idx = i*STRIDE
        if idx+WINDOW>len(dn): break
        c = dn[idx:idx+WINDOW,1]; r = np.diff(c)/(c[:-1]+1e-5)
        feats = [c[-1]/c[-6]-1 if len(c)>=6 else 0, np.polyfit(np.arange(WINDOW),c,1)[0],
                 np.std(r), np.corrcoef(np.abs(r[1:]),np.abs(r[:-1]))[0,1] if len(r)>2 else 0,
                 np.corrcoef(r[1:],r[:-1])[0,1] if len(r)>2 else 0,
                 np.corrcoef(r[5:],r[:-5])[0,1] if len(r)>6 else 0,
                 np.min(c/np.maximum.accumulate(c)-1), np.percentile(r,5),
                 np.max(r), np.min(r), float(pd.Series(r).skew()) if len(r)>2 else 0,
                 float(pd.Series(r).kurtosis()) if len(r)>3 else 0,
                 (c.max()-c.min())/max(c.mean(),1e-5), np.mean(r**2)/(np.var(r)+1e-10),
                 np.mean(np.diff(dn[idx:idx+WINDOW,4])/(dn[idx:idx+WINDOW-1,4]+1e-5)),
                 np.corrcoef(r, np.diff(dn[idx:idx+WINDOW,4])[:len(r)]/(dn[idx:idx+WINDOW-1,4][:len(r)]+1e-5))[0,1] if len(r)>2 else 0]
        all_labels.append(feats)
    return np.array(all_labels)

def extract_acts(wins, n_win):
    acts = []
    def hook_fn(m,i,o):
        a = o[0] if isinstance(o,tuple) else o
        acts.append(a[:,-1,:].detach().cpu().float().numpy())
    hook = model.transformer[LAYER].register_forward_hook(hook_fn)
    with torch.no_grad():
        for b in range(0, n_win, 64):
            batch = torch.from_numpy(wins[b:b+64]).float().to(device)
            s1,s2 = tok.encode(batch, half=True); model(s1,s2)
    hook.remove()
    return np.concatenate(acts)

print(f"Shared SAE on {N_STOCKS} stocks")
all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])

# Phase 1: Collect all training activations and test data
all_train_acts = []
per_stock_test = []  # (fname, test_acts, labels, n_test)

for fi, fname in enumerate(all_csvs):
    loaded = load_stock(fname)
    if loaded is None:
        print(f"[{fi+1}/{len(all_csvs)}] {fname}: SKIP")
        continue
    wins, dn = loaded
    n_tr = int(len(wins)*TRAIN_SPLIT); n_val = int(len(wins)*VAL_SPLIT)
    n_test = len(wins)-n_tr-n_val
    if n_test < 10:
        print(f"[{fi+1}/{len(all_csvs)}] {fname}: SKIP (n_test={n_test})")
        continue

    train_acts = extract_acts(wins, n_tr)
    all_train_acts.append(train_acts)

    test_wins = wins[n_tr+n_val:]
    labels = compute_labels(dn, len(wins))[n_tr+n_val:]
    min_len = min(len(labels), n_test)
    labels = labels[:min_len]
    test_acts = extract_acts(test_wins[:min_len], min_len)

    per_stock_test.append({"fname": fname, "test_acts": test_acts, "labels": labels, "n_test": min_len})
    if (fi+1) % 20 == 0:
        print(f"[{fi+1}/{len(all_csvs)}] collected")

# Concatenate
all_train = np.concatenate(all_train_acts, axis=0)
n_stocks_valid = len(per_stock_test)
print(f"\n{n_stocks_valid} valid stocks, {all_train.shape[0]} training windows")

# Phase 2: Train shared SAE
print(f"Training shared SAE ({STEPS} steps, batch {BATCH_SIZE})...")
sae = TopKSAE(d_model, d_hidden, K).to(device)
at = torch.from_numpy(all_train).float().to(device)
opt = torch.optim.Adam(sae.parameters(), lr=LR)
t0 = time.time()
for s in range(STEPS):
    idx = torch.randint(0, len(at), (BATCH_SIZE,))
    xr = sae.encode(at[idx])
    loss = torch.nn.functional.mse_loss(sae.decode(xr), at[idx])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()
    if (s+1) % 1000 == 0:
        print(f"  step {s+1}/{STEPS}, loss={loss.item():.6f}")
train_time = time.time() - t0
print(f"Training done in {train_time:.0f}s")

# Phase 3: Evaluate on each stock's test set
print("Evaluating...")
all_corrs = []
all_lat_list = []
all_labels_list = []
type_dist = defaultdict(int)
per_stock_results = []

for ps in per_stock_test:
    at_test = torch.from_numpy(ps["test_acts"]).float().to(device)
    with torch.no_grad():
        lat_t = sae.encode(at_test)
        lat = lat_t.cpu().numpy()
        recon = sae.decode(lat_t).cpu().numpy()

    var_total = float(np.var(ps["test_acts"]))
    mse = float(np.mean((recon - ps["test_acts"])**2))
    ve = float(1.0 - mse/max(var_total, 1e-10))
    dead = float(((lat != 0).sum(axis=0)==0).mean())
    alive = int(((lat != 0).sum(axis=0) > 0).sum())

    # Feature-statistic correlations
    labels = ps["labels"]
    alive_where = (lat != 0).sum(axis=0) > 10
    n_stock_assignments = 0
    for j in np.where(alive_where)[0]:
        a = lat[:,j] != 0
        if a.sum() < 5: continue
        corrs = [abs(np.corrcoef(lat[a,j], labels[a,k])[0,1]) for k in range(len(LABEL_NAMES))]
        corrs = [0 if np.isnan(c) else c for c in corrs]; best_idx = int(np.argmax(corrs))
        all_corrs.append(corrs)
        all_lat_list.append(lat[a,j])
        all_labels_list.append(labels[a])
        if corrs[best_idx] > PRE_CAL_THRESHOLD:
            type_dist[LABEL_NAMES[best_idx]] += 1
            n_stock_assignments += 1

    # Top-50 ablation
    freq = (lat != 0).sum(axis=0)
    top50 = np.argsort(freq)[-50:]
    lat_ab = lat_t.clone()
    lat_ab[:, top50] = 0
    with torch.no_grad():
        recon_ab = sae.decode(lat_ab).cpu().numpy()
    cos_sim = float(np.mean([
        np.dot(recon[i], recon_ab[i])/(np.linalg.norm(recon[i])*np.linalg.norm(recon_ab[i])+1e-10)
        for i in range(len(recon))
    ]))

    per_stock_results.append({
        "ticker": ps["fname"].replace(".csv",""),
        "n_test": ps["n_test"],
        "var_explained": ve, "dead_rate": dead, "alive_count": alive,
        "n_assignments": n_stock_assignments,
        "ablation_cosine": cos_sim,
    })

total_assignments = sum(type_dist.values())
merged_pct = {k: round(v/max(total_assignments,1)*100, 1) for k,v in sorted(type_dist.items(), key=lambda x:-x[1])[:12]}
largest_pct = max(type_dist.values())/max(total_assignments,1) if type_dist else 0

# Phase 4: Pooled calibration (fixed thresholds)
print("\nPooled calibration...")
n_pooled = len(all_corrs)
for thresh in [0.15, 0.25, 0.35, 0.40, 0.50]:
    surviving = sum(1 for corrs in all_corrs if max(corrs) > thresh)
    print(f"  |r| > {thresh:.2f}: {surviving}/{n_pooled} ({100*surviving/max(n_pooled,1):.1f}%)")

# Aggregate
agg_ve = [r["var_explained"] for r in per_stock_results]
agg_dead = [r["dead_rate"] for r in per_stock_results]
agg_alive = [r["alive_count"] for r in per_stock_results]
agg_abl = [r["ablation_cosine"] for r in per_stock_results]

result = {
    "n_stocks": n_stocks_valid,
    "n_train_windows": int(all_train.shape[0]),
    "config": "k64_exp4x",
    "train_steps": STEPS,
    "train_time_s": int(train_time),
    "var_explained_mean": float(np.mean(agg_ve)),
    "var_explained_std": float(np.std(agg_ve)),
    "dead_rate_mean": float(np.mean(agg_dead)),
    "dead_rate_std": float(np.std(agg_dead)),
    "alive_count": int(np.mean(agg_alive)),
    "ablation_cosine_mean": float(np.mean(agg_abl)),
    "ablation_cosine_std": float(np.std(agg_abl)),
    "n_pooled_alive_features": n_pooled,
    "total_assignments": int(total_assignments),
    "n_families": len(type_dist),
    "largest_family_pct": round(largest_pct*100, 1),
    "top_10_distribution": merged_pct,
    "per_stock": per_stock_results,
}

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f: json.dump(result, f, indent=2)
print(f"\nShared SAE (120 stocks):")
print(f"  VE: {result['var_explained_mean']:.4f} +/- {result['var_explained_std']:.4f}")
print(f"  Dead: {result['dead_rate_mean']:.4f}, Alive: {result['alive_count']}")
print(f"  Families: {result['n_families']}, Largest: {result['largest_family_pct']}%")
print(f"  Ablation cosine: {result['ablation_cosine_mean']:.4f}")
print(f"  Total assignments: {result['total_assignments']}")
print(f"  Saved to {OUTPUT}")
