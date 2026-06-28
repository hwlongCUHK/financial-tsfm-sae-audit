#!/usr/bin/env python3
"""Full SAE pipeline on all 120 stocks: concept distribution, ablation, permutation, steering.
Saves to /data/houwanlong/finllm-mi/outputs/sae/scale120_results.json
"""
import torch, numpy as np, json, time, os, sys
from pathlib import Path; import pandas as pd
from collections import defaultdict
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/scale120_results.json"
STATE_FILE = "/data/houwanlong/finllm-mi/outputs/sae/scale120_state.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH_SIZE, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1
PRE_CAL_THRESHOLD = 0.15

LABEL_NAMES = ["momentum_5","trend","volatility","vol_persistence","autocorr_lag1",
               "autocorr_lag5","max_drawdown","var_95","max_1day_gain","max_1day_loss",
               "skewness","kurtosis","price_range","vol_clustering","volume_trend","volume_price_corr"]

# --- Model loading ---
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
    if len(data) < 100:
        print(f"    WARNING: {fname} has only {len(data)} rows, skipping")
        return None
    mn, st = data.mean(0), data.std(0)
    dn = np.clip((data-mn)/(st+1e-5), -5, 5)
    nw = min(2000, (len(dn)-WINDOW)//STRIDE)
    if nw < 25:
        print(f"    WARNING: {fname} has only {nw} windows, skipping")
        return None
    wins = np.stack([dn[i:i+WINDOW] for i in range(0, nw*STRIDE, STRIDE)])
    return wins, dn

def compute_labels(dn, n_wins):
    all_labels = []
    for i in range(n_wins):
        idx = i * STRIDE
        if idx + WINDOW > len(dn): break
        c = dn[idx:idx+WINDOW, 1]; r = np.diff(c)/(c[:-1]+1e-5)
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

def extract_acts(wins, n_train):
    acts = []
    def hook_fn(m,i,o):
        a = o[0] if isinstance(o,tuple) else o
        acts.append(a[:,-1,:].detach().cpu().float().numpy())
    hook = model.transformer[LAYER].register_forward_hook(hook_fn)
    with torch.no_grad():
        for b in range(0, n_train, 64):
            batch = torch.from_numpy(wins[b:b+64]).float().to(device)
            s1,s2 = tok.encode(batch, half=True); model(s1,s2)
    hook.remove()
    return np.concatenate(acts)

def train_sae(sae, train_acts):
    at = torch.from_numpy(train_acts).float().to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    for s in range(STEPS):
        idx = torch.randint(0, len(at), (BATCH_SIZE,))
        xr = sae.encode(at[idx])
        loss = torch.nn.functional.mse_loss(sae.decode(xr), at[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()
    return sae

def evaluate_concepts(sae, test_acts, labels):
    at = torch.from_numpy(test_acts).float().to(device)
    with torch.no_grad():
        lat_tensor = sae.encode(at)
        lat = lat_tensor.cpu().numpy()
        recon = sae.decode(lat_tensor).cpu().numpy()
    var_total = float(np.var(test_acts))
    mse = float(np.mean((recon - test_acts)**2))
    var_explained = float(1.0 - mse/max(var_total, 1e-10))
    dead_mask = (lat != 0).sum(axis=0) == 0
    dead_rate = float(dead_mask.mean())
    alive_count = int((~dead_mask).sum())

    # Feature-statistic correlations
    alive_where = (lat != 0).sum(axis=0) > 10
    type_dist_pre = defaultdict(int)
    feature_corrs = {}
    for j in np.where(alive_where)[0]:
        a = lat[:,j] != 0
        if a.sum() < 5: continue
        corrs = [abs(np.corrcoef(lat[a,j], labels[a,k])[0,1]) for k in range(len(LABEL_NAMES))]
        corrs = [0 if np.isnan(c) else c for c in corrs]; best = np.argmax(corrs)
        feature_corrs[int(j)] = corrs
        if corrs[best] > PRE_CAL_THRESHOLD:
            type_dist_pre[LABEL_NAMES[best]] += 1

    total = sum(type_dist_pre.values())
    largest = max(type_dist_pre.values())/max(total,1) if type_dist_pre else 0
    n_families = len(type_dist_pre)

    return {"var_explained": var_explained, "dead_rate": dead_rate, "alive_count": alive_count,
            "largest_pct": largest, "n_families_pre": n_families, "type_dist_pre": dict(type_dist_pre),
            "_feature_corrs": feature_corrs, "_lat": lat, "_labels": labels}

def null_calibrate_all(eval_results, n_shuffles=100):
    rng = np.random.RandomState(42)
    # Per-stock calibration
    per_stock_thresholds = {}
    for er in eval_results:
        stock_id = er["_fname"]
        lat = er["_lat"]; labels = er["_labels"]
        null_maxes = []
        shuf = labels.copy()
        for _ in range(n_shuffles):
            for c in range(shuf.shape[1]):
                rng.shuffle(shuf[:,c])
            nm = []
            for j, _corrs in er["_feature_corrs"].items():
                a = lat[:,j] != 0
                if a.sum() < 5: continue
                corrs = [abs(np.corrcoef(lat[a,j], shuf[a,k])[0,1]) for k in range(shuf.shape[1])]
                nm.append(max([0 if np.isnan(c) else c for c in corrs]))
            if nm: null_maxes.append(max(nm))
        per_stock_thresholds[stock_id] = float(np.percentile(null_maxes, 95)) if null_maxes else 0

    # Re-count with per-stock thresholds
    type_dist_null_all = defaultdict(int)
    for er in eval_results:
        stock_id = er["_fname"]
        thresh = per_stock_thresholds[stock_id]
        lat = er["_lat"]; labels = er["_labels"]
        for j, corrs in er["_feature_corrs"].items():
            a = lat[:,j] != 0
            if a.sum() < 5: continue
            best = int(np.argmax(corrs))
            if corrs[best] > thresh:
                type_dist_null_all[LABEL_NAMES[best]] += 1

    null_total = sum(type_dist_null_all.values())
    return per_stock_thresholds, dict(type_dist_null_all)

def run_ablation(sae, test_acts, top_k=50):
    """Top-K feature ablation: zero top_k most active features, measure accuracy drop."""
    at = torch.from_numpy(test_acts).float().to(device)
    with torch.no_grad():
        lat = sae.encode(at).cpu().numpy()
    # Get top-k features by activation frequency
    freq = (lat != 0).sum(axis=0)
    top_feats = np.argsort(freq)[-top_k:]

    # Ablation
    with torch.no_grad():
        lat_tensor = sae.encode(at)
        lat_ablated = lat_tensor.clone()
        lat_ablated[:, top_feats] = 0
        recon_ablated = sae.decode(lat_ablated).cpu().numpy()
        recon_baseline = sae.decode(lat_tensor).cpu().numpy()

    # Cosine similarity
    cos_sim = np.mean([
        np.dot(recon_ablated[i], recon_baseline[i]) /
        (np.linalg.norm(recon_ablated[i])*np.linalg.norm(recon_baseline[i])+1e-10)
        for i in range(len(recon_ablated))
    ])
    return float(cos_sim)

# --- Load state for resumption ---
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {"completed": [], "results": {}}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    serializable = {"completed": state["completed"], "results": {}}
    for ticker, r in state["results"].items():
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.ndarray,)): continue  # skip numpy arrays
            if isinstance(v, (np.floating,)): sr[k] = float(v)
            elif isinstance(v, (np.integer,)): sr[k] = int(v)
            elif isinstance(v, dict):
                sr[k] = {str(kk): (float(vv) if isinstance(vv,(np.floating,)) else int(vv) if isinstance(vv,(np.integer,)) else vv) for kk,vv in v.items()}
            else: sr[k] = v
        serializable["results"][ticker] = sr
    with open(STATE_FILE, "w") as f: json.dump(serializable, f)

print("Scale-120 full SAE pipeline")
print(f"Data: {DATA}, Output: {OUTPUT}")

all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])
print(f"Found {len(all_csvs)} CSV files")

state = load_state()
completed = set(state["completed"])
results = state["results"]

for fi, fname in enumerate(all_csvs):
    ticker = fname.replace(".csv","")
    if ticker in completed:
        continue
    loaded = load_stock(fname)
    if loaded is None:
        print(f"[{fi+1}/{len(all_csvs)}] {ticker}: SKIP (insufficient data)")
        completed.add(ticker)
        save_state({"completed": list(completed), "results": results})
        continue
    wins, dn = loaded
    n_tr = int(len(wins)*TRAIN_SPLIT); n_val = int(len(wins)*VAL_SPLIT)
    n_test = len(wins)-n_tr-n_val
    if n_test < 10:
        print(f"[{fi+1}/{len(all_csvs)}] {ticker}: SKIP (n_test={n_test}<10)")
        completed.add(ticker)
        save_state({"completed": list(completed), "results": results})
        continue

    print(f"[{fi+1}/{len(all_csvs)}] {ticker}: processing (n_tr={n_tr}, n_test={n_test})...")

    train_acts = extract_acts(wins, n_tr)
    test_wins = wins[n_tr+n_val:]
    labels = compute_labels(dn, len(wins))[n_tr+n_val:]
    min_len = min(len(labels), n_test)
    labels = labels[:min_len]
    if len(labels) < 10: continue

    test_wins_only = wins[n_tr+n_val:]
    test_acts = extract_acts(test_wins_only, len(test_wins_only))

    sae = TopKSAE(d_model, d_hidden, K).to(device)
    train_sae(sae, train_acts)

    eval_r = evaluate_concepts(sae, test_acts, labels)
    eval_r["_fname"] = ticker
    abl_cos = run_ablation(sae, test_acts)
    eval_r["ablation_cosine"] = abl_cos

    del sae; torch.cuda.empty_cache()

    results[ticker] = eval_r
    completed.add(ticker)
    save_state({"completed": list(completed), "results": results})
    print(f"  var_expl={eval_r['var_explained']:.4f}, dead={eval_r['dead_rate']:.4f}, "
          f"n_fam={eval_r['n_families_pre']}, largest={eval_r['largest_pct']:.4f}, "
          f"abl_cos={abl_cos:.4f}")

# --- Aggregate and save final ---
print(f"\nAggregating {len(results)} stocks...")
per_stock_thresholds, null_dist = null_calibrate_all(list(results.values()))

# Aggregate type distributions
merged_pre = defaultdict(int)
agg_vars, agg_dead, agg_alive, agg_largest, agg_nfam, agg_cos = [],[],[],[],[],[]
for ticker, r in results.items():
    for k,v in r["type_dist_pre"].items(): merged_pre[k] += v
    agg_vars.append(r["var_explained"]); agg_dead.append(r["dead_rate"])
    agg_alive.append(r["alive_count"]); agg_largest.append(r["largest_pct"])
    agg_nfam.append(r["n_families_pre"]); agg_cos.append(r.get("ablation_cosine",0))

total = sum(merged_pre.values())
merged_pct = {k: round(v/max(total,1)*100,1) for k,v in merged_pre.items()}
null_total = sum(null_dist.values())

final = {
    "n_stocks": len(results),
    "var_explained_mean": float(np.mean(agg_vars)), "var_explained_std": float(np.std(agg_vars)),
    "dead_rate_mean": float(np.mean(agg_dead)), "dead_rate_std": float(np.std(agg_dead)),
    "alive_count_mean": float(np.mean(agg_alive)), "alive_count_std": float(np.std(agg_alive)),
    "largest_pct_mean": float(np.mean(agg_largest)), "largest_pct_std": float(np.std(agg_largest)),
    "n_families_pre_mean": float(np.mean(agg_nfam)), "n_families_pre_std": float(np.std(agg_nfam)),
    "ablation_cosine_mean": float(np.mean(agg_cos)), "ablation_cosine_std": float(np.std(agg_cos)),
    "merged_type_dist_pre_pct": merged_pct,
    "merged_total_pre": int(total),
    "null_95_median": float(np.median(list(per_stock_thresholds.values()))),
    "null_type_dist": null_dist,
    "null_n_families": len(null_dist),
    "null_total": int(null_total),
    "per_stock_thresholds": per_stock_thresholds,
}

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f: json.dump(final, f, indent=2)
print(f"Saved to {OUTPUT}")
print(f"N={len(results)}, var_expl={final['var_explained_mean']:.4f}, "
      f"dead={final['dead_rate_mean']:.4f}, largest={final['largest_pct_mean']:.4f}, "
      f"n_fam_pre={final['n_families_pre_mean']:.1f}, null_95_median={final['null_95_median']:.4f}, "
      f"ablation_cos={final['ablation_cosine_mean']:.4f}")
