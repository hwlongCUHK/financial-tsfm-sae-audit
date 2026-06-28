"""Cross-stock confirmation: split 120 stocks into discovery/confirmation,
test whether feature-statistic associations replicate out-of-sample.
No new SAE training needed — uses per-stock activations from training data.
"""
import torch, numpy as np, json, time, os
from pathlib import Path; import pandas as pd
from collections import defaultdict
from scipy.stats import fisher_exact, combine_pvalues
import sys
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/cross_stock_confirmation_120.json"
STATE = "/data/houwanlong/finllm-mi/outputs/sae/cross_stock_confirmation_state.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1
DISCOVERY_RATIO = 0.5  # 60 discovery / 60 confirmation

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
    if len(data) < 100: return None, None
    mn, st = data.mean(0), data.std(0)
    dn = np.clip((data-mn)/(st+1e-5), -5, 5)
    nw = min(2000, (len(dn)-WINDOW)//STRIDE)
    if nw < 25: return None, None
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
            batch = torch.as_tensor(wins[b:b+64].copy(), dtype=torch.float32, device=device)
            s1,s2 = tok.encode(batch, half=True); model(s1,s2)
    hook.remove()
    return np.concatenate(acts)

# Load state
if os.path.exists(STATE):
    s = json.load(open(STATE))
    completed = set(s.get("completed",[]))
else:
    completed = set()

def save_state():
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump({"completed":list(completed)}, open(STATE,"w"))

all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])
rng = np.random.RandomState(42)
t0 = time.time()

# Phase 1: Collect per-stock activations and labels for all 120 stocks
print("Phase 1: Collecting per-stock data...")
per_stock = {}  # ticker -> (activations, labels, n_windows)
for fi, fname in enumerate(all_csvs):
    ticker = fname.replace(".csv","")
    if ticker in completed: continue
    loaded = load_stock(fname)
    if loaded is None: completed.add(ticker); save_state(); continue
    wins, dn = loaded
    n_tr = int(len(wins)*TRAIN_SPLIT); n_val = int(len(wins)*VAL_SPLIT)
    n_test = len(wins)-n_tr-n_val
    if n_test < 10: completed.add(ticker); save_state(); continue

    acts = extract_acts(wins, n_tr)
    labels = compute_labels(dn, len(wins))[:n_tr]
    m = min(len(acts), len(labels))
    per_stock[ticker] = (acts[:m], labels[:m])
    completed.add(ticker)
    if (fi+1) % 20 == 0:
        save_state()
        print(f"  [{len(completed)}] collected")

valid_tickers = list(per_stock.keys())
rng.shuffle(valid_tickers)
n_disc = int(len(valid_tickers) * DISCOVERY_RATIO)
disc_tickers = set(valid_tickers[:n_disc])
conf_tickers = set(valid_tickers[n_disc:])
print(f"Discovery: {len(disc_tickers)}, Confirmation: {len(conf_tickers)}")

# Phase 2: Train shared SAE (pooled discovery stocks only)
print("\nPhase 2: Training shared SAE on discovery stocks...")
disc_acts = np.concatenate([per_stock[t][0] for t in disc_tickers], axis=0)
sae = TopKSAE(d_model, d_hidden, K).to(device)
at_disc = torch.as_tensor(np.ascontiguousarray(disc_acts), dtype=torch.float32, device=device)
opt = torch.optim.Adam(sae.parameters(), lr=LR)
for s in range(STEPS):
    idx = torch.randint(0, len(at_disc), (BATCH,))
    xr = sae.encode(at_disc[idx])
    loss = torch.nn.functional.mse_loss(sae.decode(xr), at_disc[idx])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()
    if (s+1) % 1000 == 0: print(f"  step {s+1}/{STEPS}, loss={loss.item():.4f}")

# Get alive features on discovery set
with torch.no_grad():
    lat_disc = sae.encode(at_disc).detach().cpu().numpy()
alive_mask = (lat_disc != 0).sum(axis=0) > 0
alive_features = np.where(alive_mask)[0]
print(f"Alive features: {len(alive_features)}")

# Phase 3: Feature labeling on discovery set
print("\nPhase 3: Labeling features on discovery set...")
disc_labels = np.concatenate([per_stock[t][1] for t in disc_tickers], axis=0)
m = min(len(lat_disc), len(disc_labels))
lat_disc = lat_disc[:m]; disc_labels = disc_labels[:m]

feature_label = {}  # feature_idx -> (best_statistic, best_corr)
for j in alive_features:
    a = lat_disc[:,j] != 0
    if a.sum() < 10: continue
    corrs = []
    for k in range(len(LABEL_NAMES)):
        cc = np.corrcoef(lat_disc[a,j], disc_labels[a,k])[0,1]
        corrs.append(0.0 if np.isnan(cc) else abs(float(cc)))
    best_k = int(np.argmax(corrs))
    feature_label[int(j)] = (LABEL_NAMES[best_k], corrs[best_k])

print(f"Labeled features: {len(feature_label)}")

# Phase 4: Confirmation on held-out stocks
print("\nPhase 4: Confirmation analysis...")
confirmation_results = []
for ticker in conf_tickers:
    acts, labels = per_stock[ticker]
    m = min(len(acts), len(labels))
    acts = acts[:m]; labels = labels[:m]
    at = torch.as_tensor(np.ascontiguousarray(acts), dtype=torch.float32, device=device)
    with torch.no_grad():
        lat_stock = sae.encode(at).detach().cpu().numpy()

    for feat_j, (label, disc_corr) in feature_label.items():
        a = lat_stock[:,feat_j] != 0
        if a.sum() < 5: continue
        # Test the SAME association on confirmation stock
        stat_k = LABEL_NAMES.index(label)
        cc = np.corrcoef(lat_stock[a,feat_j], labels[a,stat_k])[0,1]
        conf_corr = 0.0 if np.isnan(cc) else float(cc)
        sign_agree = 1 if conf_corr > 0 else 0  # direction agreement
        confirmation_results.append({
            "ticker": ticker, "feature": feat_j,
            "label": label, "disc_corr": float(disc_corr),
            "conf_corr": conf_corr, "sign_agree": sign_agree,
        })

# Aggregate by feature
feat_agg = defaultdict(list)
for r in confirmation_results:
    feat_agg[r["feature"]].append(r)

# Fisher-z meta-analysis per feature
from scipy import stats
feature_level = []
for feat_j, results in feat_agg.items():
    conf_corrs = [r["conf_corr"] for r in results]
    sign_agreements = [r["sign_agree"] for r in results]
    mean_conf = float(np.mean(conf_corrs))
    mean_disc = float(np.mean([r["disc_corr"] for r in results]))
    sign_rate = float(np.mean(sign_agreements))
    n_stocks = len(results)

    # Fisher-z: test whether average confirmation correlation > 0
    z_vals = [np.arctanh(max(min(c, 0.999), -0.999)) for c in conf_corrs]
    mean_z = np.mean(z_vals)
    se_z = 1.0 / np.sqrt(n_stocks * np.mean([1.0 / max(len([x for x in results if x["ticker"] == t]), 3) for t in set(r["ticker"] for r in results)]))
    z_stat = mean_z / se_z
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    feature_level.append({
        "feature": int(feat_j),
        "label": results[0]["label"],
        "n_confirmation_stocks": n_stocks,
        "mean_disc_corr": mean_disc,
        "mean_conf_corr": mean_conf,
        "sign_agreement_rate": sign_rate,
        "p_value": float(p_val),
    })

# BH-FDR correction
p_vals = [f["p_value"] for f in feature_level]
n = len(p_vals)
sorted_idx = np.argsort(p_vals)
bh_thresholds = np.array([(i+1)/n*0.05 for i in range(n)])
significant = np.zeros(n, dtype=bool)
last_sig = -1
for i in range(n-1, -1, -1):
    if p_vals[sorted_idx[i]] <= bh_thresholds[i]:
        last_sig = i
        break
for i in range(last_sig+1):
    significant[sorted_idx[i]] = True

for i, f in enumerate(feature_level):
    f["bh_significant"] = bool(significant[i])

n_sig = sum(significant)
print(f"\nBH-FDR significant features: {n_sig}/{n} ({100*n_sig/max(n,1):.1f}%)")

# Concept-level aggregation
concept_conf = defaultdict(list)
for f in feature_level:
    concept_conf[f["label"]].append(f)

concept_summary = {}
for label, feats in concept_conf.items():
    sigs = sum(f["bh_significant"] for f in feats)
    concept_summary[label] = {
        "n_features": len(feats),
        "n_significant": sigs,
        "sign_rate": float(sigs / max(len(feats), 1)),
        "mean_conf_corr": float(np.mean([f["mean_conf_corr"] for f in feats])),
        "mean_sign_agreement": float(np.mean([f["sign_agreement_rate"] for f in feats])),
    }

final = {
    "n_discovery_stocks": len(disc_tickers),
    "n_confirmation_stocks": len(conf_tickers),
    "n_labeled_features": len(feature_label),
    "n_bh_significant": int(n_sig),
    "bh_significant_rate": float(n_sig / max(n, 1)),
    "overall_sign_agreement_rate": float(np.mean([f["sign_agreement_rate"] for f in feature_level])),
    "mean_confirmation_corr": float(np.mean([f["mean_conf_corr"] for f in feature_level])),
    "concept_summary": concept_summary,
}

with open(OUTPUT, "w") as f: json.dump(final, f, indent=2)
save_state()
print(f"\nDone in {time.time()-t0:.0f}s")
print(f"BH-FDR significant: {n_sig}/{n} ({100*n_sig/max(n,1):.1f}%)")
print(f"Overall sign agreement: {final['overall_sign_agreement_rate']:.4f}")
print(f"Mean confirmation corr: {final['mean_confirmation_corr']:.4f}")
