"""Pooled null calibration on 30 stocks. Computes a single global threshold
by pooling all alive features across all stocks.
"""
import torch, numpy as np, json, time, os, sys
from pathlib import Path; import pandas as pd
from collections import defaultdict
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/pooled_calibration_30.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH_SIZE, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1
N_STOCKS = 120
N_SHUFFLES = 100
SEED = 42

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

print(f"Pooled null calibration on {N_STOCKS} stocks, {N_SHUFFLES} shuffles")

all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])
valid_stocks = []
for fname in all_csvs:
    if len(valid_stocks) >= N_STOCKS: break
    loaded = load_stock(fname)
    if loaded is None: continue
    wins, dn = loaded
    n_tr = int(len(wins)*TRAIN_SPLIT); n_val = int(len(wins)*VAL_SPLIT)
    n_test = len(wins)-n_tr-n_val
    if n_test < 10: continue
    valid_stocks.append((fname, wins, dn, n_tr, n_val, n_test))

print(f"Collected {len(valid_stocks)} valid stocks")

# Collect all features and labels (pooled across stocks)
all_lat_list = []
all_labels_list = []
all_corrs_list = []  # per-feature max |r| values

for fi, (fname, wins, dn, n_tr, n_val, n_test) in enumerate(valid_stocks):
    train_acts = extract_acts(wins, n_tr)
    sae = TopKSAE(d_model, d_hidden, K).to(device)
    train_sae(sae, train_acts)

    test_wins = wins[n_tr+n_val:]
    labels = compute_labels(dn, len(wins))[n_tr+n_val:]
    min_len = min(len(labels), n_test)
    labels = labels[:min_len]
    test_acts = extract_acts(test_wins[:min_len], min_len)

    at = torch.from_numpy(test_acts).float().to(device)
    with torch.no_grad():
        lat = sae.encode(at).cpu().numpy()

    # Collect alive features' correlations
    for j in range(lat.shape[1]):
        a = lat[:,j] != 0
        if a.sum() < 5: continue
        corrs = [abs(np.corrcoef(lat[a,j], labels[a,k])[0,1]) for k in range(len(LABEL_NAMES))]
        corrs = [0 if np.isnan(c) else c for c in corrs]
        all_lat_list.append(lat[a,j])
        all_labels_list.append(labels[a])
        all_corrs_list.append(corrs)

    del sae; torch.cuda.empty_cache()
    print(f"[{fi+1}/{len(valid_stocks)}] {fname}: {len(all_corrs_list)} total pooled features")

print(f"\nTotal pooled alive features: {len(all_corrs_list)}")

# Pooled null calibration: shuffle labels, compute max |r| across ALL pooled features
rng = np.random.RandomState(SEED)
null_maxes = []
for si in range(N_SHUFFLES):
    round_maxes = []
    for i in range(len(all_corrs_list)):
        shuf_lab = all_labels_list[i].copy()
        for c in range(shuf_lab.shape[1]):
            rng.shuffle(shuf_lab[:,c])
        corrs = [abs(np.corrcoef(all_lat_list[i], shuf_lab[:,k])[0,1]) for k in range(len(LABEL_NAMES))]
        round_maxes.append(max([0 if np.isnan(c) else c for c in corrs]))
    if round_maxes:
        null_maxes.append(max(round_maxes))
    if (si+1) % 20 == 0:
        print(f"  shuffle {si+1}/{N_SHUFFLES}")

null_95 = float(np.percentile(null_maxes, 95))
print(f"\nPooled null_95 threshold: {null_95:.4f}")

# Count surviving assignments at different thresholds
for threshold in [0.15, 0.25, null_95, 0.35, 0.40, 0.50]:
    type_dist = defaultdict(int)
    for corrs in all_corrs_list:
        best = int(np.argmax(corrs))
        if corrs[best] > threshold:
            type_dist[LABEL_NAMES[best]] += 1
    total = sum(type_dist.values())
    print(f"  threshold={threshold:.4f}: {total} surviving assignments, {len(type_dist)} families")
    if threshold == null_95:
        print(f"    Top families: {sorted(type_dist.items(), key=lambda x:-x[1])[:5]}")

result = {
    "n_stocks": len(valid_stocks),
    "n_pooled_features": len(all_corrs_list),
    "n_shuffles": N_SHUFFLES,
    "pooled_null_95": null_95,
    "per_stock_median_null_95": None,  # would need per-stock computation
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f: json.dump(result, f, indent=2)
print(f"\nSaved to {OUTPUT}")
