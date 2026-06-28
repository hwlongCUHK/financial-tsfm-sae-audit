"""Family-level out-of-sample decoding: test whether 6 coarse concept families
are decodable from SAE activations on chronologically held-out windows.
Uses per-stock ridge probes + baselines. 120 stocks.
"""
import torch, numpy as np, json, time, os
from pathlib import Path; import pandas as pd
from collections import defaultdict
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/family_decoding_120.json"
STATE = "/data/houwanlong/finllm-mi/outputs/sae/family_decoding_state.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1

# Coarse family definitions (indices into LABEL_NAMES)
FAMILIES = {
    "Momentum/Trend": [0, 1],          # momentum_5, trend
    "Volatility": [2, 3, 13],          # volatility, vol_persistence, vol_clustering
    "Autocorrelation": [4, 5],         # autocorr_lag1, autocorr_lag5
    "Tail Risk": [6, 7, 8, 9, 10, 11], # max_drawdown, var_95, max_gain, max_loss, skew, kurt
    "Price Structure": [6, 12],        # max_drawdown, price_range
    "Volume": [14, 15],                # volume_trend, volume_price_corr
}

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

def compute_all_labels(dn, n_wins):
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
    results = s.get("results",[])
else:
    completed = set()
    results = []

def save_state():
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump({"completed":list(completed),"results":results}, open(STATE,"w"))

all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])
rng = np.random.RandomState(42)
t0 = time.time()

for fi, fname in enumerate(all_csvs):
    ticker = fname.replace(".csv","")
    if ticker in completed: continue

    loaded = load_stock(fname)
    if loaded is None: completed.add(ticker); save_state(); continue
    wins, dn = loaded
    n_tr = int(len(wins)*TRAIN_SPLIT); n_val = int(len(wins)*VAL_SPLIT)
    n_test = len(wins)-n_tr-n_val
    if n_test < 10: completed.add(ticker); save_state(); continue

    train_acts = extract_acts(wins, n_tr)
    test_wins = wins[n_tr+n_val:]
    all_labels = compute_all_labels(dn, len(wins))
    test_labels = all_labels[n_tr+n_val:]
    m_test = min(len(test_labels), n_test)
    test_labels = test_labels[:m_test]
    test_acts_np = extract_acts(test_wins[:m_test], m_test)

    # Train SAE
    sae = TopKSAE(d_model, d_hidden, K).to(device)
    at_tr = torch.as_tensor(np.ascontiguousarray(train_acts), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    for _ in range(STEPS):
        idx = torch.randint(0, len(at_tr), (BATCH,))
        xr = sae.encode(at_tr[idx])
        loss = torch.nn.functional.mse_loss(sae.decode(xr), at_tr[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

    # Get SAE latents for train and test
    with torch.no_grad():
        lat_train = sae.encode(at_tr).detach().cpu().numpy()
        lat_test = sae.encode(torch.as_tensor(np.ascontiguousarray(test_acts_np), dtype=torch.float32, device=device)).detach().cpu().numpy()

    # Compute family targets: mean of member statistics
    train_targets = {}
    test_targets = {}
    for fname_family, indices in FAMILIES.items():
        valid_idx = [i for i in indices if i < all_labels.shape[1]]
        if not valid_idx: continue
        train_t = all_labels[:n_tr, valid_idx].mean(axis=1)
        test_t = test_labels[:, valid_idx].mean(axis=1)
        m = min(len(lat_train), len(train_t))
        train_targets[fname_family] = train_t[:m]
        m2 = min(len(lat_test), len(test_t))
        test_targets[fname_family] = test_t[:m2]

    # For each family, decode from SAE latents
    for family, train_t in train_targets.items():
        if len(train_t) < 10: continue
        test_t = test_targets.get(family)
        if test_t is None or len(test_t) < 5: continue

        # Align latents and targets
        m_tr = min(len(lat_train), len(train_t))
        X_tr = lat_train[:m_tr]
        y_tr = train_t[:m_tr]
        m_te = min(len(lat_test), len(test_t))
        X_te = lat_test[:m_te]
        y_te = test_t[:m_te]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Ridge probe
        probe = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        probe.fit(X_tr_s, y_tr)
        y_pred = probe.predict(X_te_s)

        # Held-out R²
        ss_res = np.sum((y_te - y_pred)**2)
        ss_tot = np.sum((y_te - np.mean(y_te))**2)
        r2 = float(1 - ss_res/max(ss_tot, 1e-8))
        pearson_r = float(np.corrcoef(y_te, y_pred)[0,1]) if len(y_te) > 1 else 0

        # Baseline: shuffle labels (break temporal structure)
        y_shuf = rng.permutation(y_tr)
        probe_shuf = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        probe_shuf.fit(X_tr_s, y_shuf)
        y_pred_shuf = probe_shuf.predict(X_te_s)
        ss_res_shuf = np.sum((y_te - y_pred_shuf)**2)
        r2_shuf = float(1 - ss_res_shuf/max(ss_tot, 1e-8))

        # Baseline: random SAE features (same dimensionality)
        X_rand = rng.randn(*X_tr_s.shape)
        X_rand_te = rng.randn(*X_te_s.shape)
        probe_rand = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        probe_rand.fit(X_rand, y_tr)
        r2_rand = float(1 - np.sum((y_te - probe_rand.predict(X_rand_te))**2)/max(ss_tot, 1e-8))

        results.append({
            "ticker": ticker, "family": family,
            "n_train": m_tr, "n_test": m_te,
            "r2": r2, "pearson_r": pearson_r,
            "r2_shuffled": r2_shuf, "r2_random": r2_rand,
        })

    completed.add(ticker)
    del sae; torch.cuda.empty_cache()
    if (fi+1) % 20 == 0:
        save_state()
        print(f"[{len(completed)}/{len(all_csvs)}] {ticker}")

# Aggregate by family
agg = {}
for family in FAMILIES:
    fam_results = [r for r in results if r["family"] == family]
    if not fam_results: continue
    r2s = [r["r2"] for r in fam_results]
    r2s_shuf = [r["r2_shuffled"] for r in fam_results]
    r2s_rand = [r["r2_random"] for r in fam_results]
    pearsons = [r["pearson_r"] for r in fam_results]
    agg[family] = {
        "n_stocks": len(fam_results),
        "r2_mean": float(np.mean(r2s)),
        "r2_median": float(np.median(r2s)),
        "r2_shuffled_mean": float(np.mean(r2s_shuf)),
        "r2_random_mean": float(np.mean(r2s_rand)),
        "pearson_r_mean": float(np.mean(pearsons)),
        "significant_stocks": int(sum(1 for r in fam_results if r["r2"] > r["r2_shuffled"] and r["r2"] > r["r2_random"])),
        "r2_ci": [float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))],
    }

overall_r2s = [r["r2"] for r in results]
final = {
    "n_stocks": len(completed),
    "n_tests": len(results),
    "overall_r2_mean": float(np.mean(overall_r2s)),
    "per_family": agg,
}

with open(OUTPUT, "w") as f: json.dump(final, f, indent=2)
save_state()
print(f"\nDone {len(completed)} stocks in {time.time()-t0:.0f}s")
for family, a in agg.items():
    print(f"  {family:20s}: R²={a['r2_mean']:.4f}, {a['significant_stocks']}/{a['n_stocks']} sig vs baselines")
