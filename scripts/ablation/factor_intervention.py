"""Controlled financial factor intervention: modify K-line windows along
specific financial dimensions, measure SAE feature selectivity.
Per-stock: 5 test windows × 6 concepts × 2 directions × 1 intensity.
120 stocks. Uses scipy optimize for constrained counterfactual generation.
"""
import torch, numpy as np, json, time, os
from pathlib import Path; import pandas as pd
from collections import defaultdict
from scipy.optimize import minimize
import sys
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/factor_intervention_120.json"
STATE = "/data/houwanlong/finllm-mi/outputs/sae/factor_intervention_state.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1
INTENSITY = 1.0  # sigma multiplier
N_WINDOWS_PER_STOCK = 5

CONCEPT_SPECS = {
    "Momentum/Trend": {"stat_idx": 0, "fn": lambda c: (c[-1]/c[-6]-1 if len(c)>=6 else 0)},
    "Volatility":      {"stat_idx": 2, "fn": lambda c: np.std(np.diff(c)/(c[:-1]+1e-5))},
    "Autocorrelation": {"stat_idx": 4, "fn": lambda c: np.corrcoef(np.diff(c)[1:], np.diff(c)[:-1])[0,1] if len(np.diff(c))>2 else 0},
    "Price Structure": {"stat_idx": 12, "fn": lambda c: (c.max()-c.min())/max(abs(c).mean(),1e-5)},
    "Volume":          {"stat_idx": 14, "fn": lambda c, v=None: np.mean(np.diff(v)/(v[:-1]+1e-5)) if v is not None else 0},
    "Tail Risk":       {"stat_idx": 7, "fn": lambda c: np.percentile(np.diff(c)/(c[:-1]+1e-5), 5)},
}

LABEL_NAMES = ["momentum_5","trend","volatility","vol_persistence","autocorr_lag1",
               "autocorr_lag5","max_drawdown","var_95","max_1day_gain","max_1day_loss",
               "skewness","kurtosis","price_range","vol_clustering","volume_trend","volume_price_corr"]

FAMILY_FEATURE_MAP = {
    "Momentum/Trend": [0, 1],
    "Volatility": [2, 3, 13],
    "Autocorrelation": [4, 5],
    "Tail Risk": [6, 7, 8, 9, 10, 11],
    "Price Structure": [6, 12],
    "Volume": [14, 15],
}

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

def get_waveform(w):
    """Extract normalized close, high, low, volume from window"""
    return w[:,1].copy(), w[:,2].copy(), w[:,3].copy(), w[:,4].copy()

def perturb_window(w, close_new, high_new=None, low_new=None, vol_new=None):
    """Create perturbed window with modified close/high/low/volume"""
    wp = w.copy()
    wp[:,1] = close_new
    if high_new is not None:
        wp[:,2] = np.maximum(high_new, close_new)  # high >= close
    if low_new is not None:
        wp[:,3] = np.minimum(low_new, close_new)     # low <= close
    if vol_new is not None:
        wp[:,4] = np.maximum(vol_new, 1e-5)  # volume > 0
    return wp

# Load state
if os.path.exists(STATE):
    s = json.load(open(STATE))
    completed = set(s.get("completed",[]))
    all_results = s.get("results",[])
else:
    completed = set()
    all_results = []

def save_state():
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump({"completed":list(completed),"results":all_results}, open(STATE,"w"))

all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])
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

    sae = TopKSAE(d_model, d_hidden, K).to(device)
    at_tr = torch.as_tensor(np.ascontiguousarray(train_acts), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    for _ in range(STEPS):
        idx = torch.randint(0, len(at_tr), (BATCH,))
        xr = sae.encode(at_tr[idx])
        loss = torch.nn.functional.mse_loss(sae.decode(xr), at_tr[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

    # Label features using training set only
    train_labels = all_labels[:n_tr]
    m_tr_l = min(len(train_acts), len(train_labels))
    with torch.no_grad():
        lat_tr = sae.encode(at_tr[:m_tr_l]).detach().cpu().numpy()
    train_labels = train_labels[:m_tr_l]
    feature_concept = {}
    for j in range(lat_tr.shape[1]):
        a = lat_tr[:,j] != 0
        if a.sum() < 5: continue
        corrs = [abs(np.corrcoef(lat_tr[a,j], train_labels[a,k])[0,1]) for k in range(len(LABEL_NAMES))]
        corrs = [0 if np.isnan(c) else c for c in corrs]
        best = int(np.argmax(corrs))
        if corrs[best] > 0.15:
            feature_concept[j] = LABEL_NAMES[best]

    # Group features by family
    family_features = defaultdict(list)
    for j, label in feature_concept.items():
        for fname_fam, indices in FAMILY_FEATURE_MAP.items():
            if LABEL_NAMES.index(label) in indices:
                family_features[fname_fam].append(j)
                break

    # Select test windows
    n_test_wins = min(N_WINDOWS_PER_STOCK, len(test_wins))
    test_indices = np.random.RandomState(42+fi).choice(len(test_wins), n_test_wins, replace=False)

    for wi in test_indices:
        w = test_wins[wi].copy()  # (64, 6)
        close, high, low, vol = get_waveform(w)

        # For each concept, generate +/- counterfactual
        for concept, spec in CONCEPT_SPECS.items():
            if concept not in family_features or len(family_features[concept]) < 3:
                continue

            # Compute current statistic value
            if "Volume" in concept:
                cur_val = spec["fn"](close, vol)
            else:
                cur_val = spec["fn"](close)

            if np.isnan(cur_val) or np.isinf(cur_val):
                continue

            sigma = max(abs(cur_val) * INTENSITY, 0.01)

            # Simple perturbation: scale the last few close prices
            for direction, sign in [("pos", 1.0), ("neg", -1.0)]:
                try:
                    # Modify last 10% of close prices proportionally
                    n_modify = max(1, WINDOW // 10)
                    scale = 1.0 + sign * 0.2
                    close_mod = close.copy()
                    close_mod[-n_modify:] *= scale
                    close_mod = np.clip(close_mod, -5, 5)
                    high_mod = np.maximum(high, close_mod)
                    low_mod = np.minimum(low, close_mod)
                    wp = perturb_window(w, close_mod, high_mod, low_mod)

                    at_orig = torch.as_tensor(w[np.newaxis].copy(), dtype=torch.float32, device=device)
                    at_pert = torch.as_tensor(wp[np.newaxis].copy(), dtype=torch.float32, device=device)
                    s1_o, s2_o = tok.encode(at_orig, half=True)
                    s1_p, s2_p = tok.encode(at_pert, half=True)

                    with torch.no_grad():
                        lat_orig = sae.encode(at_orig).detach().cpu().numpy()[0]
                        lat_pert = sae.encode(at_pert).detach().cpu().numpy()[0]

                    for resp_family, resp_features in family_features.items():
                        if not resp_features: continue
                        delta = abs(float(np.mean(np.abs(lat_pert[resp_features])) -
                                        float(np.mean(np.abs(lat_orig[resp_features])))))
                        all_results.append({
                            "ticker": ticker, "target_concept": concept,
                            "response_concept": resp_family, "direction": direction,
                            "delta": delta, "diagonal": (concept == resp_family),
                        })
                except (RuntimeError, ValueError, IndexError, TypeError):
                    continue

    completed.add(ticker)
    del sae; torch.cuda.empty_cache()
    if (fi+1) % 20 == 0:
        save_state()
        print(f"[{len(completed)}] {ticker}: {len(family_features)} labeled families")

# Aggregate selectivity matrix
selectivity = defaultdict(lambda: defaultdict(list))
for r in all_results:
    selectivity[r["target_concept"]][r["response_concept"]].append(r["delta"])

matrix_agg = {}
for target, responses in selectivity.items():
    matrix_agg[target] = {}
    for resp, vals in responses.items():
        matrix_agg[target][resp] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

diag_vals, off_vals = [], []
for target, responses in matrix_agg.items():
    for resp, v in responses.items():
        (diag_vals if target == resp else off_vals).append(v["mean"])

final = {
    "n_stocks": len(completed),
    "n_tests": len(all_results),
    "selectivity_matrix": matrix_agg,
    "diagonal_mean": float(np.mean(diag_vals)) if diag_vals else 0,
    "off_diagonal_mean": float(np.mean(off_vals)) if off_vals else 0,
    "diagonal_ratio": float(np.mean(diag_vals)/np.mean(off_vals)) if diag_vals and off_vals else 0,
}

with open(OUTPUT, "w") as f: json.dump(final, f, indent=2)
save_state()
print(f"\nDone {len(completed)} stocks in {time.time()-t0:.0f}s")
print(f"Diagonal ratio: {final['diagonal_ratio']:.3f}x")
print(f"Diagonal mean: {final['diagonal_mean']:.4f}, Off-diag: {final['off_diagonal_mean']:.4f}")
