"""Real-market event localization: test whether concept-labeled SAE features
selectively respond to naturally occurring financial events in held-out windows.
No counterfactual generation — uses only real K-line windows that the tokenizer accepts.
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
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/event_localization_120.json"
STATE = "/data/houwanlong/finllm-mi/outputs/sae/event_localization_state.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1

# Primary statistic for each concept family (pre-specified, no cherry-picking)
EVENT_DEFS = {
    "Momentum/Trend":  {"stat_idx": 0, "direction": "extremal", "name": "momentum_5"},
    "Volatility":      {"stat_idx": 2, "direction": "high", "name": "volatility"},
    "Autocorrelation": {"stat_idx": 4, "direction": "extremal", "name": "autocorr_lag1"},
    "Tail Risk":       {"stat_idx": 7, "direction": "low", "name": "var_95"},
    "Price Structure": {"stat_idx": 12, "direction": "high", "name": "price_range"},
    "Volume":          {"stat_idx": 14, "direction": "high", "name": "volume_trend"},
}

OTHER_STATS = {
    "Momentum/Trend":  [2, 4, 7, 12, 14],
    "Volatility":      [0, 4, 7, 12, 14],
    "Autocorrelation": [0, 2, 7, 12, 14],
    "Tail Risk":       [0, 2, 4, 12, 14],
    "Price Structure": [0, 2, 4, 7, 14],
    "Volume":          [0, 2, 4, 7, 12],
}

LABEL_NAMES = ["momentum_5","trend","volatility","vol_persistence","autocorr_lag1",
               "autocorr_lag5","max_drawdown","var_95","max_1day_gain","max_1day_loss",
               "skewness","kurtosis","price_range","vol_clustering","volume_trend","volume_price_corr"]

FAMILY_FEATURE_MAP = {
    "Momentum/Trend": [0, 1], "Volatility": [2, 3, 13], "Autocorrelation": [4, 5],
    "Tail Risk": [6, 7, 8, 9, 10, 11], "Price Structure": [6, 12], "Volume": [14, 15],
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
    if n_test < 20: completed.add(ticker); save_state(); continue

    train_acts = extract_acts(wins, n_tr)
    all_labels = compute_all_labels(dn, len(wins))
    test_labels = all_labels[n_tr+n_val:]
    m_test = min(len(test_labels), n_test)
    test_labels = test_labels[:m_test]
    test_acts_np = extract_acts(wins[n_tr+n_val:][:m_test], m_test)

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

    # Group features by family and compute training-set normalization
    family_features = defaultdict(list)
    family_train_act_mean = {}
    family_train_act_std = {}
    for fname_fam, indices in FAMILY_FEATURE_MAP.items():
        feats = [j for j, label in feature_concept.items() if LABEL_NAMES.index(label) in indices]
        if not feats: continue
        family_features[fname_fam] = feats
        vals = np.abs(lat_tr[:, feats]).mean(axis=1)
        family_train_act_mean[fname_fam] = float(vals.mean())
        family_train_act_std[fname_fam] = float(vals.std()) + 1e-8

    if len(family_features) < 4:  # need at least 4 families for meaningful comparison
        completed.add(ticker); save_state(); continue

    # Get test latents
    with torch.no_grad():
        lat_test = sae.encode(torch.as_tensor(np.ascontiguousarray(test_acts_np), dtype=torch.float32, device=device)).detach().cpu().numpy()

    # For each concept family, find event and control windows
    for event_family, ev_def in EVENT_DEFS.items():
        if event_family not in family_features or len(family_features[event_family]) < 3:
            continue

        stat_idx = ev_def["stat_idx"]
        direction = ev_def["direction"]
        if stat_idx >= test_labels.shape[1]:
            continue
        stat_vals = test_labels[:, stat_idx]

        # Define event windows
        n_test_wins = len(stat_vals)
        if n_test_wins < 20: continue

        if direction == "high":
            event_threshold = np.percentile(stat_vals, 90)
            event_mask = stat_vals >= event_threshold
        elif direction == "low":
            event_threshold = np.percentile(stat_vals, 10)
            event_mask = stat_vals <= event_threshold
        else:  # extremal
            event_threshold_high = np.percentile(stat_vals, 90)
            event_threshold_low = np.percentile(stat_vals, 10)
            event_mask = (stat_vals >= event_threshold_high) | (stat_vals <= event_threshold_low)

        control_mid = (np.percentile(stat_vals, 40) <= stat_vals) & (stat_vals <= np.percentile(stat_vals, 60))
        event_indices = np.where(event_mask)[0]
        control_pool = np.where(control_mid)[0]

        if len(event_indices) < 3 or len(control_pool) < 3: continue

        # For each event window, find best matching control on OTHER stats
        other_stats = OTHER_STATS.get(event_family, [])
        for ei in event_indices:
            if len(control_pool) < 1: break
            # Find closest control based on other stats
            dists = []
            for ci in control_pool:
                if ci == ei: continue
                d = 0
                for si in other_stats:
                    if si < test_labels.shape[1]:
                        d += abs(test_labels[ei, si] - test_labels[ci, si])
                dists.append((ci, d))
            if not dists: continue
            best_ci, _ = min(dists, key=lambda x: x[1])

            # Compute standardized family activation for event and control
            for resp_family, resp_features in family_features.items():
                event_act = np.mean(np.abs(lat_test[ei, resp_features]))
                ctrl_act = np.mean(np.abs(lat_test[best_ci, resp_features]))
                # Standardize
                mu = family_train_act_mean.get(resp_family, 0)
                sd = family_train_act_std.get(resp_family, 1)
                delta = (event_act - ctrl_act) / sd
                all_results.append({
                    "ticker": ticker, "event_family": event_family,
                    "response_family": resp_family,
                    "delta": float(delta), "diagonal": (event_family == resp_family),
                })

    completed.add(ticker)
    del sae; torch.cuda.empty_cache()
    if (fi+1) % 20 == 0: save_state(); print(f"[{len(completed)}] {ticker}: {len(family_features)} families")

# Aggregate
selectivity = defaultdict(lambda: defaultdict(list))
for r in all_results:
    selectivity[r["event_family"]][r["response_family"]].append(r["delta"])

matrix_agg = {}
for ev, responses in selectivity.items():
    matrix_agg[ev] = {}
    for resp, vals in responses.items():
        matrix_agg[ev][resp] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

diag_vals, off_vals = [], []
sign_test_pos = []
for ev, responses in matrix_agg.items():
    diag_delta = responses.get(ev, {"mean": 0})["mean"]
    for resp, v in responses.items():
        if ev == resp:
            diag_vals.append(v["mean"])
        else:
            off_vals.append(v["mean"])
            # Is diagonal higher than off-diagonal for this event?
            if diag_delta > v["mean"]:
                sign_test_pos.append(1)
            else:
                sign_test_pos.append(0)

from scipy.stats import binomtest
n_opp = len(sign_test_pos)
n_pos = sum(sign_test_pos)
p_sign = binomtest(n_pos, n_opp, 0.5).pvalue if n_opp > 0 else 1.0

final = {
    "n_stocks": len(completed), "n_events": len(all_results),
    "selectivity_matrix": matrix_agg,
    "diagonal_mean": float(np.mean(diag_vals)) if diag_vals else 0,
    "off_diagonal_mean": float(np.mean(off_vals)) if off_vals else 0,
    "diagonal_ratio": float(np.mean(diag_vals)/np.mean(off_vals)) if diag_vals and off_vals else 0,
    "sign_test_diag_wins": f"{n_pos}/{n_opp}",
    "sign_test_p": float(p_sign),
}

with open(OUTPUT, "w") as f: json.dump(final, f, indent=2)
save_state()
print(f"\nDone {len(completed)} stocks in {time.time()-t0:.0f}s")
print(f"Diagonal ratio: {final['diagonal_ratio']:.3f}x")
print(f"Sign test: {final['sign_test_diag_wins']} diagonal wins, p={final['sign_test_p']:.4f}")
