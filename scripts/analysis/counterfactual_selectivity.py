"""Counterfactual selectivity: test whether concept-labeled SAE features
respond selectively to changes in the matching financial statistic.
Uses matched window pairs on 120 stocks.
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
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/counterfactual_selectivity_120.json"
STATE = "/data/houwanlong/finllm-mi/outputs/sae/counterfactual_state.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1
N_PAIRS_PER_CONCEPT = 20  # matched pairs per concept per stock
MATCHING_TOLERANCE = 0.15  # other stats must be within 15% of original

# Coarse concept groups with their constituent statistics
CONCEPT_GROUPS = {
    "Momentum": ["momentum_5", "trend"],
    "Volatility": ["volatility", "vol_persistence", "vol_clustering"],
    "Autocorrelation": ["autocorr_lag1", "autocorr_lag5"],
    "Tail Risk": ["var_95", "max_1day_gain", "max_1day_loss", "skewness", "kurtosis"],
    "Volume": ["volume_trend", "volume_price_corr"],
    "Price Structure": ["price_range", "max_drawdown"],
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

def train_sae(sae, acts):
    at = torch.as_tensor(np.ascontiguousarray(acts), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    for _ in range(STEPS):
        idx = torch.randint(0, len(at), (BATCH,))
        xr = sae.encode(at[idx])
        loss = torch.nn.functional.mse_loss(sae.decode(xr), at[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

# Load state for resumption
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
    test_wins = wins[n_tr+n_val:]
    all_labels = compute_labels(dn, len(wins))
    test_labels = all_labels[n_tr+n_val:][:min(n_test, len(all_labels)-(n_tr+n_val))]

    sae = TopKSAE(d_model, d_hidden, K).to(device)
    train_sae(sae, train_acts)

    # Label features on training set only (frozen labels)
    train_labels = all_labels[:n_tr]
    m_train = min(len(train_acts), len(train_labels))
    train_labels = train_labels[:m_train]
    train_acts_label = train_acts[:m_train]
    feature_concepts = {}
    lat_train = sae.encode(torch.as_tensor(np.ascontiguousarray(train_acts_label), dtype=torch.float32, device=device))
    lat_train_np = lat_train.detach().cpu().numpy()
    for j in range(lat_train_np.shape[1]):
        a = lat_train_np[:,j] != 0
        if a.sum() < 5: continue
        corrs = [abs(np.corrcoef(lat_train_np[a,j], train_labels[a,k])[0,1]) for k in range(len(LABEL_NAMES))]
        corrs = [0 if np.isnan(c) else c for c in corrs]
        best = int(np.argmax(corrs))
        if corrs[best] > 0.15:
            feature_concepts[j] = LABEL_NAMES[best]

    # Group features by coarse concept
    concept_features = defaultdict(list)
    for j, label in feature_concepts.items():
        for group, members in CONCEPT_GROUPS.items():
            if label in members:
                concept_features[group].append(j)
                break

    # Encode test windows with frozen SAE
    m_test = min(len(test_labels), n_test)
    test_labels = test_labels[:m_test]
    test_acts_np = extract_acts(test_wins[:m_test], m_test)
    at_test = torch.as_tensor(np.ascontiguousarray(test_acts_np), dtype=torch.float32, device=device)
    with torch.no_grad():
        lat_test = sae.encode(at_test).detach().cpu().numpy()

    # For each coarse concept, find matched pairs
    for group, members in CONCEPT_GROUPS.items():
        if group not in concept_features or len(concept_features[group]) < 3:
            continue

        # Compute group statistic (mean of member z-scores for test labels)
        member_indices = [LABEL_NAMES.index(m) for m in members if m in LABEL_NAMES]
        if not member_indices: continue
        group_stat = test_labels[:, member_indices].mean(axis=1)

        # For each other group, compute its statistic
        other_stats = {}
        for other_group, other_members in CONCEPT_GROUPS.items():
            if other_group == group: continue
            oi = [LABEL_NAMES.index(m) for m in other_members if m in LABEL_NAMES]
            if not oi: continue
            other_stats[other_group] = test_labels[:, oi].mean(axis=1)

        # Find high vs low pairs for this group
        med = np.median(group_stat)
        high_idx = np.where(group_stat > np.percentile(group_stat, 75))[0]
        low_idx = np.where(group_stat < np.percentile(group_stat, 25))[0]

        selectivity_deltas = defaultdict(list)

        for hi in high_idx[:N_PAIRS_PER_CONCEPT]:
            # Find best matching low window
            best_low = None
            best_dist = float('inf')
            for lo in low_idx:
                if lo == hi: continue
                # Distance on OTHER stats only
                dist = 0
                for og, os_arr in other_stats.items():
                    diff = abs(os_arr[hi] - os_arr[lo])
                    dist += diff
                if dist < best_dist:
                    best_dist = dist
                    best_low = lo
            if best_low is None: continue

            # Measure activation change for each concept family
            act_hi = lat_test[hi]
            act_lo = lat_test[best_low]
            for cg, feat_list in concept_features.items():
                if not feat_list: continue
                delta = np.mean(np.abs(act_hi[feat_list]) - np.abs(act_lo[feat_list]))
                selectivity_deltas[cg].append(float(delta))

        # Average across pairs
        for cg, deltas in selectivity_deltas.items():
            if deltas:
                all_results.append({
                    "ticker": ticker,
                    "target_concept": group,
                    "response_concept": cg,
                    "mean_delta": float(np.mean(deltas)),
                    "std_delta": float(np.std(deltas)),
                    "n_pairs": len(deltas),
                    "diagonal": (cg == group),
                })

    completed.add(ticker)
    del sae; torch.cuda.empty_cache()
    if (fi+1) % 20 == 0:
        save_state()
        print(f"[{len(completed)}] {ticker}: {len(concept_features)} labeled concepts, {len(concept_features)} groups")

# Aggregate: build selectivity matrix
selectivity_matrix = defaultdict(lambda: defaultdict(list))
for r in all_results:
    selectivity_matrix[r["target_concept"]][r["response_concept"]].append(r["mean_delta"])

matrix_agg = {}
for target, responses in selectivity_matrix.items():
    matrix_agg[target] = {}
    for resp, vals in responses.items():
        matrix_agg[target][resp] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

# Compute diagonal dominance
diagonal_vals = []
off_diagonal_vals = []
for target, responses in matrix_agg.items():
    for resp, v in responses.items():
        if target == resp:
            diagonal_vals.append(v["mean"])
        else:
            off_diagonal_vals.append(v["mean"])

final = {
    "n_stocks": len(completed),
    "n_tests": len(all_results),
    "selectivity_matrix": matrix_agg,
    "diagonal_mean": float(np.mean(diagonal_vals)) if diagonal_vals else 0,
    "off_diagonal_mean": float(np.mean(off_diagonal_vals)) if off_diagonal_vals else 0,
    "diagonal_ratio": float(np.mean(diagonal_vals)/np.mean(off_diagonal_vals)) if diagonal_vals and off_diagonal_vals else 0,
}

with open(OUTPUT, "w") as f: json.dump(final, f, indent=2)
save_state()
print(f"\nDone {len(completed)} stocks in {time.time()-t0:.0f}s")
print(f"Diagonal mean: {final['diagonal_mean']:.4f}")
print(f"Off-diagonal mean: {final['off_diagonal_mean']:.4f}")
print(f"Diagonal ratio: {final['diagonal_ratio']:.2f}x")
