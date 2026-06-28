"""Rerun all key experiments on 120 stocks instead of 30.
Produces: financial_120.json, importance_matched_120.json
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
OUT_FIN = "/data/houwanlong/finllm-mi/outputs/sae/financial_120.json"
OUT_IMP = "/data/houwanlong/finllm-mi/outputs/sae/importance_matched_120.json"
STATE = "/data/houwanlong/finllm-mi/outputs/sae/scale_all_120_state.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1
N_CONCEPT_FEATURES, N_CONTROL_SAMPLES, PRE_CAL = 20, 10, 0.15

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
    close_raw = data[:, 1].copy()
    mn, st = data.mean(0), data.std(0)
    dn = np.clip((data-mn)/(st+1e-5), -5, 5)
    nw = min(2000, (len(dn)-WINDOW)//STRIDE)
    if nw < 25: return None, None
    wins = np.stack([dn[i:i+WINDOW] for i in range(0, nw*STRIDE, STRIDE)])
    return wins, dn, close_raw, st[1], mn[1]

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
            batch = to_tensor(wins[b:b+64].copy()).float().to(device)
            s1,s2 = tok.encode(batch, half=True); model(s1,s2)
    hook.remove()
    return np.concatenate(acts)

def train_sae(sae, train_acts):
    at = to_tensor(np.ascontiguousarray(train_acts)).float().to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    for _ in range(STEPS):
        idx = torch.randint(0, len(at), (BATCH,))
        xr = sae.encode(at[idx])
        loss = torch.nn.functional.mse_loss(sae.decode(xr), at[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

# Load state for resumption
if os.path.exists(STATE):
    with open(STATE) as f: state = json.load(f)
    completed = set(state.get("completed", []))
    fin_results = state.get("fin_results", [])
    imp_results = state.get("imp_results", [])
else:
    completed = set()
    fin_results = []
    imp_results = []

def save_state():
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump({"completed": list(completed), "fin_results": fin_results, "imp_results": imp_results}, f)

all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])
def to_tensor(arr):
    """Safely convert numpy to tensor, fixing negative strides."""
    arr = np.ascontiguousarray(arr)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)

rng = np.random.RandomState(42)
t0 = time.time()

for fi, fname in enumerate(all_csvs):
    ticker = fname.replace(".csv", "")
    if ticker in completed:
        continue

    loaded = load_stock(fname)
    if loaded is None: continue
    wins, dn, close_raw, close_std, close_mean = loaded
    n_tr = int(len(wins)*TRAIN_SPLIT)
    n_val = int(len(wins)*VAL_SPLIT)
    n_test = len(wins) - n_tr - n_val
    if n_test < 10:
        completed.add(ticker)
        save_state()
        continue

    train_acts = extract_acts(wins, n_tr)
    test_wins = wins[n_tr+n_val:]
    labels_arr = compute_labels(dn, len(wins))[n_tr+n_val:]
    min_len = min(len(labels_arr), n_test)
    labels_arr = labels_arr[:min_len]
    test_acts_np = extract_acts(test_wins[:min_len], min_len)

    sae = TopKSAE(d_model, d_hidden, K).to(device)
    train_sae(sae, train_acts)
    at_test = to_tensor(test_acts_np.copy()).float().to(device)
    lat_t = sae.encode(at_test)
    with torch.no_grad():
        lat_test = lat_t.cpu().numpy()
        recon_base = sae.decode(lat_t).cpu().numpy()

    # ---- Financial metrics ----
    top50 = np.argsort((lat_test != 0).sum(axis=0))[-50:][::-1].copy()
    with torch.no_grad():
        lat_ab50 = lat_t.clone().contiguous()
        lat_ab50[:, top50] = 0
        recon_ab50 = sae.decode(lat_ab50).cpu().numpy()

    base_ret = np.diff(recon_base.argmax(axis=-1).astype(float))
    ab_ret = np.diff(recon_ab50.argmax(axis=-1).astype(float))
    base_vol = np.std(base_ret) if len(base_ret) > 1 else 1.0
    ab_vol = np.std(ab_ret) if len(ab_ret) > 1 else 1.0
    dir_agree = np.mean(np.sign(base_ret[:min(len(base_ret),len(ab_ret))]) ==
                         np.sign(ab_ret[:min(len(base_ret),len(ab_ret))]))

    fin_results.append({
        "ticker": ticker, "n_test": min_len,
        "vol_ratio": float(ab_vol/(base_vol+1e-5)),
        "dir_agreement": float(dir_agree),
    })

    # ---- Importance-matched controls ----
    feature_concepts = {}
    feature_importance = {}
    for j in range(lat_test.shape[1]):
        a = lat_test[:,j] != 0
        if a.sum() < 5: continue
        corrs = [abs(np.corrcoef(lat_test[a,j], labels_arr[a,k])[0,1]) for k in range(len(LABEL_NAMES))]
        corrs = [0 if np.isnan(c) else c for c in corrs]
        best = int(np.argmax(corrs))
        if corrs[best] > PRE_CAL:
            feature_concepts[j] = LABEL_NAMES[best]
        feature_importance[j] = float(np.mean(np.abs(lat_test[a,j])))

    concept_features = defaultdict(list)
    for j, c in feature_concepts.items():
        concept_features[c].append(j)

    for concept, feat_list in concept_features.items():
        if len(feat_list) < 10: continue
        top_feats = sorted(feat_list, key=lambda j: feature_importance.get(j, 0), reverse=True)[:N_CONCEPT_FEATURES]
        concept_imps = [feature_importance.get(j, 0) for j in top_feats]
        concept_mean_imp = np.mean(concept_imps)

        with torch.no_grad():
            lat_c = lat_t.clone().contiguous()
            lat_c[:, top_feats] = 0
            rc_c = sae.decode(lat_c).cpu().numpy()
        c_ret = np.diff(rc_c.argmax(axis=-1).astype(float))
        c_vol = np.std(c_ret) if len(c_ret) > 1 else 1.0

        non_concept = [j for j in range(lat_test.shape[1]) if (lat_test[:,j] != 0).sum() >= 5 and j not in top_feats]
        freq_ctrl_vols = []
        imp_ctrl_vols = []

        for _ in range(N_CONTROL_SAMPLES):
            # Frequency-matched
            fc = []
            for ft in top_feats:
                tgt_freq = (lat_test[:,ft] != 0).mean()
                if non_concept:
                    best = min(non_concept, key=lambda j: abs((lat_test[:,j] != 0).mean() - tgt_freq))
                    fc.append(best)
            with torch.no_grad():
                lat_f = lat_t.clone().contiguous()
                lat_f[:, np.unique(fc)] = 0
                rf = sae.decode(lat_f).cpu().numpy()
            fr = np.diff(rf.argmax(axis=-1).astype(float))
            freq_ctrl_vols.append(np.std(fr) if len(fr) > 1 else 1.0)

            # Importance-matched
            ic = []
            for imp in concept_imps:
                if non_concept:
                    best = min(non_concept, key=lambda j: abs(feature_importance.get(j, 0) - imp))
                    ic.append(best)
            with torch.no_grad():
                lat_i = lat_t.clone().contiguous()
                lat_i[:, np.unique(ic)] = 0
                ri = sae.decode(lat_i).cpu().numpy()
            ir = np.diff(ri.argmax(axis=-1).astype(float))
            imp_ctrl_vols.append(np.std(ir) if len(ir) > 1 else 1.0)

        base_vol_c = np.std(np.diff(recon_base.argmax(axis=-1).astype(float)))
        imp_results.append({
            "ticker": ticker, "concept": concept,
            "concept_vol_ratio": float(c_vol / (base_vol_c + 1e-5)),
            "freq_matched_mean": float(np.mean(freq_ctrl_vols) / (base_vol_c + 1e-5)),
            "imp_matched_mean": float(np.mean(imp_ctrl_vols) / (base_vol_c + 1e-5)),
        })

    completed.add(ticker)
    del sae; torch.cuda.empty_cache()
    if (fi+1) % 10 == 0:
        save_state()
        print(f"[{len(completed)}] {ticker}: {len(feature_concepts)} labeled, {len(concept_features)} families, vol_ratio={fin_results[-1]['vol_ratio']:.3f}")

# Save final
save_state()

# Aggregate financial
n = len(fin_results)
fin_agg = {
    "n_stocks": n,
    "vol_ratio_mean": float(np.mean([r["vol_ratio"] for r in fin_results])),
    "vol_ratio_ci": [float(np.percentile([r["vol_ratio"] for r in fin_results], 2.5)),
                     float(np.percentile([r["vol_ratio"] for r in fin_results], 97.5))],
    "dir_agreement_mean": float(np.mean([r["dir_agreement"] for r in fin_results])),
}
with open(OUT_FIN, "w") as f: json.dump(fin_agg, f, indent=2)

# Aggregate importance-matched
imp_agg = {"n_stocks": n, "n_total_tests": len(imp_results)}
ratios = [r["concept_vol_ratio"] for r in imp_results]
freqs = [r["freq_matched_mean"] for r in imp_results]
imps = [r["imp_matched_mean"] for r in imp_results]
imp_agg["overall_concept"] = float(np.mean(ratios))
imp_agg["overall_freq"] = float(np.mean(freqs))
imp_agg["overall_imp"] = float(np.mean(imps))
imp_agg["delta_vs_freq"] = float(np.mean(ratios) - np.mean(freqs))
imp_agg["delta_vs_imp"] = float(np.mean(ratios) - np.mean(imps))
with open(OUT_IMP, "w") as f: json.dump(imp_agg, f, indent=2)

print(f"\nDone in {time.time()-t0:.0f}s")
print(f"Financial: vol={fin_agg['vol_ratio_mean']:.4f}, dir_agree={fin_agg['dir_agreement_mean']:.4f}")
print(f"Importance: concept={imp_agg['overall_concept']:.4f}, freq={imp_agg['overall_freq']:.4f}, imp={imp_agg['overall_imp']:.4f}")
