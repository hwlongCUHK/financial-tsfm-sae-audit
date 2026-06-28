"""Importance-matched control experiment: compares concept-specific ablation
against random features matched by activation magnitude (not just frequency).
Runs on 30 stocks. Saves to importance_matched_control.json
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
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/importance_matched_control.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH_SIZE, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1
N_STOCKS = 30
N_CONCEPT_FEATURES = 20
N_CONTROL_SAMPLES = 10  # resamples for random baselines
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

def train_sae(sae, train_acts):
    at = torch.from_numpy(train_acts).float().to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    for _ in range(STEPS):
        idx = torch.randint(0, len(at), (BATCH_SIZE,))
        xr = sae.encode(at[idx])
        loss = torch.nn.functional.mse_loss(sae.decode(xr), at[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

def volatility_ratio(rets):
    """std(ablated_returns) / std(baseline_returns)"""
    return float(np.std(rets))

# Collect results: per-stock, per-concept
all_results = []
rng = np.random.RandomState(42)

all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])
valid_count = 0

for fi, fname in enumerate(all_csvs):
    if valid_count >= N_STOCKS: break
    loaded = load_stock(fname)
    if loaded is None: continue
    wins, dn = loaded
    n_tr = int(len(wins)*TRAIN_SPLIT); n_val = int(len(wins)*VAL_SPLIT)
    n_test = len(wins)-n_tr-n_val
    if n_test < 10: continue
    valid_count += 1

    train_acts = extract_acts(wins, n_tr)
    test_wins = wins[n_tr+n_val:]
    labels_arr = compute_labels(dn, len(wins))[n_tr+n_val:]
    min_len = min(len(labels_arr), n_test)
    labels_arr = labels_arr[:min_len]
    test_acts = extract_acts(test_wins[:min_len], min_len)

    # Train SAE and get latents
    sae = TopKSAE(d_model, d_hidden, K).to(device)
    train_sae(sae, train_acts)
    at_test = torch.from_numpy(test_acts).float().to(device)
    with torch.no_grad():
        lat_test = sae.encode(at_test).cpu().numpy()

    # Label features
    feature_concepts = {}
    feature_importance = {}  # mean activation magnitude
    for j in range(lat_test.shape[1]):
        a = lat_test[:,j] != 0
        if a.sum() < 5: continue
        corrs = [abs(np.corrcoef(lat_test[a,j], labels_arr[a,k])[0,1]) for k in range(len(LABEL_NAMES))]
        corrs = [0 if np.isnan(c) else c for c in corrs]
        best = int(np.argmax(corrs))
        if corrs[best] > PRE_CAL_THRESHOLD:
            feature_concepts[j] = LABEL_NAMES[best]
        feature_importance[j] = float(np.mean(np.abs(lat_test[a,j])))

    # Group features by concept
    concept_features = defaultdict(list)
    for j, concept in feature_concepts.items():
        concept_features[concept].append(j)

    # For each concept with enough features, test ablation vs both baselines
    for concept, feat_list in concept_features.items():
        if len(feat_list) < 10: continue
        top_features = sorted(feat_list, key=lambda j: feature_importance.get(j, 0), reverse=True)[:N_CONCEPT_FEATURES]

        # Baseline: no ablation (just SAE reconstruction)
        with torch.no_grad():
            recon_base = sae.decode(torch.from_numpy(lat_test).float().to(device)).cpu().numpy()

        # Concept ablation
        with torch.no_grad():
            lat_ab = torch.from_numpy(lat_test.copy()).float().to(device)
            lat_ab[:, top_features] = 0
            recon_ab = sae.decode(lat_ab).cpu().numpy()

        # Volatility ratio for concept ablation
        base_ret = recon_base.argmax(axis=-1).astype(float)
        ab_ret = recon_ab.argmax(axis=-1).astype(float)
        base_vol = np.std(np.diff(base_ret) / (base_ret[:-1] + 1e-5))
        ab_vol = np.std(np.diff(ab_ret) / (ab_ret[:-1] + 1e-5))
        concept_vol_ratio = ab_vol / (base_vol + 1e-5)

        # Importance of concept features
        concept_importances = [feature_importance.get(j, 0) for j in top_features]
        concept_mean_imp = np.mean(concept_importances)
        concept_freqs = [(lat_test[:,j] != 0).mean() for j in top_features]
        concept_mean_freq = np.mean(concept_freqs)

        # Frequency-matched random (10 resamples)
        freq_ratios = []
        imp_ratios = []
        non_concept_features = [j for j in range(lat_test.shape[1])
                                if (lat_test[:,j] != 0).sum() >= 5 and j not in top_features]

        for _ in range(N_CONTROL_SAMPLES):
            # Frequency-matched
            freq_control = []
            for freq_target in concept_freqs:
                best_match = min(non_concept_features,
                    key=lambda j: abs((lat_test[:,j] != 0).mean() - freq_target))
                freq_control.append(best_match)
            with torch.no_grad():
                lat_fc = torch.from_numpy(lat_test.copy()).float().to(device)
                lat_fc[:, np.unique(freq_control)] = 0
                recon_fc = sae.decode(lat_fc).cpu().numpy()
            fc_ret = recon_fc.argmax(axis=-1).astype(float)
            fc_vol = np.std(np.diff(fc_ret) / (fc_ret[:-1] + 1e-5))
            freq_ratios.append(fc_vol / (base_vol + 1e-5))

            # Importance-matched (match by mean activation magnitude)
            imp_control = []
            for imp_target in concept_importances:
                best_match = min(non_concept_features,
                    key=lambda j: abs(feature_importance.get(j, 0) - imp_target))
                imp_control.append(best_match)
            with torch.no_grad():
                lat_ic = torch.from_numpy(lat_test.copy()).float().to(device)
                lat_ic[:, np.unique(imp_control)] = 0
                recon_ic = sae.decode(lat_ic).cpu().numpy()
            ic_ret = recon_ic.argmax(axis=-1).astype(float)
            ic_vol = np.std(np.diff(ic_ret) / (ic_ret[:-1] + 1e-5))
            imp_ratios.append(ic_vol / (base_vol + 1e-5))

        all_results.append({
            "ticker": fname.replace(".csv",""),
            "concept": concept,
            "concept_vol_ratio": concept_vol_ratio,
            "freq_matched_mean": float(np.mean(freq_ratios)),
            "freq_matched_std": float(np.std(freq_ratios)),
            "imp_matched_mean": float(np.mean(imp_ratios)),
            "imp_matched_std": float(np.std(imp_ratios)),
            "concept_mean_imp": concept_mean_imp,
            "concept_mean_freq": concept_mean_freq,
        })

    del sae; torch.cuda.empty_cache()
    print(f"[{valid_count}/{N_STOCKS}] {fname}: {len(feature_concepts)} labeled features, {len(concept_features)} concept families")

# Aggregate
agg = {}
for concept in set(r["concept"] for r in all_results):
    cr = [r for r in all_results if r["concept"] == concept]
    vol_ratios = [r["concept_vol_ratio"] for r in cr]
    freq_means = [r["freq_matched_mean"] for r in cr]
    imp_means = [r["imp_matched_mean"] for r in cr]
    agg[concept] = {
        "n_stocks": len(cr),
        "concept_vol_ratio_mean": float(np.mean(vol_ratios)),
        "freq_matched_vol_ratio_mean": float(np.mean(freq_means)),
        "imp_matched_vol_ratio_mean": float(np.mean(imp_means)),
        "delta_vs_freq": float(np.mean(vol_ratios) - np.mean(freq_means)),
        "delta_vs_imp": float(np.mean(vol_ratios) - np.mean(imp_means)),
    }

overall_vol = [r["concept_vol_ratio"] for r in all_results]
overall_freq = [r["freq_matched_mean"] for r in all_results]
overall_imp = [r["imp_matched_mean"] for r in all_results]

final = {
    "n_stocks": valid_count,
    "n_total_concept_tests": len(all_results),
    "overall_concept_vol_ratio_mean": float(np.mean(overall_vol)),
    "overall_freq_matched_mean": float(np.mean(overall_freq)),
    "overall_imp_matched_mean": float(np.mean(overall_imp)),
    "delta_concept_vs_freq": float(np.mean(overall_vol) - np.mean(overall_freq)),
    "delta_concept_vs_imp": float(np.mean(overall_vol) - np.mean(overall_imp)),
    "per_concept": agg,
    "per_stock_concept": all_results,
}

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f: json.dump(final, f, indent=2)
print(f"\nSaved to {OUTPUT}")
print(f"Overall concept vol ratio: {final['overall_concept_vol_ratio_mean']:.4f}")
print(f"Freq-matched control:     {final['overall_freq_matched_mean']:.4f}")
print(f"Imp-matched control:      {final['overall_imp_matched_mean']:.4f}")
print(f"Delta vs freq: {final['delta_concept_vs_freq']:.4f}")
print(f"Delta vs imp:  {final['delta_concept_vs_imp']:.4f}")
