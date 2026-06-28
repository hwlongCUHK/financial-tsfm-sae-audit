"""Extract SAE activations for 10 stocks for visualization."""
import torch, numpy as np, json, os, sys
from pathlib import Path; import pandas as pd
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/activation_viz_10.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH_SIZE, LR = 4, 64, 3000, 256, 1e-4
STOCKS = ["sh600000.csv","sh600016.csv","sh600036.csv","sh600028.csv","sh600079.csv",
          "sh600088.csv","sh600132.csv","sh600157.csv","sh600184.csv","sh600196.csv"]

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

results = {}
for fname in STOCKS:
    print(f"Processing {fname}...")
    df = pd.read_csv(DATA / fname)
    for c in ["open","close","high","low","volume","amount"]:
        if c not in df.columns: df[c] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 100: continue
    close_raw = data[:, 1].copy()
    mn, st = data.mean(0), data.std(0)
    dn = np.clip((data-mn)/(st+1e-5), -5, 5)
    nw = min(200, (len(dn)-WINDOW)//STRIDE)
    if nw < 25: continue
    wins = np.stack([dn[i:i+WINDOW] for i in range(0, nw*STRIDE, STRIDE)])
    n_tr = int(len(wins)*0.6)

    acts = []
    def hook_fn(m,i,o):
        a = o[0] if isinstance(o,tuple) else o
        acts.append(a[:,-1,:].detach().cpu().float().numpy())
    hook = model.transformer[LAYER].register_forward_hook(hook_fn)
    with torch.no_grad():
        for b in range(0, len(wins), 64):
            batch = torch.from_numpy(wins[b:b+64]).float().to(device)
            s1,s2 = tok.encode(batch, half=True); model(s1,s2)
    hook.remove()
    all_acts = np.concatenate(acts)

    sae = TopKSAE(d_model, d_hidden, K).to(device)
    at = torch.from_numpy(all_acts[:n_tr]).float().to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    for s in range(STEPS):
        idx = torch.randint(0, len(at), (BATCH_SIZE,))
        xr = sae.encode(at[idx])
        loss = torch.nn.functional.mse_loss(sae.decode(xr), at[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

    with torch.no_grad():
        lat = sae.encode(torch.from_numpy(all_acts).float().to(device)).cpu().numpy()

    labels_arr = []
    for i in range(len(wins)):
        idx_start = i * STRIDE
        if idx_start + WINDOW > len(dn): break
        c = dn[idx_start:idx_start+WINDOW, 1]; r = np.diff(c)/(c[:-1]+1e-5)
        feats = [c[-1]/c[-6]-1 if len(c)>=6 else 0, np.polyfit(np.arange(WINDOW),c,1)[0],
                 np.std(r), np.corrcoef(np.abs(r[1:]),np.abs(r[:-1]))[0,1] if len(r)>2 else 0,
                 np.corrcoef(r[1:],r[:-1])[0,1] if len(r)>2 else 0,
                 np.corrcoef(r[5:],r[:-5])[0,1] if len(r)>6 else 0,
                 np.min(c/np.maximum.accumulate(c)-1), np.percentile(r,5),
                 np.max(r), np.min(r), float(pd.Series(r).skew()) if len(r)>2 else 0,
                 float(pd.Series(r).kurtosis()) if len(r)>3 else 0,
                 (c.max()-c.min())/max(c.mean(),1e-5), np.mean(r**2)/(np.var(r)+1e-10),
                 np.mean(np.diff(dn[idx_start:idx_start+WINDOW,4])/(dn[idx_start:idx_start+WINDOW-1,4]+1e-5)),
                 np.corrcoef(r, np.diff(dn[idx_start:idx_start+WINDOW,4])[:len(r)]/(dn[idx_start:idx_start+WINDOW-1,4][:len(r)]+1e-5))[0,1] if len(r)>2 else 0]
        labels_arr.append(feats)
    labels_arr = np.array(labels_arr)

    top20 = np.argsort((lat != 0).sum(axis=0))[-20:][::-1]
    feature_labels = {}
    for j in top20:
        a = lat[:,j] != 0
        if a.sum() < 5: feature_labels[int(j)] = "unknown"; continue
        corrs = [abs(np.corrcoef(lat[a,j], labels_arr[a,k])[0,1]) if k < labels_arr.shape[1] else 0 for k in range(len(LABEL_NAMES))]
        corrs = [0 if np.isnan(c) else c for c in corrs]
        best = np.argmax(corrs)
        feature_labels[int(j)] = LABEL_NAMES[best] if corrs[best] > 0.15 else "unlabeled"

    act_matrix = lat[:, top20].tolist()
    close_prices = close_raw[:len(wins)*STRIDE:STRIDE][:len(wins)].tolist()

    results[fname.replace(".csv","")] = {
        "n_windows": len(wins), "top20_features": [int(j) for j in top20],
        "feature_labels": {str(j): feature_labels[int(j)] for j in top20},
        "activation_matrix": act_matrix[:80], "close_prices": close_prices[:80],
    }
    del sae; torch.cuda.empty_cache()
    print(f"  Done: {len(wins)} windows")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f: json.dump(results, f)
print(f"\nSaved {len(results)} stocks to {OUTPUT}")
