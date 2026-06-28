"""Per-sector SAE consistency."""
import torch, numpy as np, json, time, os, sys
from pathlib import Path; import pandas as pd
from collections import defaultdict
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/sector_consistency.json"
LAYER, EXPANSION, K, STEPS = 6, 2, 64, 3000

with open("/tmp/sectors120.json") as f:
    sector_map = json.load(f)
ticker_sector = {}
for sname, tickers in sector_map.items():
    for t in tickers: ticker_sector[t] = sname

np.random.seed(42)
all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])
sector_stocks = defaultdict(list)
for f in all_csvs:
    t = f.replace(".csv","")
    s = ticker_sector.get(t, "Other")
    if s != "Other" and len(sector_stocks[s]) < 15:
        sector_stocks[s].append(f)

print(f"Sector stocks: {[(s, len(v)) for s,v in sector_stocks.items()]}")

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

class SAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(d_model, d_hidden, bias=True)
        self.dec = torch.nn.Linear(d_hidden, d_model, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(d_model))
    def encode(self, x):
        xc = x - self.b; lat = self.enc(xc)
        _, idx = torch.topk(lat, K, dim=-1)
        m = torch.zeros_like(lat); m.scatter_(-1, idx, 1.0)
        return lat * m
    def decode(self, lat): return self.dec(lat) + self.b

label_names = ["momentum_5","trend","volatility","vol_persistence","autocorr_lag1",
               "autocorr_lag5","max_drawdown","var_95","max_1day_gain","max_1day_loss",
               "skewness","kurtosis","price_range","vol_clustering","volume_trend","volume_price_corr"]

def process_one(fname):
    df = pd.read_csv(DATA / fname)
    for c in ["open","close","high","low","volume","amount"]:
        if c not in df.columns: df[c] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 200: return None
    mn, st = data.mean(0), data.std(0)
    dn = np.clip((data-mn)/(st+1e-5), -5, 5)
    nw = min(1500, (len(dn)-64)//32)
    if nw < 50: return None
    wins = np.stack([dn[i:i+64] for i in range(0, nw*32, 32)])
    n_tr = int(len(wins)*0.8)

    acts = []
    def hook_fn(m,i,o):
        a = o[0] if isinstance(o,tuple) else o
        acts.append(a[:,-1,:].detach().cpu().float().numpy())
    hook = model.transformer[LAYER].register_forward_hook(hook_fn)
    with torch.no_grad():
        for b in range(0, n_tr, 64):
            batch = torch.from_numpy(wins[b:b+64]).float().to(device)
            s1,s2 = tok.encode(batch, half=True); model(s1,s2)
    hook.remove()
    acts = np.concatenate(acts)

    sae = SAE().to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
    at = torch.from_numpy(acts).float().to(device)
    for s in range(STEPS):
        idx = torch.randint(0, len(at), (256,))
        loss = torch.nn.functional.mse_loss(sae.decode(sae.encode(at[idx])), at[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

    all_labels = []
    for i in range(0, len(dn)-84, 32):
        c = dn[i:i+64, 1]; r = np.diff(c)/(c[:-1]+1e-5)
        feats = [c[-1]/c[-6]-1 if len(c)>=6 else 0, np.polyfit(np.arange(64),c,1)[0],
                 np.std(r), np.corrcoef(np.abs(r[1:]),np.abs(r[:-1]))[0,1] if len(r)>2 else 0,
                 np.corrcoef(r[1:],r[:-1])[0,1] if len(r)>2 else 0,
                 np.corrcoef(r[5:],r[:-5])[0,1] if len(r)>6 else 0,
                 np.min(c/np.maximum.accumulate(c)-1), np.percentile(r,5),
                 np.max(r), np.min(r), float(pd.Series(r).skew()) if len(r)>2 else 0,
                 float(pd.Series(r).kurtosis()) if len(r)>3 else 0,
                 (c.max()-c.min())/max(c.mean(),1e-5), np.mean(r**2)/(np.var(r)+1e-10),
                 np.mean(np.diff(dn[i:i+64,4])/(dn[i:i+63,4]+1e-5)),
                 np.corrcoef(r, np.diff(dn[i:i+64,4])[:len(r)]/(dn[i:i+63,4][:len(r)]+1e-5))[0,1] if len(r)>2 else 0]
        all_labels.append(feats)
    label_arr = np.array(all_labels)[:len(acts)]

    with torch.no_grad():
        lat = sae.encode(torch.from_numpy(acts).float().to(device)).cpu().numpy()

    type_dist = {}
    alive_mask = (lat != 0).sum(0) > 10
    for j in np.where(alive_mask)[0]:
        a = lat[:,j] != 0
        if a.sum() < 5: continue
        corrs = [abs(np.corrcoef(lat[a,j], label_arr[a,k])[0,1]) for k in range(len(label_names))]
        corrs = [0 if np.isnan(c) else c for c in corrs]; best = np.argmax(corrs)
        if corrs[best] > 0.15: type_dist[label_names[best]] = type_dist.get(label_names[best],0)+1

    total = sum(type_dist.values())
    largest = max(type_dist.values())/max(total,1) if type_dist else 0
    entropy_val = -sum((c/total)*np.log(c/total) for c in type_dist.values()) if total>0 else 0
    del sae; torch.cuda.empty_cache()
    return type_dist, largest, entropy_val

t0 = time.time()
results = {}
for sector in ["Bank","Energy","Tech","Consumer"]:
    files = sector_stocks.get(sector, [])[:15]
    print(f"\n{sector}: {len(files)} stocks")
    merged = defaultdict(int)
    largests = []; entropies = []

    for fi, fname in enumerate(files):
        r = process_one(fname)
        if r:
            type_dist, largest, entropy_val = r
            for k, v in type_dist.items(): merged[k] += v
            largests.append(largest); entropies.append(entropy_val)
        if (fi+1) % 5 == 0: print(f"  [{fi+1}/{len(files)}]")

    total = sum(merged.values())
    results[sector] = {
        "n_stocks": len(largests),
        "total": int(total),
        "mean_largest": float(np.mean(largests)),
        "mean_entropy": float(np.mean(entropies)),
        "top5": [(name, int(count), float(count/total*100)) for name, count in sorted(merged.items(), key=lambda x:-x[1])[:5]],
    }

print(f"\n=== PER-SECTOR ===")
for sector in ["Bank","Energy","Tech","Consumer"]:
    if sector not in results: continue
    r = results[sector]
    print(f"\n{sector} ({r['n_stocks']} stocks): largest={r['mean_largest']:.1%}, entropy={r['mean_entropy']:.3f}")
    for name, count, pct in r["top5"]:
        print(f"  {name:<25}: {pct:5.1f}%")

with open(OUTPUT, "w") as f: json.dump(results, f, indent=2)
print(f"\nSaved. {time.time()-t0:.0f}s")
