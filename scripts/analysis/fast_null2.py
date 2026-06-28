"""Fast null recalibration: separate thresholds for SAE, PCA, Random."""
import torch, numpy as np, json, time, os
from pathlib import Path; import pandas as pd
import sys
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file
from sklearn.decomposition import PCA

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUT = "/data/houwanlong/finllm-mi/outputs/sae/recalibrated_null.json"

tok = KronosTokenizer.from_pretrained("/data/houwanlong/models/Kronos-Tokenizer-base").to(device).eval()
with open("/data/houwanlong/models/Kronos-base/config.json") as f: cfg = json.load(f)
model = Kronos(s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"], n_layers=cfg["n_layers"],
               d_model=cfg["d_model"], n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
               ffn_dropout_p=cfg["ffn_dropout_p"], attn_dropout_p=cfg["attn_dropout_p"],
               resid_dropout_p=cfg["resid_dropout_p"], token_dropout_p=cfg["token_dropout_p"],
               learn_te=cfg["learn_te"])
sd = load_file("/data/houwanlong/models/Kronos-base/model.safetensors")
model.load_state_dict(sd, strict=False); model = model.to(device).half().eval()
d_model = cfg["d_model"]

class TopKSAE(torch.nn.Module):
    def __init__(self, d, h, k=64):
        super().__init__()
        self.enc = torch.nn.Linear(d, h, bias=True)
        self.dec = torch.nn.Linear(h, d, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(d))
        self.k = k

    def encode(self, x):
        xc = x - self.b
        lat = self.enc(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        m = torch.zeros_like(lat)
        m.scatter_(-1, idx, 1.0)
        return lat * m

    def decode(self, lat):
        return self.dec(lat) + self.b

csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])[:15]
print(f"Processing {len(csvs)} stocks...")

results = {"sae": {"null_95": [], "pct_above": [], "max_corr": []},
           "pca": {"null_95": [], "pct_above": [], "max_corr": []},
           "random": {"null_95": [], "pct_above": [], "max_corr": []}}

for fi, fname in enumerate(csvs):
    df = pd.read_csv(DATA / fname)
    for c in ["open","close","high","low","volume","amount"]:
        if c not in df.columns: df[c] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 200: continue
    mn, st = data.mean(0), data.std(0)
    dn = np.clip((data-mn)/(st+1e-5), -5, 5)
    nw = min(1000, (len(dn)-64)//32)
    if nw < 50: continue
    wins = np.stack([dn[i:i+64] for i in range(0, nw*32, 32)])
    n_tr = int(len(wins)*0.8)

    acts = []
    def hook_fn(m,i,o):
        a = o[0] if isinstance(o,tuple) else o
        acts.append(a[:,-1,:].detach().cpu().float().numpy())
    hook = model.transformer[6].register_forward_hook(hook_fn)
    with torch.no_grad():
        for b in range(0, n_tr, 64):
            batch = torch.from_numpy(wins[b:b+64]).float().to(device)
            s1,s2 = tok.encode(batch, half=True)
            model(s1,s2)
    hook.remove()
    acts = np.concatenate(acts)

    feats_list = []
    for i in range(0, len(dn)-84, 32):
        c = dn[i:i+64,1]; r = np.diff(c)/(c[:-1]+1e-5)
        feats_list.append([
            c[-1]/c[-6]-1 if len(c)>=6 else 0,
            np.polyfit(np.arange(64),c,1)[0],
            np.std(r),
            np.corrcoef(np.abs(r[1:]),np.abs(r[:-1]))[0,1] if len(r)>2 else 0,
            np.corrcoef(r[1:],r[:-1])[0,1] if len(r)>2 else 0,
            np.min(c/np.maximum.accumulate(c)-1),
            np.percentile(r,5),
            np.max(r),
            np.max(np.abs(r)),
            np.mean(r**2)/(np.var(r)+1e-10)])
    labels = np.array(feats_list)[:len(acts)]
    n_feats = labels.shape[1]

    sae = TopKSAE(d_model, d_model*2).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
    at = torch.from_numpy(acts).float().to(device)
    for s in range(3000):
        idx = torch.randint(0, len(at), (256,))
        xr = sae.encode(at[idx])
        loss = torch.nn.functional.mse_loss(sae.decode(xr), at[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

    with torch.no_grad():
        sae_feats = sae.encode(at[:len(acts)]).cpu().numpy()

    pca = PCA(n_components=min(64, d_model, len(acts)))
    pca_feats = pca.fit_transform(acts)

    np.random.seed(42)
    rand_basis = np.random.randn(d_model, 64)
    rand_basis /= np.linalg.norm(rand_basis, axis=0, keepdims=True)
    rand_feats = acts @ rand_basis

    for name, feats in [("sae", sae_feats), ("pca", pca_feats), ("random", rand_feats)]:
        max_corrs = []
        for j in range(feats.shape[1]):
            if name == "sae":
                a = feats[:,j] != 0
                if a.sum() < 5: continue
                corrs = [abs(np.corrcoef(feats[a,j], labels[a,k])[0,1]) for k in range(n_feats)]
            else:
                corrs = [abs(np.corrcoef(feats[:,j], labels[:,k])[0,1]) for k in range(n_feats)]
            max_corrs.append(max([0 if np.isnan(c) else c for c in corrs]))

        null_maxes = []
        shuf = labels.copy()
        for _ in range(50):
            for c in range(n_feats):
                np.random.shuffle(shuf[:,c])
            nm = []
            for j in range(feats.shape[1]):
                if name == "sae":
                    a = feats[:,j] != 0
                    if a.sum() < 5: continue
                    corrs = [abs(np.corrcoef(feats[a,j], shuf[a,k])[0,1]) for k in range(n_feats)]
                else:
                    corrs = [abs(np.corrcoef(feats[:,j], shuf[:,k])[0,1]) for k in range(n_feats)]
                nm.append(max([0 if np.isnan(c) else c for c in corrs]))
            if nm:
                null_maxes.append(max(nm))

        null_95 = np.percentile(null_maxes, 95) if null_maxes else 0
        pct_above = np.mean([1 if c > null_95 else 0 for c in max_corrs]) if max_corrs else 0
        max_c = max(max_corrs) if max_corrs else 0
        results[name]["null_95"].append(float(null_95))
        results[name]["pct_above"].append(float(pct_above))
        results[name]["max_corr"].append(float(max_c))

    del sae; torch.cuda.empty_cache()
    if (fi+1) % 5 == 0:
        print(f"[{fi+1}/{len(csvs)}]")

agg = {}
for name in ["sae","pca","random"]:
    agg[name] = {
        "null_95_mean": float(np.mean(results[name]["null_95"])),
        "pct_above_mean": float(np.mean(results[name]["pct_above"])),
        "max_corr_mean": float(np.mean(results[name]["max_corr"])),
    }
    print(f"{name}: null_95={agg[name]['null_95_mean']:.3f}, pct_above={agg[name]['pct_above_mean']:.1%}, max_corr={agg[name]['max_corr_mean']:.3f}")

agg["n_stocks"] = len(results["sae"]["null_95"])
with open(OUT, "w") as f:
    json.dump(agg, f, indent=2)
print(f"Saved to {OUT}")
