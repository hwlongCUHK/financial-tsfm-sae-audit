"""Block-permutation null calibration: preserve temporal structure by shuffling
labels in blocks of 10 consecutive windows rather than independently.

Compares block-permuted null thresholds against standard random-shuffle thresholds
to quantify temporal autocorrelation inflation.
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
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/block_permutation.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH_SIZE, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1
N_STOCKS = 30
N_SHUFFLES = 50
BLOCK_SIZE = 10  # windows

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
    def __init__(self, d, h, k=64):
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

LABEL_NAMES = ["momentum_5","trend","volatility","vol_persistence","autocorr_lag1",
               "autocorr_lag5","max_drawdown","var_95","max_1day_gain","max_1day_loss",
               "skewness","kurtosis","price_range","vol_clustering","volume_trend","volume_price_corr"]

def compute_labels(dn, n_wins):
    all_labels = []
    for i in range(n_wins):
        idx = i * STRIDE
        if idx + WINDOW > len(dn): break
        c = dn[idx:idx+WINDOW, 1]; r = np.diff(c)/(c[:-1]+1e-5)
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

def block_shuffle(arr, block_size):
    """Shuffle arr by blocks of block_size consecutive rows."""
    n = len(arr)
    n_blocks = n // block_size
    blocks = [arr[i*block_size:(i+1)*block_size].copy() for i in range(n_blocks)]
    # Handle remainder
    remainder = arr[n_blocks*block_size:] if n_blocks*block_size < n else None
    rng = np.random.RandomState()
    indices = list(range(n_blocks))
    rng.shuffle(indices)
    shuffled = np.concatenate([blocks[i] for i in indices], axis=0)
    if remainder is not None:
        shuffled = np.concatenate([shuffled, remainder], axis=0)
    return shuffled

print(f"Processing {N_STOCKS} stocks with block-size={BLOCK_SIZE} permutation...")

all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])[:N_STOCKS]
results = {}

for fi, fname in enumerate(all_csvs):
    df = pd.read_csv(DATA / fname)
    for c in ["open","close","high","low","volume","amount"]:
        if c not in df.columns: df[c] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 200: continue
    mn, st = data.mean(0), data.std(0)
    dn = np.clip((data-mn)/(st+1e-5), -5, 5)
    nw = min(1500, (len(dn)-WINDOW)//STRIDE)
    if nw < 50: continue
    wins = np.stack([dn[i:i+WINDOW] for i in range(0, nw*STRIDE, STRIDE)])
    n_tr = int(len(wins)*TRAIN_SPLIT); n_val = int(len(wins)*VAL_SPLIT)
    n_test = len(wins) - n_tr - n_val

    # Extract activations
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

    # Labels (test windows only)
    labels = compute_labels(dn, len(wins))[n_tr+n_val:]
    if len(labels) < 20: continue

    # Train SAE
    sae = TopKSAE(d_model, d_hidden, K).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    at = torch.from_numpy(acts).float().to(device)
    for s in range(STEPS):
        idx = torch.randint(0, len(at), (BATCH_SIZE,))
        xr = sae.encode(at[idx])
        loss = torch.nn.functional.mse_loss(sae.decode(xr), at[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

    with torch.no_grad():
        lat_full = sae.encode(at).cpu().numpy()

    # Use only test-window portion of lat to match labels
    test_start = n_tr + n_val
    lat = lat_full[test_start:]  # align with labels
    if len(lat) < 20: continue

    # Null calibration: standard (random shuffle) vs block permutation
    stock_result = {"stock": fname, "n_test": len(labels),
                    "standard_null_95": None, "block_null_95": None}

    # Standard random-shuffle null
    null_maxes_std = []
    n_feats = labels.shape[1]
    rng = np.random.RandomState(42)
    for _ in range(N_SHUFFLES):
        shuf = labels.copy()
        for c in range(n_feats):
            rng.shuffle(shuf[:, c])
        nm = []
        for j in range(lat.shape[1]):
            a = lat[:, j] != 0
            if a.sum() < 5: continue
            corrs = [abs(np.corrcoef(lat[a,j], shuf[a,k])[0,1]) for k in range(n_feats)]
            nm.append(max([0 if np.isnan(c) else c for c in corrs]))
        if nm: null_maxes_std.append(max(nm))

    # Block-permutation null
    null_maxes_block = []
    for _ in range(N_SHUFFLES):
        shuf = block_shuffle(labels, BLOCK_SIZE)
        nm = []
        for j in range(lat.shape[1]):
            a = lat[:, j] != 0
            if a.sum() < 5: continue
            corrs = [abs(np.corrcoef(lat[a,j], shuf[a,k])[0,1]) for k in range(n_feats)]
            nm.append(max([0 if np.isnan(c) else c for c in corrs]))
        if nm: null_maxes_block.append(max(nm))

    stock_result["standard_null_95"] = float(np.percentile(null_maxes_std, 95)) if null_maxes_std else 0
    stock_result["block_null_95"] = float(np.percentile(null_maxes_block, 95)) if null_maxes_block else 0
    stock_result["standard_null_mean"] = float(np.mean(null_maxes_std)) if null_maxes_std else 0
    stock_result["block_null_mean"] = float(np.mean(null_maxes_block)) if null_maxes_block else 0
    stock_result["n_shuffles"] = N_SHUFFLES

    del sae; torch.cuda.empty_cache()
    results[fname] = stock_result
    print(f"[{fi+1}/{len(all_csvs)}] {fname}: std_null={stock_result['standard_null_95']:.4f}, block_null={stock_result['block_null_95']:.4f}")

# Aggregate
std_nulls = [r["standard_null_95"] for r in results.values()]
block_nulls = [r["block_null_95"] for r in results.values()]
inflation_factors = [b/s if s > 0 else 1.0 for b,s in zip(block_nulls, std_nulls)]

aggregate = {
    "n_stocks": len(results),
    "block_size": BLOCK_SIZE,
    "n_shuffles": N_SHUFFLES,
    "standard_null_95_mean": float(np.mean(std_nulls)),
    "block_null_95_mean": float(np.mean(block_nulls)),
    "standard_null_95_std": float(np.std(std_nulls)),
    "block_null_95_std": float(np.std(block_nulls)),
    "mean_inflation_factor": float(np.mean(inflation_factors)),
    "median_inflation_factor": float(np.median(inflation_factors)),
    "per_stock": results,
}

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(aggregate, f, indent=2)
print(f"\nSaved to {OUTPUT}")
print(f"Standard null_95 mean: {aggregate['standard_null_95_mean']:.4f}")
print(f"Block null_95 mean: {aggregate['block_null_95_mean']:.4f}")
print(f"Mean inflation factor: {aggregate['mean_inflation_factor']:.3f}")
print(f"Inflation < 1.0 means block perm is MORE conservative (narrower null)")
