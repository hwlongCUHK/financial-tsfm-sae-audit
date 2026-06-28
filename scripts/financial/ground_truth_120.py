"""Ground-truth financial metrics on 120 stocks: directional accuracy,
RankIC, and volatility forecast error vs actual future returns.
"""
import torch, numpy as np, json, time, os
from pathlib import Path; import pandas as pd
from scipy.stats import spearmanr
import sys
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/ground_truth_120.json"
STATE = "/data/houwanlong/finllm-mi/outputs/sae/ground_truth_120_state.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1

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

# Load state
if os.path.exists(STATE):
    s = json.load(open(STATE))
    completed = set(s.get("completed",[]))
    results = s.get("results",[])
else:
    completed = set()
    results = []

def save():
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump({"completed":list(completed),"results":results}, open(STATE,"w"))

all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])
t0 = time.time()

for fi, fname in enumerate(all_csvs):
    ticker = fname.replace(".csv","")
    if ticker in completed: continue

    loaded = load_stock(fname)
    if loaded is None: completed.add(ticker); save(); continue
    wins, dn, close_raw, close_std, close_mean = loaded
    n_tr = int(len(wins)*TRAIN_SPLIT); n_val = int(len(wins)*VAL_SPLIT)
    n_test = len(wins)-n_tr-n_val
    if n_test < 10: completed.add(ticker); save(); continue

    train_acts = extract_acts(wins, n_tr)
    test_wins = wins[n_tr+n_val:]
    test_acts_np = extract_acts(test_wins[:n_test], n_test)

    sae = TopKSAE(d_model, d_hidden, K).to(device)
    train_sae(sae, train_acts)
    at_test = torch.as_tensor(np.ascontiguousarray(test_acts_np), dtype=torch.float32, device=device)
    with torch.no_grad():
        lat_t = sae.encode(at_test)
    top50 = np.argsort((lat_t.detach().cpu().numpy() != 0).sum(axis=0))[-50:][::-1].copy()

    with torch.no_grad():
        recon_base = sae.decode(lat_t).detach().cpu().numpy()
        lat_ab = lat_t.clone()
        lat_ab[:, top50] = 0
        recon_ab = sae.decode(lat_ab).detach().cpu().numpy()

    # Predicted token IDs (proxy for K-line pattern)
    base_preds = recon_base.argmax(axis=-1).astype(float)
    ab_preds = recon_ab.argmax(axis=-1).astype(float)

    # Actual future close prices after each test window
    test_start = (n_tr + n_val) * STRIDE + WINDOW
    actual_close = close_raw[test_start:test_start + n_test * STRIDE:STRIDE]
    actual_close = actual_close[:n_test]
    prev_close = close_raw[test_start - 1:test_start + (n_test-1) * STRIDE:STRIDE][:n_test]
    realized_ret = (actual_close - prev_close) / (prev_close + 1e-5)

    # Convert token preds to return proxies
    base_ret = np.diff(base_preds)
    ab_ret = np.diff(ab_preds)
    m = min(len(realized_ret), len(base_ret), len(ab_ret))
    realized_ret = realized_ret[:m]
    base_ret = base_ret[:m]
    ab_ret = ab_ret[:m]

    # Directional accuracy vs ground truth
    base_dir = np.mean(np.sign(base_ret) == np.sign(realized_ret)) if m > 0 else 0.5
    ab_dir = np.mean(np.sign(ab_ret) == np.sign(realized_ret)) if m > 0 else 0.5

    # RankIC vs realized
    base_ric = spearmanr(base_ret, realized_ret)[0] if m > 1 else 0
    ab_ric = spearmanr(ab_ret, realized_ret)[0] if m > 1 else 0

    # Vol forecast error
    realized_vol = np.std(realized_ret)
    base_ve = abs(np.std(base_ret) - realized_vol)
    ab_ve = abs(np.std(ab_ret) - realized_vol)

    results.append({"ticker": ticker, "n_test": m,
        "base_dir_acc": float(base_dir), "ab_dir_acc": float(ab_dir),
        "base_rankic": float(base_ric), "ab_rankic": float(ab_ric),
        "base_vol_err": float(base_ve), "ab_vol_err": float(ab_ve)})

    completed.add(ticker)
    del sae; torch.cuda.empty_cache()
    if (fi+1) % 20 == 0: save(); print(f"[{len(completed)}] {ticker}: base_dir={base_dir:.3f}")

# Aggregate
n = len(results)
agg = {
    "n_stocks": n,
    "base_dir_acc_mean": float(np.mean([r["base_dir_acc"] for r in results])),
    "ab_dir_acc_mean": float(np.mean([r["ab_dir_acc"] for r in results])),
    "dir_delta_mean": float(np.mean([r["ab_dir_acc"]-r["base_dir_acc"] for r in results])),
    "base_rankic_mean": float(np.mean([r["base_rankic"] for r in results])),
    "ab_rankic_mean": float(np.mean([r["ab_rankic"] for r in results])),
    "base_vol_err_mean": float(np.mean([r["base_vol_err"] for r in results])),
    "ab_vol_err_mean": float(np.mean([r["ab_vol_err"] for r in results])),
}
with open(OUTPUT, "w") as f: json.dump(agg, f, indent=2)
save()
print(f"\nDone {n} stocks in {time.time()-t0:.0f}s")
print(f"Dir: base={agg['base_dir_acc_mean']:.4f} ab={agg['ab_dir_acc_mean']:.4f} delta={agg['dir_delta_mean']:.4f}")
print(f"RankIC: base={agg['base_rankic_mean']:.4f} ab={agg['ab_rankic_mean']:.4f}")
print(f"Vol err: base={agg['base_vol_err_mean']:.1f} ab={agg['ab_vol_err_mean']:.1f}")
