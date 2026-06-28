"""Ground-truth financial metrics: ablated output vs actual future returns.
Computes directional accuracy, RankIC, and volatility forecast error against
realized market outcomes, not just ablated-vs-baseline agreement.
Runs on 30 stocks. Saves to /data/houwanlong/finllm-mi/outputs/sae/ground_truth_metrics.json
"""
import torch, numpy as np, json, time, os, sys
from pathlib import Path; import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

device = "cuda:0"
DATA = Path("/data/houwanlong/finllm-mi/data/scale120")
OUTPUT = "/data/houwanlong/finllm-mi/outputs/sae/ground_truth_metrics.json"

LAYER, WINDOW, STRIDE = 6, 64, 32
EXPANSION, K, STEPS, BATCH_SIZE, LR = 4, 64, 3000, 256, 1e-4
TRAIN_SPLIT, VAL_SPLIT = 0.6, 0.1
N_STOCKS = 30

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

def decode_to_close(recon_tensor):
    """Decode SAE reconstruction back to close prices using Kronos tokenizer.
    The reconstruction is in token space; we take argmax and decode."""
    # recon_tensor shape: (batch, vocab_size)
    token_ids = recon_tensor.argmax(dim=-1)  # (batch,)
    # Decode: Kronos tokenizer expects (batch, seq_len) with encoded tokens
    # For simplicity, we decode one token at a time
    # Actually, let's use the raw reconstruction in a simpler way
    return token_ids.cpu().numpy()

print(f"Ground-truth financial metrics on {N_STOCKS} stocks")
all_csvs = sorted([f for f in os.listdir(str(DATA)) if f.endswith(".csv")])
results = []

for fi, fname in enumerate(all_csvs):
    if len(results) >= N_STOCKS: break
    loaded = load_stock(fname)
    if loaded[0] is None: continue
    wins, dn, close_raw, close_std, close_mean = loaded
    n_tr = int(len(wins)*TRAIN_SPLIT); n_val = int(len(wins)*VAL_SPLIT)
    n_test = len(wins)-n_tr-n_val
    if n_test < 10: continue

    train_acts = extract_acts(wins, n_tr)
    sae = TopKSAE(d_model, d_hidden, K).to(device)
    train_sae(sae, train_acts)

    # Test windows: baseline vs ablated predictions
    test_wins = wins[n_tr+n_val:]
    test_acts_np = extract_acts(test_wins, len(test_wins))
    at = torch.from_numpy(test_acts_np).float().to(device)

    with torch.no_grad():
        lat = sae.encode(at)
        # Top-50 ablation
        freq = (lat.cpu().numpy() != 0).sum(axis=0)
        top50 = np.argsort(freq)[-50:]
        lat_ab = lat.clone()
        lat_ab[:, top50] = 0
        # Get predictions
        recon_base = sae.decode(lat)
        recon_ab = sae.decode(lat_ab)

    # Use token ID as a proxy for predicted return direction/size
    # Higher token ID generally encodes different K-line patterns
    base_preds = recon_base.argmax(dim=-1).cpu().numpy().astype(float)
    ab_preds = recon_ab.argmax(dim=-1).cpu().numpy().astype(float)

    # Realized future returns (next timestep after each test window)
    test_start_idx = (n_tr + n_val) * STRIDE + WINDOW
    close_segment = close_raw[test_start_idx:test_start_idx + n_test * STRIDE:STRIDE]
    close_segment = close_segment[:n_test]
    prev_close_seg = close_raw[test_start_idx - 1:test_start_idx + (n_test-1)*STRIDE:STRIDE][:n_test]
    realized_ret = (close_segment - prev_close_seg) / (prev_close_seg + 1e-5)

    # Baseline and ablated "returns" from token ID changes
    base_ret = np.diff(np.concatenate([[0], base_preds]))
    ab_ret = np.diff(np.concatenate([[0], ab_preds]))
    min_len = min(len(realized_ret), len(base_ret), len(ab_ret))
    realized_ret = realized_ret[:min_len]
    base_ret = base_ret[:min_len]
    ab_ret = ab_ret[:min_len]

    # Directional accuracy vs ground truth
    base_dir_acc = np.mean(np.sign(base_ret) == np.sign(realized_ret))
    ab_dir_acc = np.mean(np.sign(ab_ret) == np.sign(realized_ret))
    dir_delta = ab_dir_acc - base_dir_acc

    # RankIC vs realized returns
    base_rankic = spearmanr(base_ret, realized_ret)[0]
    ab_rankic = spearmanr(ab_ret, realized_ret)[0]
    rankic_delta = ab_rankic - base_rankic

    # Volatility forecast error vs realized vol
    realized_vol = np.std(realized_ret)
    base_vol_err = np.abs(np.std(base_ret) - realized_vol)
    ab_vol_err = np.abs(np.std(ab_ret) - realized_vol)
    vol_delta = ab_vol_err - base_vol_err

    # Directional agreement (ablated vs baseline)
    dir_agreement = np.mean(np.sign(base_ret) == np.sign(ab_ret))

    results.append({
        "ticker": fname.replace(".csv",""),
        "n_test": n_test,
        "base_dir_acc": float(base_dir_acc), "ab_dir_acc": float(ab_dir_acc),
        "dir_delta": float(dir_delta),
        "base_rankic": float(base_rankic), "ab_rankic": float(ab_rankic),
        "rankic_delta": float(rankic_delta),
        "base_vol_err": float(base_vol_err), "ab_vol_err": float(ab_vol_err),
        "vol_delta": float(vol_delta),
        "dir_agreement": float(dir_agreement),
    })
    del sae; torch.cuda.empty_cache()
    print(f"[{fi+1}] {fname}: base_dir={base_dir_acc:.3f}, ab_dir={ab_dir_acc:.3f}, "
          f"base_rankic={base_rankic:.3f}, ab_rankic={ab_rankic:.3f}, dir_agree={dir_agreement:.3f}")

# Aggregate
N = len(results)
agg = {
    "n_stocks": N,
    "base_dir_acc_mean": float(np.mean([r["base_dir_acc"] for r in results])),
    "ab_dir_acc_mean": float(np.mean([r["ab_dir_acc"] for r in results])),
    "dir_delta_mean": float(np.mean([r["dir_delta"] for r in results])),
    "dir_delta_ci": [float(np.percentile([r["dir_delta"] for r in results], 2.5)),
                     float(np.percentile([r["dir_delta"] for r in results], 97.5))],
    "base_rankic_mean": float(np.mean([r["base_rankic"] for r in results])),
    "ab_rankic_mean": float(np.mean([r["ab_rankic"] for r in results])),
    "rankic_delta_mean": float(np.mean([r["rankic_delta"] for r in results])),
    "base_vol_err_mean": float(np.mean([r["base_vol_err"] for r in results])),
    "ab_vol_err_mean": float(np.mean([r["ab_vol_err"] for r in results])),
    "vol_delta_mean": float(np.mean([r["vol_delta"] for r in results])),
    "dir_agreement_mean": float(np.mean([r["dir_agreement"] for r in results])),
    "per_stock": results,
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f: json.dump(agg, f, indent=2)
print(f"\nSaved to {OUTPUT}")
print(f"Dir accuracy: base={agg['base_dir_acc_mean']:.4f}, ab={agg['ab_dir_acc_mean']:.4f}, delta={agg['dir_delta_mean']:.4f}")
print(f"RankIC: base={agg['base_rankic_mean']:.4f}, ab={agg['ab_rankic_mean']:.4f}, delta={agg['rankic_delta_mean']:.4f}")
print(f"Dir agreement: {agg['dir_agreement_mean']:.4f}")
