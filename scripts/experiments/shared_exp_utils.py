"""Shared utilities for all 7 advanced experiments on Kronos SAE interpretability.
Self-contained: loads model, tokenizer, data, SAE, computes statistics.
"""
import torch, numpy as np, json, os, time
from pathlib import Path
import pandas as pd
from collections import defaultdict

DEVICE = "cuda:0"
DATA_DIR = "/data/houwanlong/finllm-mi/data/scale120"
MODEL_DIR = "/data/houwanlong/models"
OUTPUT_DIR = "/data/houwanlong/finllm-mi/outputs/sae"

# Model / SAE hyperparams
LAYER = 6
WINDOW = 64
STRIDE = 32
EXPANSION = 4
K = 64
SAE_STEPS = 3000
SAE_BATCH = 256
SAE_LR = 1e-4
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.1

# Concept families (coarse, 6 families)
FAMILIES = {
    "Momentum/Trend": [0, 1],
    "Volatility": [2, 3, 13],
    "Autocorrelation": [4, 5],
    "Tail Risk": [6, 7, 8, 9, 10, 11],
    "Price Structure": [6, 12],
    "Volume": [14, 15],
}

LABEL_NAMES = ["momentum_5","trend","volatility","vol_persistence","autocorr_lag1",
               "autocorr_lag5","max_drawdown","var_95","max_1day_gain","max_1day_loss",
               "skewness","kurtosis","price_range","vol_clustering","volume_trend","volume_price_corr"]

# Sectors (by ticker prefix pattern)
SECTOR_MAP = {}  # populated from ticker patterns: sh60xxxx = Bank/Finance, etc.

_g_model = None
_g_tokenizer = None
_g_cfg = None
_g_d_model = None
_g_d_hidden = None


def get_model():
    global _g_model, _g_tokenizer, _g_cfg, _g_d_model, _g_d_hidden
    if _g_model is not None:
        return _g_model, _g_tokenizer, _g_cfg, _g_d_model, _g_d_hidden

    import sys
    sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
    from model.kronos import Kronos, KronosTokenizer
    from safetensors.torch import load_file

    _g_tokenizer = KronosTokenizer.from_pretrained(f"{MODEL_DIR}/Kronos-Tokenizer-base").to(DEVICE).eval()
    with open(f"{MODEL_DIR}/Kronos-base/config.json") as f:
        _g_cfg = json.load(f)
    _g_model = Kronos(s1_bits=_g_cfg["s1_bits"], s2_bits=_g_cfg["s2_bits"],
                       n_layers=_g_cfg["n_layers"], d_model=_g_cfg["d_model"],
                       n_heads=_g_cfg["n_heads"], ff_dim=_g_cfg["ff_dim"],
                       ffn_dropout_p=_g_cfg["ffn_dropout_p"],
                       attn_dropout_p=_g_cfg["attn_dropout_p"],
                       resid_dropout_p=_g_cfg["resid_dropout_p"],
                       token_dropout_p=_g_cfg["token_dropout_p"],
                       learn_te=_g_cfg["learn_te"])
    sd = load_file(f"{MODEL_DIR}/Kronos-base/model.safetensors")
    _g_model.load_state_dict(sd, strict=False)
    _g_model = _g_model.to(DEVICE).half().eval()
    _g_d_model = _g_cfg["d_model"]
    _g_d_hidden = _g_d_model * EXPANSION
    return _g_model, _g_tokenizer, _g_cfg, _g_d_model, _g_d_hidden


class TopKSAE(torch.nn.Module):
    def __init__(self, d, h, k):
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


def load_stock(fname):
    df = pd.read_csv(Path(DATA_DIR) / fname)
    for c in ["open","close","high","low","volume","amount"]:
        if c not in df.columns:
            df[c] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 100:
        return None, None
    mn, st = data.mean(0), data.std(0)
    dn = np.clip((data - mn) / (st + 1e-5), -5, 5)
    nw = min(2000, (len(dn) - WINDOW) // STRIDE)
    if nw < 25:
        return None, None
    wins = np.stack([dn[i:i+WINDOW] for i in range(0, nw * STRIDE, STRIDE)])
    return wins, dn


def compute_statistics_window(close_series, open_series=None, high_series=None,
                               low_series=None, volume_series=None):
    """Compute all 16 statistics for a single window."""
    c = np.asarray(close_series, dtype=np.float64)
    r = np.diff(c) / (c[:-1] + 1e-5)

    feats = [
        c[-1] / c[-6] - 1 if len(c) >= 6 else 0.0,                    # 0: momentum_5
        np.polyfit(np.arange(len(c)), c, 1)[0],                         # 1: trend
        np.std(r),                                                       # 2: volatility
        np.corrcoef(np.abs(r[1:]), np.abs(r[:-1]))[0,1] if len(r)>2 else 0.0,  # 3: vol_persistence
        np.corrcoef(r[1:], r[:-1])[0,1] if len(r)>2 else 0.0,          # 4: autocorr_lag1
        np.corrcoef(r[5:], r[:-5])[0,1] if len(r)>6 else 0.0,          # 5: autocorr_lag5
        np.min(c / np.maximum.accumulate(c) - 1),                       # 6: max_drawdown
        np.percentile(r, 5),                                             # 7: var_95
        np.max(r),                                                       # 8: max_1day_gain
        np.min(r),                                                       # 9: max_1day_loss
        float(pd.Series(r).skew()) if len(r)>2 else 0.0,               # 10: skewness
        float(pd.Series(r).kurtosis()) if len(r)>3 else 0.0,           # 11: kurtosis
        (c.max()-c.min()) / max(c.mean(), 1e-5),                        # 12: price_range
        np.mean(r**2) / (np.var(r) + 1e-10),                            # 13: vol_clustering
    ]

    # Volume stats
    if volume_series is not None:
        v = np.asarray(volume_series, dtype=np.float64)
        dv = np.diff(v) / (v[:-1] + 1e-5)
        feats.append(np.mean(dv))                                        # 14: volume_trend
        if len(r) > 2:
            min_len = min(len(r), len(dv))
            feats.append(np.corrcoef(r[:min_len], dv[:min_len])[0,1])  # 15: volume_price_corr
        else:
            feats.append(0.0)
    else:
        feats.extend([0.0, 0.0])

    return np.array(feats, dtype=np.float32)


def compute_all_labels(dn, n_wins):
    """Compute 16 statistics for each window."""
    all_labels = []
    for i in range(n_wins):
        idx = i * STRIDE
        if idx + WINDOW > len(dn):
            break
        c = dn[idx:idx+WINDOW, 1]
        v = dn[idx:idx+WINDOW, 4] if dn.shape[1] > 4 else None
        feats = compute_statistics_window(c, volume_series=v)
        all_labels.append(feats)
    return np.array(all_labels)


def compute_rolling_statistics(dn, window_positions):
    """Compute statistics using data up to each position (for temporal alignment)."""
    all_rolling = []
    for pos in range(len(window_positions)):
        # Use data from position 0 to pos
        c = dn[:pos+1, 1] if pos+1 <= len(dn) else dn[:, 1]
        v = dn[:pos+1, 4] if (pos+1 <= len(dn) and dn.shape[1] > 4) else None
        feats = compute_statistics_window(c, volume_series=v)
        all_rolling.append(feats)
    return np.array(all_rolling)


def extract_acts(wins, n_win, layer=None):
    """Extract activations at given layer. Returns (n_wins, d_model)."""
    if layer is None:
        layer = LAYER
    model, tok, _, d_model, _ = get_model()
    acts = []

    def hook_fn(m, i, o):
        a = o[0] if isinstance(o, tuple) else o
        acts.append(a[:, -1, :].detach().cpu().float().numpy())

    hook = model.transformer[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        for b in range(0, n_win, 64):
            batch = torch.as_tensor(wins[b:b+64].copy(), dtype=torch.float32, device=DEVICE)
            s1, s2 = tok.encode(batch, half=True)
            model(s1, s2)
    hook.remove()
    return np.concatenate(acts)


def extract_all_token_acts(wins, n_win, layer=None):
    """Extract activations at ALL 64 token positions. Returns (n_wins*64, d_model)."""
    if layer is None:
        layer = LAYER
    model, tok, _, d_model, _ = get_model()
    all_acts = []

    def hook_fn(m, i, o):
        a = o[0] if isinstance(o, tuple) else o
        # a shape: (batch, 64, d_model) — keep all positions
        all_acts.append(a.detach().cpu().float().numpy())

    hook = model.transformer[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        for b in range(0, n_win, 64):
            batch = torch.as_tensor(wins[b:b+64].copy(), dtype=torch.float32, device=DEVICE)
            s1, s2 = tok.encode(batch, half=True)
            model(s1, s2)
    hook.remove()
    # Concatenate along batch dim: (total_wins, 64, d_model)
    return np.concatenate(all_acts, axis=0)


def train_sae(train_acts, n_steps=SAE_STEPS):
    """Train a TopK SAE on given activations. Returns SAE."""
    _, _, _, d_model, d_hidden = get_model()
    sae = TopKSAE(d_model, d_hidden, K).to(DEVICE)
    at_tr = torch.as_tensor(np.ascontiguousarray(train_acts), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
    for _ in range(n_steps):
        idx = torch.randint(0, len(at_tr), (SAE_BATCH,))
        xr = sae.encode(at_tr[idx])
        loss = torch.nn.functional.mse_loss(sae.decode(xr), at_tr[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
    return sae


def encode_sae(sae, acts):
    """Encode activations through SAE, return latents."""
    with torch.no_grad():
        at = torch.as_tensor(np.ascontiguousarray(acts), dtype=torch.float32, device=DEVICE)
        lat = sae.encode(at).detach().cpu().numpy()
    return lat


def get_sector(ticker):
    """Heuristic sector classification for Chinese A-share tickers."""
    if ticker.startswith("sh600") or ticker.startswith("sh601"):
        n = int(ticker[4:6]) if len(ticker) >= 6 else 0
        if n >= 0 and n < 16:
            return "bank"
        elif n < 30:
            return "energy"
    if ticker.startswith("sz000") or ticker.startswith("sz002"):
        return "technology"
    return "consumer"


def get_all_csvs():
    return sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])


def create_output_dir(name):
    d = os.path.join(OUTPUT_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d
