"""L1 SAE (fewer dead features) + LOOKBACK=20 (more windows) + permutation test."""
import torch, numpy as np, json, time, argparse
from pathlib import Path; import pandas as pd
from scipy import stats
from collections import defaultdict
import sys
sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

class L1SAE(torch.nn.Module):
    """L1-penalized SAE — all features get gradients, fewer dead features than TopK."""
    def __init__(self, d, h, l1_alpha=0.001):
        super().__init__()
        self.enc = torch.nn.Linear(d, h, bias=True)
        self.dec = torch.nn.Linear(h, d, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(d))
        self.l1_alpha = l1_alpha
    def forward(self, x):
        xc = x - self.b
        lat = torch.nn.functional.relu(self.enc(xc))  # ReLU for non-neg + sparsity
        return self.dec(lat) + self.b, lat
    def permute_encode_decode(self, x, pids):
        xc = x - self.b
        lat = torch.nn.functional.relu(self.enc(xc))
        if pids:
            perm = torch.randperm(lat.shape[0], device=lat.device)
            lat[:, pids] = lat[perm][:, pids]
        return self.dec(lat) + self.b

def process_one(fp, tok, model, layer, device):
    df = pd.read_csv(str(fp))
    for c in ["open","close","high","low","volume","amount"]:
        if c not in df.columns: df[c] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 200: return None
    mn, st = data.mean(0), data.std(0)
    dn = np.clip((data - mn) / (st + 1e-5), -5, 5)

    LOOKBACK, STRIDE = 20, 10  # More windows!
    nw = min(2000, (len(dn) - LOOKBACK) // STRIDE)
    if nw < 50: return None
    wins = np.stack([dn[i:i+LOOKBACK] for i in range(0, nw*STRIDE, STRIDE)])
    n_tr = int(len(wins) * 0.8)

    # Activations
    acts = []
    def hook_fn(m, i, o):
        a = o[0] if isinstance(o, tuple) else o
        acts.append(a[:, -1, :].detach().cpu().float().numpy())
    hook = model.transformer[layer].register_forward_hook(hook_fn)
    bs = 64
    with torch.no_grad():
        for b in range(0, n_tr, bs):
            batch = torch.from_numpy(wins[b:b+bs]).float().to(device)
            s1, s2 = tok.encode(batch, half=True)
            model(s1, s2)
    hook.remove()
    acts = np.concatenate(acts, axis=0)

    d_model, d_hidden = acts.shape[1], acts.shape[1] * 2
    sae = L1SAE(d_model, d_hidden, l1_alpha=0.001).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
    at = torch.from_numpy(acts).float().to(device)

    for step in range(5000):
        idx = torch.randint(0, len(at), (512,))
        recon, lat = sae(at[idx])
        loss = torch.nn.functional.mse_loss(recon, at[idx]) + sae.l1_alpha * lat.abs().mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

    # Diagnostics
    with torch.no_grad():
        xd = at[:min(500, len(at))]
        recon_d, lat_d = sae(xd)
        ve = 1 - torch.nn.functional.mse_loss(recon_d, xd).item() / (xd.var().item() + 1e-10)
        l0 = (lat_d > 1e-4).float().sum(-1).mean().item()
        dead = (lat_d.abs().sum(0) < 1e-4).float().mean().item()
        freq = (lat_d > 1e-4).float().sum(0)

    # Permutation test
    n_te = min(30, len(wins) - n_tr)
    if n_te < 5: del sae; torch.cuda.empty_cache(); return None
    tw = torch.from_numpy(wins[n_tr:n_tr+n_te]).float().to(device)
    with torch.no_grad():
        s1b, s2b = tok.encode(tw, half=True)
        base = model(s1b, s2b)
    base_logits = base[0].float()

    top50 = freq.argsort(descending=True)[:50].tolist()

    def intervene(perm_ids):
        def h(m, i, o):
            orig = o[0] if isinstance(o, tuple) else o
            B, T, D = orig.shape
            modified = sae.permute_encode_decode(orig.reshape(-1, D).float(), perm_ids)
            if isinstance(o, tuple):
                return (modified.reshape(B, T, D).half(),) + o[1:]
            return modified.reshape(B, T, D).half()
        return h

    hk = model.transformer[layer].register_forward_hook(intervene(top50))
    with torch.no_grad(): s1t, s2t = tok.encode(tw, half=True); po = model(s1t, s2t)
    hk.remove()
    cs_perm = torch.nn.functional.cosine_similarity(
        base_logits.reshape(-1, base_logits.shape[-1]),
        po[0].float().reshape(-1, base_logits.shape[-1]), dim=-1).mean().item()

    rids = np.random.choice(d_hidden, 50, replace=False).tolist()
    rand_cs = []
    for _ in range(10):
        hk2 = model.transformer[layer].register_forward_hook(intervene(rids))
        with torch.no_grad(): s1r, s2r = tok.encode(tw, half=True); ro = model(s1r, s2r)
        hk2.remove()
        rand_cs.append(torch.nn.functional.cosine_similarity(
            base_logits.reshape(-1, base_logits.shape[-1]),
            ro[0].float().reshape(-1, base_logits.shape[-1]), dim=-1).mean().item())

    rand_mean = np.mean(rand_cs); rand_std = np.std(rand_cs)
    z = (cs_perm - rand_mean) / (rand_std + 1e-10)
    p = 2 * stats.norm.sf(abs(z))

    del sae; torch.cuda.empty_cache()
    return {"ticker": fp.stem, "ve": float(ve), "l0": float(l0), "dead": float(dead),
            "perm_cos": float(cs_perm), "rand_cos": float(rand_mean), "z": float(z), "p": float(p)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/houwanlong/finllm-mi/data/scale120")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/l1_sae.json")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device; t0 = time.time()
    tok = KronosTokenizer.from_pretrained("/data/houwanlong/models/Kronos-Tokenizer-base").to(device).eval()
    with open("/data/houwanlong/models/Kronos-base/config.json") as f: cfg = json.load(f)
    model = Kronos(s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"], n_layers=cfg["n_layers"],
                   d_model=cfg["d_model"], n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
                   ffn_dropout_p=cfg["ffn_dropout_p"], attn_dropout_p=cfg["attn_dropout_p"],
                   resid_dropout_p=cfg["resid_dropout_p"], token_dropout_p=cfg["token_dropout_p"],
                   learn_te=cfg["learn_te"])
    sd = load_file("/data/houwanlong/models/Kronos-base/model.safetensors")
    model.load_state_dict(sd, strict=False); model = model.to(device).half().eval()

    csvs = sorted(Path(args.data_dir).glob("*.csv"))[:args.n]
    print(f"L1 SAE + LOOKBACK=20 + permutation on {len(csvs)} stocks")

    results = []
    for i, fp in enumerate(csvs):
        r = process_one(fp, tok, model, 6, device)
        if r: results.append(r)
        if (i+1) % 15 == 0: print(f"[{i+1}/{len(csvs)}] dead={np.mean([x['dead'] for x in results[-15:]]):.1%}")

    n = len(results)
    deads = [r["dead"] for r in results]
    l0s = [r["l0"] for r in results]
    ves = [r["ve"] for r in results]
    ps = [r["p"] for r in results]
    sig = sum(1 for p in ps if p < 0.05)
    perm_cos = [r["perm_cos"] for r in results]
    rand_cos = [r["rand_cos"] for r in results]

    print(f"\nL1 SAE Results (n={n}):")
    print(f"  VE: {np.mean(ves):.4f}, L0: {np.mean(l0s):.1f}, DEAD: {np.mean(deads):.1%}")
    print(f"  Perm cos: {np.mean(perm_cos):.4f} vs rand: {np.mean(rand_cos):.4f}")
    print(f"  Sig: {sig}/{n} ({sig/n:.0%})")

    zs = [r["z"] for r in results]
    t_all, p_all = stats.ttest_1samp(zs, 0)
    print(f"  Paired z-test: t={t_all:.2f}, p={p_all:.6f}")

    # Compare with TopK results
    print(f"\n  vs TopK SAE (dead=82%, sig=98%)")
    print(f"  L1 SAE dead={np.mean(deads):.1%} — {82-np.mean(deads)*100:.0f}pp improvement")

    final = {"n": n, "sae_type": "L1", "lookback": 20,
             "dead": float(np.mean(deads)), "l0": float(np.mean(l0s)), "ve": float(np.mean(ves)),
             "perm_cos": float(np.mean(perm_cos)), "rand_cos": float(np.mean(rand_cos)),
             "sig_pct": float(sig/n), "paired_p": float(p_all)}
    with open(args.output, "w") as f: json.dump(final, f, indent=2)
    print(f"Saved. {time.time()-t0:.0f}s")

if __name__ == "__main__": main()
