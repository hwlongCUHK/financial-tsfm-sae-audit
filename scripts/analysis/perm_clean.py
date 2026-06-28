"""Clean permutation test + SAE diagnostics."""
import torch, numpy as np, json, time, argparse, sys
from pathlib import Path; import pandas as pd
from scipy import stats
from collections import defaultdict

sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

class SAE(torch.nn.Module):
    def __init__(self, d, h, k=64):
        super().__init__()
        self.enc = torch.nn.Linear(d, h, bias=True)
        self.dec = torch.nn.Linear(h, d, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(d)); self.k = k
    def encode(self, x):
        xc = x - self.b; lat = self.enc(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        return lat * mask
    def decode(self, lat):
        return self.dec(lat) + self.b
    def permute_encode_decode(self, x, pids):
        lat = self.encode(x)
        if pids:
            perm = torch.randperm(lat.shape[0], device=lat.device)
            lat[:, pids] = lat[perm][:, pids]
        return self.decode(lat)

def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/houwanlong/finllm-mi/data/scale120")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/perm_clean.json")
    parser.add_argument("--n", type=int, default=30)
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
    model.load_state_dict(sd, strict=False)
    model = model.to(device).half().eval()
    layer, d_model, d_hidden = 6, cfg["d_model"], cfg["d_model"]*4

    results = []
    csvs = sorted(Path(args.data_dir).glob("*.csv"))[:args.n]
    print(f"Processing {len(csvs)} stocks...")

    for fi, fp in enumerate(csvs):
        df = pd.read_csv(str(fp))
        for c in ["open","close","high","low","volume","amount"]:
            if c not in df.columns: df[c] = 0.0
        data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
        data = data[~np.isnan(data).any(axis=1)]
        if len(data) < 200: continue
        mn, st = data.mean(0), data.std(0)
        dn = np.clip((data - mn) / (st + 1e-5), -5, 5)
        lb, stride = 64, 32
        nw = min(2000, (len(dn) - lb) // stride)
        if nw < 50: continue
        wins = np.stack([dn[i:i+lb] for i in range(0, nw*stride, stride)])
        n_tr = int(len(wins) * 0.8)

        # Activations
        acts = []
        def hook_fn(m, i, o):
            acts.append(o[0][:, -1, :].detach().cpu().float().numpy() if isinstance(o, tuple) else o[:, -1, :].detach().cpu().float().numpy())
        hook = model.transformer[layer].register_forward_hook(hook_fn)
        bs = 64
        with torch.no_grad():
            for b in range(0, n_tr, bs):
                batch = torch.from_numpy(wins[b:b+bs]).float().to(device)
                s1, s2 = tok.encode(batch, half=True); model(s1, s2)
        hook.remove()
        acts = np.concatenate(acts, axis=0)

        # Train SAE
        sae = SAE(d_model, d_hidden).to(device)
        opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
        at = torch.from_numpy(acts).float().to(device)
        for step in range(3000):
            idx = torch.randint(0, len(at), (512,))
            xr = sae.decode(sae.encode(at[idx]))
            loss = torch.nn.functional.mse_loss(xr, at[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

        # Diagnostics
        with torch.no_grad():
            xd = at[:min(500, len(at))]
            lat_d = sae.encode(xd)
            ve = 1 - torch.nn.functional.mse_loss(sae.decode(lat_d), xd).item() / (xd.var().item() + 1e-10)
            l0 = (lat_d != 0).float().sum(-1).mean().item()
            dead = (lat_d.abs().sum(0) < 1e-6).float().mean().item()
            freq = (lat_d != 0).float().sum(0)

        # Permutation test
        n_te = min(30, len(wins) - n_tr)
        if n_te < 5: del sae; torch.cuda.empty_cache(); continue
        tw = torch.from_numpy(wins[n_tr:n_tr+n_te]).float().to(device)
        with torch.no_grad():
            s1b, s2b = tok.encode(tw, half=True)
            base = model(s1b, s2b)
        base_logits = base[0].float()

        top50 = freq.argsort(descending=True)[:50].tolist()
        rids50 = np.random.choice(d_hidden, 50, replace=False).tolist()

        def intervene(permute_ids):
            def h(m, i, o):
                orig = o[0] if isinstance(o, tuple) else o
                B, T, D = orig.shape
                modified = sae.permute_encode_decode(orig.reshape(-1, D).float(), permute_ids)
                if isinstance(o, tuple):
                    return (modified.reshape(B, T, D).half(),) + o[1:]
                return modified.reshape(B, T, D).half()
            return h

        # Top-50 permutation
        hk = model.transformer[layer].register_forward_hook(intervene(top50))
        with torch.no_grad(): s1t, s2t = tok.encode(tw, half=True); perm_out = model(s1t, s2t)
        hk.remove()
        perm_logits = perm_out[0].float()
        cs_perm = torch.nn.functional.cosine_similarity(
            base_logits.reshape(-1, base_logits.shape[-1]),
            perm_logits.reshape(-1, base_logits.shape[-1]), dim=-1).mean().item()

        # Random 50 permutation (10 trials)
        rand_cs = []
        for _ in range(10):
            hk2 = model.transformer[layer].register_forward_hook(intervene(rids50))
            with torch.no_grad(): s1r, s2r = tok.encode(tw, half=True); ro = model(s1r, s2r)
            hk2.remove()
            rl = ro[0].float()
            rand_cs.append(torch.nn.functional.cosine_similarity(
                base_logits.reshape(-1, base_logits.shape[-1]),
                rl.reshape(-1, base_logits.shape[-1]), dim=-1).mean().item())

        rand_mean = np.mean(rand_cs); rand_std = np.std(rand_cs)
        z = (cs_perm - rand_mean) / (rand_std + 1e-10)
        p = 2 * stats.norm.sf(abs(z))

        results.append({"ticker": fp.stem, "ve": float(ve), "l0": float(l0),
                        "dead": float(dead), "perm_cos": float(cs_perm),
                        "rand_cos": float(rand_mean), "z": float(z), "p": float(p)})

        if (fi+1) % 10 == 0: print(f"[{fi+1}/{len(csvs)}] dead={dead:.1%} perm_p={p:.4f}")
        del sae; torch.cuda.empty_cache()

    n = len(results)
    print(f"\nDone {n} stocks in {time.time()-t0:.0f}s")

    deads = [r["dead"] for r in results]; ves = [r["ve"] for r in results]
    l0s = [r["l0"] for r in results]; ps = [r["p"] for r in results]
    perm_cos = [r["perm_cos"] for r in results]; rand_cos = [r["rand_cos"] for r in results]
    sig = sum(1 for p in ps if p < 0.05)

    print(f"SAE: VE={np.mean(ves):.4f}, L0={np.mean(l0s):.1f}, dead={np.mean(deads):.1%}")
    print(f"Permutation: perm_cos={np.mean(perm_cos):.4f}, rand_cos={np.mean(rand_cos):.4f}")
    print(f"Significant in {sig}/{n} stocks ({sig/n:.0%})")

    zs = [r["z"] for r in results]
    t_all, p_all = stats.ttest_1samp(zs, 0)
    print(f"Mean z={np.mean(zs):.2f}, t={t_all:.2f}, p={p_all:.4f}")

    final = {"n": n, "sae_ve": float(np.mean(ves)), "sae_l0": float(np.mean(l0s)),
             "sae_dead": float(np.mean(deads)), "perm_sig_pct": float(sig/n),
             "perm_cos_mean": float(np.mean(perm_cos)), "rand_cos_mean": float(np.mean(rand_cos)),
             "paired_p": float(p_all), "per_stock": results}
    with open(args.output, "w") as f: json.dump(final, f, indent=2)
    print("Saved")

if __name__ == "__main__": demo()
