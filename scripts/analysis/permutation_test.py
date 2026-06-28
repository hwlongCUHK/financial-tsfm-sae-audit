"""Permutation test + SAE config robustness + comprehensive diagnostics table."""
import torch, numpy as np, json, sys, time, argparse
from pathlib import Path
import pandas as pd
from scipy import stats
from collections import defaultdict

sys.path.insert(0, "/data/houwanlong/finllm-mi/code")
from model.kronos import Kronos, KronosTokenizer
from safetensors.torch import load_file

class TopKSAE(torch.nn.Module):
    def __init__(self, d_model, d_hidden, k=64):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = torch.nn.Linear(d_hidden, d_model, bias=False)
        self.b_pre = torch.nn.Parameter(torch.zeros(d_model))
        self.k = k
    def forward(self, x):
        xc = x - self.b_pre; lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        return self.decoder(lat * mask) + self.b_pre, lat * mask
    def encode_decode(self, x):
        xc = x - self.b_pre; lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        return self.decoder(lat * mask) + self.b_pre
    def permute_and_decode(self, x, permute_ids):
        """Permute activation values of specified features across batch."""
        xc = x - self.b_pre; lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        lat = lat * mask
        # Permute specified features across batch dimension
        if len(permute_ids) > 0:
            perm = torch.randperm(lat.shape[0], device=lat.device)
            lat[:, permute_ids] = lat[perm][:, permute_ids]
        return self.decoder(lat) + self.b_pre

def compute_features(close, high, low, volume):
    ret = np.diff(close) / (close[:-1] + 1e-5)
    feats = {}
    feats["momentum_5"] = close[-1] / (close[-6] + 1e-5) - 1 if len(close) >= 6 else 0
    feats["trend"] = np.polyfit(np.arange(len(close)), close, 1)[0]
    feats["volatility"] = np.std(ret)
    feats["vol_persistence"] = np.corrcoef(np.abs(ret[1:]), np.abs(ret[:-1]))[0,1] if len(ret) > 2 else 0
    feats["autocorr_lag1"] = np.corrcoef(ret[1:], ret[:-1])[0,1] if len(ret) > 2 else 0
    feats["autocorr_lag5"] = np.corrcoef(ret[5:], ret[:-5])[0,1] if len(ret) > 6 else 0
    feats["max_drawdown"] = np.min(close / np.maximum.accumulate(close) - 1)
    feats["var_95"] = np.percentile(ret, 5)
    feats["max_1day_gain"] = np.max(ret)
    feats["skewness"] = float(stats.skew(ret)) if len(ret) > 2 else 0
    for k in list(feats.keys()):
        if not np.isfinite(feats[k]): feats[k] = 0.0
    return feats

def process_one(csv_path, tokenizer, model, layer, device, args):
    t0 = time.time()
    df = pd.read_csv(str(csv_path))
    for col in ["open","close","high","low","volume","amount"]:
        if col not in df.columns: df[col] = 0.0
    data = df[["open","close","high","low","volume","amount"]].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) < 200: return None
    mn, st_d = data.mean(0), data.std(0)
    data_norm = np.clip((data - mn) / (st_d + 1e-5), -5, 5)

    lb, stride = 64, 32
    nw = min(2000, (len(data_norm) - lb) // stride)
    windows = np.stack([data_norm[i:i+lb] for i in range(0, nw*stride, stride)])
    n_train = int(len(windows) * 0.8)

    acts_list = []
    def make_hook(storage):
        def h(m, i, o):
            a = o[0] if isinstance(o, tuple) else o
            storage.append(a[:, -1, :].detach().cpu().float().numpy())
        return h
    acts_train = []
    hook = model.transformer[layer].register_forward_hook(make_hook(acts_train))
    bs = 64
    with torch.no_grad():
        for b in range(0, n_train, bs):
            batch = torch.from_numpy(windows[b:b+bs]).float().to(device)
            s1, s2 = tokenizer.encode(batch, half=True)
            model(s1, s2)
    hook.remove()
    acts = np.concatenate(acts_train, axis=0)

    d_model, d_hidden = acts.shape[1], acts.shape[1] * 4
    sae = TopKSAE(d_model, d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
    at = torch.from_numpy(acts).float().to(device)
    for step in range(args.steps):
        idx = torch.randint(0, len(at), (args.batch,))
        xr, _ = sae(at[idx])
        loss = torch.nn.functional.mse_loss(xr, at[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0); opt.step()

    # SAE diagnostics
    with torch.no_grad():
        xt = at[:min(500, len(at))]
        recon, lat = sae(xt)
        ve_train = 1 - torch.nn.functional.mse_loss(recon, xt).item() / (xt.var().item() + 1e-10)
        l0 = (lat != 0).float().sum(-1).mean().item()
        dead = (lat.abs().sum(0) < 1e-6).float().mean().item()
        alive_rate = (lat.abs().sum(0) > 1e-6).float().mean().item()
    freq = (lat != 0).float().sum(0).cpu().numpy()

    # Label features
    feature_names = None
    all_labels = []
    for i in range(0, len(data_norm) - lb * 2, stride):
        c = data_norm[i:i+lb, 1]; h = data_norm[i:i+lb, 2]; l = data_norm[i:i+lb, 3]; v = data_norm[i:i+lb, 4]
        feats = compute_features(c, h, l, v)
        if feature_names is None: feature_names = sorted(feats.keys())
        all_labels.append([feats[k] for k in feature_names])
    label_arr = np.array(all_labels)
    n = min(len(acts), len(label_arr))
    label_arr = label_arr[:n]
    all_lat_np = []
    with torch.no_grad():
        for i in range(0, n, 256):
            _, lat2 = sae(at[i:i+256])
            all_lat_np.append(lat2.cpu().numpy())
    all_lat_np = np.concatenate(all_lat_np)[:n]

    alive_mask = (all_lat_np != 0).sum(0) > 10
    concept_features = defaultdict(list)
    for j in np.where(alive_mask)[0]:
        act_j = all_lat_np[:, j]; a = act_j != 0
        if a.sum() < 5: continue
        corrs = [np.corrcoef(act_j[a], label_arr[a, k])[0,1] for k in range(len(feature_names))]
        corrs = [0 if np.isnan(c) else abs(c) for c in corrs]
        best = np.argmax(corrs)
        if corrs[best] > 0.15:
            concept_features[feature_names[best]].append(j)

    # ─── PERMUTATION TEST ───
    test_start = n_train
    n_test = min(30, len(windows) - test_start)
    if n_test < 5: del sae; torch.cuda.empty_cache(); return None
    test_wins_t = torch.from_numpy(windows[test_start:test_start+n_test]).float().to(device)

    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(test_wins_t, half=True)
        base_out = model(s1_ids, s2_ids)
    base_logits = base_out[0].float()  # (B, T, vocab) — model final output

    # Top-50 features by frequency
    top50 = np.argsort(freq)[-50:].tolist()

    # Test: permute concept features vs permute random features
    permute_results = {}

    def make_permute_hook(permute_ids):
        def intervene(m, i, o):
            orig = o[0] if isinstance(o, tuple) else o
            B, T, D = orig.shape
            modified = sae.permute_and_decode(orig.reshape(-1, orig.shape[-1]).float(), permute_ids)
            return (modified.reshape(B, T, D).half(),) + o[1:] if isinstance(o, tuple) else modified.reshape(B, T, D).half()
        return intervene

    # Permute top-50
    hk = model.transformer[layer].register_forward_hook(make_permute_hook(top50))
    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(test_wins_t, half=True)
        perm50_out = model(s1_ids, s2_ids)
    hk.remove()
    cs_perm50 = torch.nn.functional.cosine_similarity(
        base_s1.reshape(-1, base_s1.shape[-1]), perm50_out[0].float().reshape(-1, d_model), dim=-1).mean()

    # Permute random 50 (10 trials)
    rand_cos = []
    for _ in range(10):
        rids = np.random.choice(d_hidden, 50, replace=False).tolist()
        hk2 = model.transformer[layer].register_forward_hook(make_permute_hook(rids))
        with torch.no_grad():
            s1_ids, s2_ids = tokenizer.encode(test_wins_t, half=True)
            ro = model(s1_ids, s2_ids)
        hk2.remove()
        rand_cos.append(torch.nn.functional.cosine_similarity(
            base_s1.reshape(-1, base_s1.shape[-1]), ro[0].float().reshape(-1, d_model), dim=-1).mean().item())

    rand_mean = np.mean(rand_cos); rand_std = np.std(rand_cos)
    z_perm = (cs_perm50.item() - rand_mean) / (rand_std + 1e-10)
    p_perm = 2 * stats.norm.sf(abs(z_perm))

    # Also test per-concept permutation
    concept_perm_results = {}
    for concept, feat_ids in concept_features.items():
        if len(feat_ids) < 10: continue
        ids = feat_ids[:20]
        hk = model.transformer[layer].register_forward_hook(make_permute_hook(ids))
        with torch.no_grad():
            s1_ids, s2_ids = tokenizer.encode(test_wins_t, half=True)
            cp_out = model(s1_ids, s2_ids)
        hk.remove()
        cs_cp = torch.nn.functional.cosine_similarity(
            base_s1.reshape(-1, base_s1.shape[-1]), cp_out[0].float().reshape(-1, d_model), dim=-1).mean()

        # Random permutation baseline
        rand_cp = []
        for _ in range(10):
            rids = np.random.choice(d_hidden, len(ids), replace=False).tolist()
            hk3 = model.transformer[layer].register_forward_hook(make_permute_hook(rids))
            with torch.no_grad():
                s1_ids, s2_ids = tokenizer.encode(test_wins_t, half=True)
                ro3 = model(s1_ids, s2_ids)
            hk3.remove()
            rand_cp.append(torch.nn.functional.cosine_similarity(
                base_s1.reshape(-1, base_s1.shape[-1]), ro3[0].float().reshape(-1, d_model), dim=-1).mean().item())

        rcp_mean = np.mean(rand_cp)
        concept_perm_results[concept] = {
            "cos": float(cs_cp.item()),
            "rand_cos": float(rcp_mean),
            "delta": float(cs_cp.item() - rcp_mean),
            "n_features": len(ids),
        }

    del sae; torch.cuda.empty_cache()
    return {
        "sae_diag": {"ve_train": float(ve_train), "l0": float(l0), "dead": float(dead), "alive": float(alive_rate)},
        "permute_top50": {"cos": float(cs_perm50.item()), "rand_cos_mean": float(rand_mean), "z": float(z_perm), "p": float(p_perm)},
        "permute_concept": concept_perm_results,
        "time": time.time() - t0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/houwanlong/finllm-mi/data/scale120")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/permutation_results.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-stocks", type=int, default=40)
    args = parser.parse_args()

    device = args.device; t_total = time.time()
    print("Loading Kronos...")
    tokenizer = KronosTokenizer.from_pretrained("/data/houwanlong/models/Kronos-Tokenizer-base").to(device).eval()
    with open("/data/houwanlong/models/Kronos-base/config.json") as f: cfg = json.load(f)
    model = Kronos(s1_bits=cfg["s1_bits"], s2_bits=cfg["s2_bits"], n_layers=cfg["n_layers"],
                   d_model=cfg["d_model"], n_heads=cfg["n_heads"], ff_dim=cfg["ff_dim"],
                   ffn_dropout_p=cfg["ffn_dropout_p"], attn_dropout_p=cfg["attn_dropout_p"],
                   resid_dropout_p=cfg["resid_dropout_p"], token_dropout_p=cfg["token_dropout_p"],
                   learn_te=cfg["learn_te"])
    sd = load_file("/data/houwanlong/models/Kronos-base/model.safetensors")
    model.load_state_dict(sd, strict=False); model = model.to(device).half().eval()

    csv_files = sorted(Path(args.data_dir).glob("*.csv"))[:args.max_stocks]
    print(f"Processing {len(csv_files)} stocks...")

    results = []
    for i, f in enumerate(csv_files):
        if (i+1) % 10 == 0: print(f"[{i+1}/{len(csv_files)}]...")
        r = process_one(f, tokenizer, model, args.layer, device, args)
        if r: r["ticker"] = f.stem; results.append(r)

    n = len(results)
    if n == 0: print("No results!"); return

    # Aggregate
    dead_rates = [r["sae_diag"]["dead"] for r in results]
    alive_rates = [r["sae_diag"]["alive"] for r in results]
    ve_trains = [r["sae_diag"]["ve_train"] for r in results]
    l0s = [r["sae_diag"]["l0"] for r in results]
    perm_ps = [r["permute_top50"]["p"] for r in results]
    perm_sig = sum(1 for p in perm_ps if p < 0.05)

    print(f"\n=== SAE Diagnostics (n={n}) ===")
    print(f"VE train: {np.mean(ve_trains):.4f} +- {np.std(ve_trains):.4f}")
    print(f"L0: {np.mean(l0s):.1f} +- {np.std(l0s):.1f}")
    print(f"Dead features: {np.mean(dead_rates):.1%} +- {np.std(dead_rates):.1%}")
    print(f"Alive features: {np.mean(alive_rates):.1%} +- {np.std(alive_rates):.1%}")

    print(f"\n=== Permutation Test ===")
    print(f"Permuting top-50 features: significant in {perm_sig}/{n} stocks")
    print(f"Mean p-value: {np.mean(perm_ps):.4f}")

    # Concept permutation aggregate
    all_cp = defaultdict(list)
    for r in results:
        for concept, data in r.get("permute_concept", {}).items():
            all_cp[concept].append(data["delta"])
    print(f"\nConcept permutation deltas (neg = more impact than random):")
    for concept in sorted(all_cp.keys(), key=lambda x: -len(all_cp[x]))[:10]:
        deltas = all_cp[concept]
        if len(deltas) < 3: continue
        md = np.mean(deltas); t, p = stats.ttest_1samp(deltas, 0)
        print(f"  {concept:<25}: n={len(deltas)}, delta={md:+.4f}, p={p:.4f}")

    final = {"n_stocks": n, "sae_diag_aggregate": {
        "ve_mean": float(np.mean(ve_trains)), "l0_mean": float(np.mean(l0s)),
        "dead_mean": float(np.mean(dead_rates)), "alive_mean": float(np.mean(alive_rates)),
    }, "permutation_sig_pct": float(perm_sig/n)}
    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved. {time.time()-t_total:.0f}s")

if __name__ == "__main__":
    main()
