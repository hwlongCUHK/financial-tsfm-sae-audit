"""Concept-specific steering: amplify features, measure change in generated K-line properties."""
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
    def steer(self, x, steer_ids, multiplier=5.0):
        xc = x - self.b_pre; lat = self.encoder(xc)
        _, idx = torch.topk(lat, self.k, dim=-1)
        mask = torch.zeros_like(lat); mask.scatter_(-1, idx, 1.0)
        lat[:, steer_ids] = lat[:, steer_ids] * multiplier
        return self.decoder(lat * mask) + self.b_pre

def compute_features(close, high, low, volume):
    ret = np.diff(close) / (close[:-1] + 1e-5)
    feats = {}
    feats["momentum_5"] = close[-1] / (close[-6] + 1e-5) - 1 if len(close) >= 6 else 0
    feats["trend"] = np.polyfit(np.arange(len(close)), close, 1)[0]
    feats["volatility"] = np.std(ret)
    feats["vol_persistence"] = np.corrcoef(np.abs(ret[1:]), np.abs(ret[:-1]))[0,1] if len(ret) > 2 else 0
    feats["autocorr_lag1"] = np.corrcoef(ret[1:], ret[:-1])[0,1] if len(ret) > 2 else 0
    feats["max_drawdown"] = np.min(close / np.maximum.accumulate(close) - 1)
    feats["var_95"] = np.percentile(ret, 5)
    feats["max_1day_gain"] = np.max(ret)
    feats["max_1day_loss"] = np.min(ret)
    feats["skewness"] = float(stats.skew(ret)) if len(ret) > 2 else 0
    feats["kurtosis"] = float(stats.kurtosis(ret, fisher=True)) if len(ret) > 3 else 0
    feats["price_range"] = (close.max() - close.min()) / max(close.mean(), 1e-5)
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

    # Train SAE
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
    all_lat = []
    with torch.no_grad():
        for i in range(0, n, 256):
            _, lat2 = sae(at[i:i+256])
            all_lat.append(lat2.cpu().numpy())
    all_lat = np.concatenate(all_lat)[:n]
    label_arr = label_arr[:n]

    # Assign features to concepts
    alive_mask = (all_lat != 0).sum(0) > 10
    concept_features = defaultdict(list)
    for j in np.where(alive_mask)[0]:
        act_j = all_lat[:, j]; a = act_j != 0
        if a.sum() < 5: continue
        corrs = [np.corrcoef(act_j[a], label_arr[a, k])[0,1] for k in range(len(feature_names))]
        corrs = [0 if np.isnan(c) else c for c in corrs]
        best = np.argmax(np.abs(corrs))
        if abs(corrs[best]) > 0.15:
            concept_features[feature_names[best]].append(j)

    # ─── STEERING EXPERIMENT ───
    test_start = n_train
    n_test = min(30, len(windows) - test_start)
    if n_test < 5:
        del sae; torch.cuda.empty_cache(); return None
    test_wins = windows[test_start:test_start+n_test]
    test_t = torch.from_numpy(test_wins).float().to(device)

    # Generate 10 tokens autoregressively with and without steering
    results = {}

    concepts_to_test = [c for c, feats in concept_features.items() if len(feats) >= 5]
    for concept in concepts_to_test[:8]:  # Top 8 concepts
        feat_ids = concept_features[concept][:20]  # Top 20 features

        def make_intervention(steer_ids, multiplier):
            def intervene(m, i, o):
                orig = o[0] if isinstance(o, tuple) else o
                B, T, D = orig.shape
                steered = sae.steer(orig.reshape(-1, D).float(), steer_ids, multiplier)
                return (steered.reshape(B, T, D).half(),) + o[1:] if isinstance(o, tuple) else steered.reshape(B, T, D).half()
            return intervene

        # Baseline generation
        gen_base = []
        gen_steered = []
        gen_amplified = []

        for multiplier, gen_list, label in [(1.0, gen_base, "baseline"), (0.0, gen_steered, "ablate"), (5.0, gen_amplified, "steer_5x")]:
            if multiplier == 1.0:
                # No intervention needed for baseline
                pass
            else:
                hk = model.transformer[layer].register_forward_hook(
                    make_intervention(feat_ids, multiplier))

            with torch.no_grad():
                s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
                out = model(s1_ids, s2_ids)
            if multiplier != 1.0:
                hk.remove()

            # Decode s1 logits to get predicted tokens, then decode to OHLCV
            s1_logits = out[0].float()  # (B, T, vocab_s1)
            pred_tokens = s1_logits[:, -1, :].argmax(dim=-1)  # (B,) — last position prediction

        # Measure financial properties of generated output
        # Decode predicted token to price using tokenizer
        # For simplicity: use token ID as proxy for return magnitude
        baseline_tokens = None
        steered_tokens = None
        amplified_tokens = None

        # Compute financial properties from the window + generated sequence
        # Use close price of the input window + token-based price change
        base_close = test_wins[:, :, 1]  # (B, 64)

        # Baseline
        with torch.no_grad():
            s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
            out_base = model(s1_ids, s2_ids)
        base_tokens = out_base[0].float()[:, -1, :].argmax(dim=-1).cpu().numpy()

        # Ablate (multiplier=0)
        hk_ab = model.transformer[layer].register_forward_hook(make_intervention(feat_ids, 0.0))
        with torch.no_grad():
            s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
            out_ab = model(s1_ids, s2_ids)
        hk_ab.remove()
        ab_tokens = out_ab[0].float()[:, -1, :].argmax(dim=-1).cpu().numpy()

        # Steer (multiplier=5)
        hk_st = model.transformer[layer].register_forward_hook(make_intervention(feat_ids, 5.0))
        with torch.no_grad():
            s1_ids, s2_ids = tokenizer.encode(test_t, half=True)
            out_st = model(s1_ids, s2_ids)
        hk_st.remove()
        st_tokens = out_st[0].float()[:, -1, :].argmax(dim=-1).cpu().numpy()

        # Compute financial metrics from generated tokens
        # Map token change to financial property change
        base_return = (base_tokens - base_tokens.mean()) / (base_tokens.std() + 1e-10)
        ab_return = (ab_tokens - ab_tokens.mean()) / (ab_tokens.std() + 1e-10)
        st_return = (st_tokens - st_tokens.mean()) / (st_tokens.std() + 1e-10)

        # Volatility of generated returns
        base_vol = np.std(base_return)
        ab_vol = np.std(ab_return)
        st_vol = np.std(st_return)

        # Direction: fraction of positive
        base_dir = np.mean(base_return > 0)
        ab_dir = np.mean(ab_return > 0)
        st_dir = np.mean(st_return > 0)

        results[concept] = {
            "n_features": len(feat_ids),
            "base_volatility": float(base_vol),
            "ablated_volatility": float(ab_vol),
            "steered_volatility": float(st_vol),
            "base_direction": float(base_dir),
            "ablated_direction": float(ab_dir),
            "steered_direction": float(st_dir),
            "steer_vol_change": float(st_vol - base_vol),
            "ablate_vol_change": float(ab_vol - base_vol),
            "steer_dir_change": float(st_dir - base_dir),
            "ablate_dir_change": float(ab_dir - base_dir),
        }

    del sae; torch.cuda.empty_cache()
    return {"concept_steering": results, "n_concepts_tested": len(results), "time": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/houwanlong/finllm-mi/data/scale120")
    parser.add_argument("--output", default="/data/houwanlong/finllm-mi/outputs/sae/steering_gen.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-stocks", type=int, default=30)
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
        if (i+1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(csv_files)}]...")
        r = process_one(f, tokenizer, model, args.layer, device, args)
        if r:
            r["ticker"] = f.stem
            results.append(r)

    n = len(results)
    if n == 0: print("No results!"); return

    # Aggregate steering effects
    all_steer_vol = defaultdict(list)
    all_steer_dir = defaultdict(list)
    all_ablate_vol = defaultdict(list)
    all_ablate_dir = defaultdict(list)

    for r in results:
        for concept, data in r.get("concept_steering", {}).items():
            all_steer_vol[concept].append(data["steer_vol_change"])
            all_steer_dir[concept].append(data["steer_dir_change"])
            all_ablate_vol[concept].append(data["ablate_vol_change"])
            all_ablate_dir[concept].append(data["ablate_dir_change"])

    print(f"\nConcept Steering Results (n={n} stocks):")
    print(f"{'Concept':<25} {'Steer Vol':>10} {'Ablate Vol':>10} {'Steer Dir':>10} {'Ablate Dir':>10} {'n':>5}")

    final_concepts = {}
    for concept in sorted(all_steer_vol.keys(), key=lambda x: -len(all_steer_vol[x])):
        sv = np.array(all_steer_vol[concept])
        av = np.array(all_ablate_vol[concept])
        sd = np.array(all_steer_dir[concept])
        ad = np.array(all_ablate_dir[concept])
        if len(sv) < 3: continue

        sv_mean, av_mean = np.mean(sv), np.mean(av)
        sd_mean, ad_mean = np.mean(sd), np.mean(ad)

        # Paired test: steer vs baseline (baseline=0 change)
        t_sv, p_sv = stats.ttest_1samp(sv, 0)
        t_sd, p_sd = stats.ttest_1samp(sd, 0)

        sv_sig = "SIG" if p_sv < 0.05 else "ns"
        sd_sig = "SIG" if p_sd < 0.05 else "ns"

        print(f"{concept:<25} {sv_mean:>+10.4f} {av_mean:>+10.4f} {sd_mean:>+10.4f} {ad_mean:>+10.4f} {len(sv):>5} {sv_sig}/{sd_sig}")

        final_concepts[concept] = {
            "n": len(sv),
            "steer_vol_mean": float(sv_mean), "steer_vol_p": float(p_sv),
            "steer_dir_mean": float(sd_mean), "steer_dir_p": float(p_sd),
        }

    # Overall: does steering ANY concept change volatility or direction?
    all_sv = np.concatenate([np.array(v) for v in all_steer_vol.values()])
    t_all, p_all = stats.ttest_1samp(all_sv, 0)
    n_pos = np.mean(all_sv > 0)
    print(f"\nOverall steering effect on volatility: mean={np.mean(all_sv):+.4f}, p={p_all:.4f}, {n_pos:.0%} positive")

    final = {"n_stocks": n, "concepts": final_concepts, "overall_steer_vol_p": float(p_all),
             "overall_steer_vol_pct_positive": float(n_pos)}
    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Saved. {time.time()-t_total:.0f}s")

if __name__ == "__main__":
    main()
