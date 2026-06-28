"""Experiment 9: Correlation Profile Stability.
Softer than top-1 label match: for each feature, compute its 16-stat
correlation vector on training data. On test data, recompute the same
vector and measure cosine similarity. Tests whether the *structure*
of correlations generalizes, not whether argmax stays the same.
"""
import torch, numpy as np, json, os, time, sys
from collections import defaultdict
from scipy.stats import pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_exp_utils import *

OUTPUT = os.path.join(OUTPUT_DIR, "exp9_correlation_profile.json")
STATE = os.path.join(OUTPUT_DIR, "exp9_correlation_profile_state.json")
N_STATS = 16


def safe_corr(x, y):
    try:
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        r, _ = pearsonr(x, y)
        return 0.0 if r is None or np.isnan(r) else r
    except Exception:
        return 0.0


def compute_corr_profile(features, labels, n_feats, n_stats):
    """Returns (n_feats, n_stats) matrix of |r| between each feature and
    each statistic."""
    profile = np.zeros((n_feats, n_stats), dtype=np.float32)
    for j in range(n_feats):
        a = features[:, j] != 0
        if a.sum() < 5:
            continue
        for k in range(n_stats):
            valid = a & (~np.isnan(labels[:, k]))
            if valid.sum() < 5:
                continue
            profile[j, k] = abs(safe_corr(features[valid, j], labels[valid, k]))
    return profile


def compute_profile_similarity(profile_train, profile_test, active_mask):
    """Compute cosine similarity between train and test correlation profiles
    for active features. Returns mean cosine, median cosine, fraction
    positive, and all individual cosines."""
    cosines = []
    for j in range(len(active_mask)):
        if not active_mask[j]:
            continue
        pt = profile_train[j]
        pt2 = profile_test[j]
        norm_t = np.linalg.norm(pt) + 1e-10
        norm_te = np.linalg.norm(pt2) + 1e-10
        if norm_t < 1e-8 or norm_te < 1e-8:
            continue
        cos = np.dot(pt, pt2) / (norm_t * norm_te)
        cosines.append(float(np.clip(cos, -1, 1)))
    if not cosines:
        return 0.0, 0.0, 0.0, []
    return float(np.mean(cosines)), float(np.median(cosines)), \
           float(np.mean([1 if c > 0 else 0 for c in cosines])), cosines


def main():
    model, tok, cfg, d_model, d_hidden = get_model()
    print("Experiment 9: Correlation Profile Stability", flush=True)

    if os.path.exists(STATE):
        with open(STATE) as f:
            s = json.load(f)
        completed = set(s.get("completed", []))
        all_agg = s.get("aggregated", [])
    else:
        completed = set()
        all_agg = []

    all_csvs = get_all_csvs()
    t0 = time.time()
    rng = np.random.RandomState(42)

    for fi, fname in enumerate(all_csvs):
        ticker = fname.replace(".csv", "")
        if ticker in completed:
            continue

        loaded = load_stock(fname)
        if loaded is None:
            completed.add(ticker)
            continue
        wins, dn = loaded
        n_tr = int(len(wins) * TRAIN_SPLIT)
        n_val = int(len(wins) * VAL_SPLIT)
        n_test = len(wins) - n_tr - n_val
        if n_tr < 10 or n_test < 5:
            completed.add(ticker)
            continue

        all_labels = compute_all_labels(dn, len(wins))
        train_labels = all_labels[:n_tr]
        test_labels = all_labels[n_tr+n_val:]

        train_acts = extract_acts(wins, n_tr)
        test_wins = wins[n_tr+n_val:]
        m_test = min(len(test_labels), n_test, 20)
        test_acts = extract_acts(test_wins[:m_test], m_test)

        if len(train_acts) < 10 or len(test_acts) < 5:
            completed.add(ticker)
            continue

        # --- SAE ---
        sae = train_sae(train_acts)
        lat_train = encode_sae(sae, train_acts)
        lat_test = encode_sae(sae, test_acts)
        m_tr = min(len(lat_train), len(train_labels))
        m_te = min(len(lat_test), len(test_labels))

        n_feats = lat_train.shape[1]
        train_profile = compute_corr_profile(lat_train[:m_tr], train_labels[:m_tr], n_feats, N_STATS)
        test_profile = compute_corr_profile(lat_test[:m_te], test_labels[:m_te], n_feats, N_STATS)

        # Active features: max |r| > 0.10 on training
        active = train_profile.max(axis=1) > 0.10
        n_active = int(active.sum())

        sae_mean, sae_med, sae_pos, sae_cos = compute_profile_similarity(
            train_profile, test_profile, active)

        # --- PCA ---
        raw_tr = train_acts[:m_tr]
        raw_te = test_acts[:m_te]
        mean = raw_tr.mean(axis=0)
        U, S, Vt = np.linalg.svd(raw_tr - mean, full_matrices=False)
        n_feats = min(n_feats, Vt.shape[0])
        pca_basis = Vt[:n_feats]
        pca_train = (raw_tr - mean) @ pca_basis.T
        pca_test = (raw_te - mean) @ pca_basis.T

        pca_tp = compute_corr_profile(pca_train, train_labels[:m_tr], n_feats, N_STATS)
        pca_ep = compute_corr_profile(pca_test, test_labels[:m_te], n_feats, N_STATS)
        pca_active = pca_tp.max(axis=1) > 0.10
        pca_mean, pca_med, pca_pos, _ = compute_profile_similarity(pca_tp, pca_ep, pca_active)
        n_pca_active = int(pca_active.sum())

        # --- Random orthogonal ---
        rand_basis = rng.randn(n_feats, d_model).astype(np.float32) / np.sqrt(d_model)
        rand_train = (raw_tr - mean) @ rand_basis.T
        rand_test = (raw_te - mean) @ rand_basis.T
        rand_tp = compute_corr_profile(rand_train, train_labels[:m_tr], n_feats, N_STATS)
        rand_ep = compute_corr_profile(rand_test, test_labels[:m_te], n_feats, N_STATS)
        rand_active = rand_tp.max(axis=1) > 0.10
        rand_mean, rand_med, rand_pos, _ = compute_profile_similarity(rand_tp, rand_ep, rand_active)
        n_rand_active = int(rand_active.sum())

        # --- Raw hidden dims (top variance) ---
        var = raw_tr.var(axis=0)
        top_idx = np.argsort(var)[-n_feats:]
        raw_train = raw_tr[:, top_idx]
        raw_test = raw_te[:, top_idx]
        raw_tp = compute_corr_profile(raw_train, train_labels[:m_tr], n_feats, N_STATS)
        raw_ep = compute_corr_profile(raw_test, test_labels[:m_te], n_feats, N_STATS)
        raw_active = raw_tp.max(axis=1) > 0.10
        raw_mean, raw_med, raw_pos, _ = compute_profile_similarity(raw_tp, raw_ep, raw_active)
        n_raw_active = int(raw_active.sum())

        all_agg.append({
            "ticker": ticker,
            "sae_mean_cos": sae_mean, "sae_med_cos": sae_med, "sae_pos_frac": sae_pos,
            "sae_n_active": n_active,
            "pca_mean_cos": pca_mean, "pca_med_cos": pca_med, "pca_pos_frac": pca_pos,
            "pca_n_active": n_pca_active,
            "rand_mean_cos": rand_mean, "rand_med_cos": rand_med, "rand_pos_frac": rand_pos,
            "rand_n_active": n_rand_active,
            "raw_mean_cos": raw_mean, "raw_med_cos": raw_med, "raw_pos_frac": raw_pos,
            "raw_n_active": n_raw_active,
        })

        completed.add(ticker)
        del sae
        torch.cuda.empty_cache()

        if (fi + 1) % 10 == 0:
            m_s = np.mean([a["sae_mean_cos"] for a in all_agg])
            m_p = np.mean([a["pca_mean_cos"] for a in all_agg])
            m_r = np.mean([a["rand_mean_cos"] for a in all_agg])
            m_w = np.mean([a["raw_mean_cos"] for a in all_agg])
            print("[%d/%d] SAE=%.3f PCA=%.3f Rand=%.3f Raw=%.3f" %
                  (len(completed), len(all_csvs), m_s, m_p, m_r, m_w), flush=True)
            with open(STATE, "w") as f:
                json.dump({"completed": list(completed), "aggregated": all_agg}, f)

    # Final
    sae_cos = [a["sae_mean_cos"] for a in all_agg]
    pca_cos = [a["pca_mean_cos"] for a in all_agg]
    rand_cos = [a["rand_mean_cos"] for a in all_agg]
    raw_cos = [a["raw_mean_cos"] for a in all_agg]

    final = {
        "experiment": "correlation_profile_stability",
        "n_stocks": len(completed),
        "sae": {"mean_cosine": float(np.mean(sae_cos)),
                "median_cosine": float(np.median(sae_cos)),
                "mean_pos_frac": float(np.mean([a["sae_pos_frac"] for a in all_agg])),
                "mean_n_active": float(np.mean([a["sae_n_active"] for a in all_agg]))},
        "pca": {"mean_cosine": float(np.mean(pca_cos)),
                "median_cosine": float(np.median(pca_cos)),
                "mean_pos_frac": float(np.mean([a["pca_pos_frac"] for a in all_agg])),
                "mean_n_active": float(np.mean([a["pca_n_active"] for a in all_agg]))},
        "random": {"mean_cosine": float(np.mean(rand_cos)),
                   "median_cosine": float(np.median(rand_cos)),
                   "mean_pos_frac": float(np.mean([a["rand_pos_frac"] for a in all_agg])),
                   "mean_n_active": float(np.mean([a["rand_n_active"] for a in all_agg]))},
        "raw": {"mean_cosine": float(np.mean(raw_cos)),
                "median_cosine": float(np.median(raw_cos)),
                "mean_pos_frac": float(np.mean([a["raw_pos_frac"] for a in all_agg])),
                "mean_n_active": float(np.mean([a["raw_n_active"] for a in all_agg]))},
        "sae_vs_pca_delta": float(np.mean(sae_cos) - np.mean(pca_cos)),
        "sae_vs_rand_delta": float(np.mean(sae_cos) - np.mean(rand_cos)),
        "sae_vs_raw_delta": float(np.mean(sae_cos) - np.mean(raw_cos)),
    }

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2)
    with open(STATE, "w") as f:
        json.dump({"completed": list(completed), "aggregated": all_agg}, f)

    print("\nDone in %.0fs" % (time.time()-t0), flush=True)
    print("\n=== Correlation Profile Stability ===")
    for name, cos_list in [("SAE", sae_cos), ("PCA", pca_cos),
                            ("Random", rand_cos), ("Raw", raw_cos)]:
        print("  %s: mean_cos=%.4f  median=%.4f" % (name, np.mean(cos_list), np.median(cos_list)))
    print("\nSAE-PCA delta: %.4f" % final["sae_vs_pca_delta"])
    print("SAE-Rand delta: %.4f" % final["sae_vs_rand_delta"])
    print("SAE-Raw delta: %.4f" % final["sae_vs_raw_delta"])


if __name__ == "__main__":
    main()
