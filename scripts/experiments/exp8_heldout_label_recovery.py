"""Experiment 8: Held-out Concept Recovery Benchmark.
Train SAE on training windows. Assign each feature a concept label on training
data only. On held-out test windows, verify whether the same feature still
correlates most strongly with the same concept. Compare against raw hidden
dims, PCA, and random orthogonal basis.
"""
import torch, numpy as np, json, os, time, sys
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_exp_utils import *

OUTPUT = os.path.join(OUTPUT_DIR, "exp8_heldout_label_recovery.json")
STATE = os.path.join(OUTPUT_DIR, "exp8_heldout_label_recovery_state.json")

N_PERM = 200  # for per-feature null threshold


def safe_corr(x, y):
    try:
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        r, _ = pearsonr(x, y)
        return 0.0 if r is None or np.isnan(r) else r
    except Exception:
        return 0.0


def label_features_on_train(lat_train, train_labels, threshold=0.15):
    """Assign each SAE feature to best statistic using training data only.
    Returns {feature_idx: (stat_idx, corr)} dict."""
    feature_label = {}
    for j in range(lat_train.shape[1]):
        a = lat_train[:, j] != 0
        if a.sum() < 5:
            continue
        best_corr = -1.0
        best_k = -1
        for k in range(min(16, train_labels.shape[1])):
            valid = a & (~np.isnan(train_labels[:, k]))
            if valid.sum() < 5:
                continue
            r = abs(safe_corr(lat_train[valid, j], train_labels[valid, k]))
            if r > best_corr:
                best_corr = r
                best_k = k
        if best_corr > threshold:
            feature_label[j] = (best_k, best_corr)
    return feature_label


def test_label_stability(lat_test, test_labels, feature_label):
    """For each labeled feature, check if the SAME statistic still has
    highest |r| on test data. Returns per-feature stability metrics."""
    stable = 0
    total = 0
    family_stable = 0  # same family, different stat
    per_feature = []

    # Build stat→family map
    stat_to_family = {}
    for fname, indices in FAMILIES.items():
        for idx in indices:
            stat_to_family[idx] = fname

    for j, (train_stat, train_corr) in feature_label.items():
        if j >= lat_test.shape[1]:
            continue
        a = lat_test[:, j] != 0
        if a.sum() < 5:
            continue

        # Recompute |r| with ALL statistics on test data
        test_corrs = {}
        best_test_stat = -1
        best_test_corr = -1.0
        for k in range(min(16, test_labels.shape[1])):
            valid = a & (~np.isnan(test_labels[:, k]))
            if valid.sum() < 5:
                continue
            r = abs(safe_corr(lat_test[valid, j], test_labels[valid, k]))
            test_corrs[k] = r
            if r > best_test_corr:
                best_test_corr = r
                best_test_stat = k

        if best_test_stat < 0:
            continue

        total += 1
        # Stat-level: is train stat still #1?
        if best_test_stat == train_stat:
            stable += 1

        # Family-level: is the best test stat in the same family?
        train_fam = stat_to_family.get(train_stat, "unknown")
        test_fam = stat_to_family.get(best_test_stat, "unknown")
        if train_fam == test_fam:
            family_stable += 1

        # Rank of train stat on test data
        test_corr_sorted = sorted(test_corrs.items(), key=lambda x: x[1], reverse=True)
        train_stat_rank = next((i for i, (k, _) in enumerate(test_corr_sorted) if k == train_stat), -1)

        per_feature.append({
            "feature": j,
            "train_stat": int(train_stat),
            "train_corr": float(train_corr),
            "test_best_stat": int(best_test_stat),
            "test_best_corr": float(best_test_corr),
            "train_stat_test_rank": int(train_stat_rank) if train_stat_rank >= 0 else -1,
            "same_stat": (best_test_stat == train_stat),
            "same_family": (train_fam == test_fam),
        })

    top1_acc = stable / total if total > 0 else 0.0
    family_acc = family_stable / total if total > 0 else 0.0
    return top1_acc, family_acc, per_feature


def test_baseline(X_test, test_labels, baseline_name, feature_label=None, rng=None):
    """Test held-out recovery for a baseline feature set.
    For raw dims / PCA / random: use same number of features as SAE."""
    if rng is None:
        rng = np.random.RandomState(42)
    n_feats = X_test.shape[1]

    # Use all features (no sparsity selection for baselines)
    # For each feature, compute best test stat (no train label to match against)
    # Instead: compute selectivity matrix on test data
    stat_to_family = {}
    for fname, indices in FAMILIES.items():
        for idx in indices:
            stat_to_family[idx] = fname

    # For baselines we compute pairwise |r| between each feature and each stat
    # Then measure: does the top-correlated stat produce meaningful structure?
    n_stats = min(16, test_labels.shape[1])
    feat_stat_corr = np.zeros((n_feats, n_stats))
    for j in range(n_feats):
        a = ~np.isnan(X_test[:, j]) & (np.std(X_test[:, j]) > 1e-10)
        if a.sum() < 5:
            continue
        for k in range(n_stats):
            valid = a & (~np.isnan(test_labels[:, k]))
            if valid.sum() < 5:
                continue
            feat_stat_corr[j, k] = abs(safe_corr(X_test[valid, j], test_labels[valid, k]))

    # Compute max |r| per feature
    max_corr_per_feat = feat_stat_corr.max(axis=1)
    mean_max_corr = float(np.mean(max_corr_per_feat[max_corr_per_feat > 0]))

    # Compute selectivity: is the max stat for each feature unique or is distribution flat?
    # Use normalized entropy of best-stat distribution
    best_per_feat = feat_stat_corr.argmax(axis=1)
    best_valid = best_per_feat[feat_stat_corr.max(axis=1) > 0]
    if len(best_valid) > 0:
        _, counts = np.unique(best_valid, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0
        norm_entropy = float(entropy / max_entropy if max_entropy > 0 else 0)
        largest_family_pct = float(counts.max() / counts.sum())
    else:
        norm_entropy = 0.0
        largest_family_pct = 0.0

    return {
        "baseline": baseline_name,
        "mean_max_corr": mean_max_corr,
        "normalized_entropy": norm_entropy,
        "largest_stat_pct": largest_family_pct,
        "n_valid_features": int(len(best_valid)) if 'best_valid' in dir() else 0,
    }


def main():
    model, tok, cfg, d_model, d_hidden = get_model()
    print("Experiment 8: Held-out Concept Recovery Benchmark")

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

    print("Starting loop over %d stocks" % len(all_csvs), flush=True)
    for fi, fname in enumerate(all_csvs):
        ticker = fname.replace(".csv", "")
        if ticker in completed:
            continue
        print("  [%d] %s..." % (fi, ticker), end="", flush=True)

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

        # 1. SAE
        train_acts = extract_acts(wins, n_tr)
        test_wins = wins[n_tr+n_val:]
        m_test = min(len(test_labels), n_test, 20)
        test_acts = extract_acts(test_wins[:m_test], m_test)

        if len(train_acts) < 10 or len(test_acts) < 5:
            completed.add(ticker)
            continue

        sae = train_sae(train_acts)
        lat_train = encode_sae(sae, train_acts)
        lat_test = encode_sae(sae, test_acts)

        m_tr = min(len(lat_train), len(train_labels))
        m_te = min(len(lat_test), len(test_labels))

        # Label on training
        feature_label = label_features_on_train(lat_train[:m_tr], train_labels[:m_tr])
        if len(feature_label) < 10:
            completed.add(ticker)
            continue

        # Test stability
        top1_acc, family_acc, per_feature = test_label_stability(
            lat_test[:m_te], test_labels[:m_te], feature_label)

        # 2. PCA baseline (same dimensionality as SAE latent)
        n_feats = lat_train.shape[1]
        train_acts_raw = train_acts[:m_tr]
        test_acts_raw = test_acts[:m_te]
        mean = train_acts_raw.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(train_acts_raw - mean, full_matrices=False)
        pca_components = Vt[:n_feats]  # (n_feats, d_model)
        pca_train = (train_acts_raw - mean) @ pca_components.T
        pca_test = (test_acts_raw - mean) @ pca_components.T

        pca_feat_label = label_features_on_train(pca_train, train_labels[:m_tr])
        pca_top1, pca_fam, pca_per = test_label_stability(
            pca_test, test_labels[:m_te], pca_feat_label) if len(pca_feat_label) >= 10 else (0, 0, [])

        # 3. Random orthogonal basis
        rand_basis = rng.randn(n_feats, d_model).astype(np.float32)
        rand_basis /= np.linalg.norm(rand_basis, axis=1, keepdims=True) + 1e-8
        # Orthogonalize
        Q, _ = np.linalg.qr(rand_basis.T)
        rand_basis = Q.T[:n_feats]
        rand_train = (train_acts_raw - mean) @ rand_basis.T
        rand_test = (test_acts_raw - mean) @ rand_basis.T

        rand_feat_label = label_features_on_train(rand_train, train_labels[:m_tr])
        rand_top1, rand_fam, rand_per = test_label_stability(
            rand_test, test_labels[:m_te], rand_feat_label) if len(rand_feat_label) >= 10 else (0, 0, [])

        # 4. Raw hidden dims (best n_feats by variance)
        var = train_acts_raw.var(axis=0)
        top_dims = np.argsort(var)[-n_feats:]
        raw_train = train_acts_raw[:, top_dims]
        raw_test = test_acts_raw[:, top_dims]

        raw_feat_label = label_features_on_train(raw_train, train_labels[:m_tr])
        raw_top1, raw_fam, raw_per = test_label_stability(
            raw_test, test_labels[:m_te], raw_feat_label) if len(raw_feat_label) >= 10 else (0, 0, [])

        all_agg.append({
            "ticker": ticker,
            "n_labeled_features": len(feature_label),
            "sae_top1_stability": float(top1_acc),
            "sae_family_stability": float(family_acc),
            "pca_top1_stability": float(pca_top1),
            "pca_family_stability": float(pca_fam),
            "pca_n_labeled": len(pca_feat_label),
            "random_top1_stability": float(rand_top1),
            "random_family_stability": float(rand_fam),
            "random_n_labeled": len(rand_feat_label),
            "raw_top1_stability": float(raw_top1),
            "raw_family_stability": float(raw_fam),
            "raw_n_labeled": len(raw_feat_label),
        })

        completed.add(ticker)
        del sae
        torch.cuda.empty_cache()
        print(" SAE=%.3f PCA=%.3f Rand=%.3f Raw=%.3f" % (top1_acc, pca_top1, rand_top1, raw_top1), flush=True)

        if (fi + 1) % 10 == 0:
            m_s = np.mean([a["sae_top1_stability"] for a in all_agg])
            m_p = np.mean([a["pca_top1_stability"] for a in all_agg])
            print(f"[{len(completed)}/{len(all_csvs)}] running_means: SAE={m_s:.3f} PCA={m_p:.3f}", flush=True)
            with open(STATE, "w") as f:
                json.dump({"completed": list(completed), "aggregated": all_agg}, f)

    # Final aggregation
    sae_top1 = [a["sae_top1_stability"] for a in all_agg]
    sae_fam = [a["sae_family_stability"] for a in all_agg]
    pca_top1 = [a["pca_top1_stability"] for a in all_agg]
    pca_fam = [a["pca_family_stability"] for a in all_agg]
    rand_top1 = [a["random_top1_stability"] for a in all_agg]
    rand_fam = [a["random_family_stability"] for a in all_agg]
    raw_top1 = [a["raw_top1_stability"] for a in all_agg]
    raw_fam = [a["raw_family_stability"] for a in all_agg]

    final = {
        "experiment": "heldout_label_recovery",
        "n_stocks": len(completed),
        "sae": {
            "mean_top1_stability": float(np.mean(sae_top1)),
            "mean_family_stability": float(np.mean(sae_fam)),
            "top1_ci": [float(np.percentile(sae_top1, 2.5)), float(np.percentile(sae_top1, 97.5))],
            "family_ci": [float(np.percentile(sae_fam, 2.5)), float(np.percentile(sae_fam, 97.5))],
        },
        "pca": {
            "mean_top1_stability": float(np.mean(pca_top1)),
            "mean_family_stability": float(np.mean(pca_fam)),
        },
        "random_orthogonal": {
            "mean_top1_stability": float(np.mean(rand_top1)),
            "mean_family_stability": float(np.mean(rand_fam)),
        },
        "raw_hidden_dims": {
            "mean_top1_stability": float(np.mean(raw_top1)),
            "mean_family_stability": float(np.mean(raw_fam)),
        },
        "sae_vs_pca_top1_delta": float(np.mean(sae_top1) - np.mean(pca_top1)),
        "sae_vs_random_top1_delta": float(np.mean(sae_top1) - np.mean(rand_top1)),
        "sae_vs_raw_top1_delta": float(np.mean(sae_top1) - np.mean(raw_top1)),
        "sae_vs_pca_family_delta": float(np.mean(sae_fam) - np.mean(pca_fam)),
        "sae_vs_random_family_delta": float(np.mean(sae_fam) - np.mean(rand_fam)),
        "sae_vs_raw_family_delta": float(np.mean(sae_fam) - np.mean(raw_fam)),
    }

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2)
    with open(STATE, "w") as f:
        json.dump({"completed": list(completed), "aggregated": all_agg}, f)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"\n=== Held-out Label Recovery ({len(completed)} stocks) ===")
    print(f"{'Method':20s} {'Top-1':>8s} {'Family':>8s}")
    print("-" * 38)
    for name, t1, fam in [("SAE", sae_top1, sae_fam), ("PCA", pca_top1, pca_fam),
                           ("Random", rand_top1, rand_fam), ("Raw dims", raw_top1, raw_fam)]:
        print(f"{name:20s} {np.mean(t1):>8.3f} {np.mean(fam):>8.3f}")
    print(f"\nSAE vs PCA delta:     top1={final['sae_vs_pca_top1_delta']:.3f}  family={final['sae_vs_pca_family_delta']:.3f}")
    print(f"SAE vs Random delta:  top1={final['sae_vs_random_top1_delta']:.3f}  family={final['sae_vs_random_family_delta']:.3f}")
    print(f"SAE vs Raw dims delta: top1={final['sae_vs_raw_top1_delta']:.3f}  family={final['sae_vs_raw_family_delta']:.3f}")


if __name__ == "__main__":
    main()
