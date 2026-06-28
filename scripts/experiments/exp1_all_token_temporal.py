"""Experiment 1: All-Token Temporal SAE.
Extract all 64 token activations per window, train shared SAE on all tokens,
use rolling statistics aligned with token position, test feature-statistic
association on held-out windows with stock-clustered permutation and BH-FDR.
"""
import torch, numpy as np, json, os, time, sys
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_exp_utils import *

OUTPUT = os.path.join(OUTPUT_DIR, "exp1_all_token_temporal.json")
STATE = os.path.join(OUTPUT_DIR, "exp1_all_token_temporal_state.json")

N_WINDOWS_PER_STOCK_TRAIN = 40  # first 40 training windows for SAE
N_WINDOWS_PER_STOCK_TEST = 20   # 20 held-out windows per stock
N_PERMUTATIONS = 100
ALPHA = 0.05


def main():
    model, tok, cfg, d_model, d_hidden = get_model()
    print(f"Model loaded. d_model={d_model}, d_hidden={d_hidden}")

    # Load state
    if os.path.exists(STATE):
        with open(STATE) as f:
            s = json.load(f)
        completed = set(s.get("completed", []))
    else:
        completed = set()

    all_csvs = get_all_csvs()
    t0 = time.time()

    # Collect all-token activations and rolling stats
    all_train_acts = []
    all_train_labels = []
    all_test_acts = []
    all_test_labels = []
    stock_ids_train = []
    stock_ids_test = []

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

        # Take first N_WINDOWS_PER_STOCK_TRAIN windows for training
        train_wins = wins[:min(n_tr, N_WINDOWS_PER_STOCK_TRAIN)]
        test_wins = wins[n_tr+n_val:][:min(n_test, N_WINDOWS_PER_STOCK_TEST)]

        # Extract all-token activations
        train_acts_3d = extract_all_token_acts(train_wins, len(train_wins))
        test_acts_3d = extract_all_token_acts(test_wins, len(test_wins))
        # Flatten: (n_wins * 64, d_model)
        train_acts_flat = train_acts_3d.reshape(-1, d_model)
        test_acts_flat = test_acts_3d.reshape(-1, d_model)

        # Compute rolling statistics aligned with each token position
        train_rolling_labels = []
        for wi in range(len(train_wins)):
            idx_start = (n_tr + n_val - len(train_wins) - len(test_wins)) * STRIDE  # approximate
            # For training: use the actual data up to each token
            base_idx = wi * STRIDE
            for pos in range(WINDOW):
                end_idx = base_idx + pos + 1
                if end_idx <= len(dn):
                    c = dn[base_idx:end_idx, 1]
                    v = dn[base_idx:end_idx, 4] if dn.shape[1] > 4 else None
                    try:
                        stats = compute_statistics_window(c, volume_series=v)
                    except Exception:
                        stats = np.zeros(16, dtype=np.float32)
                    train_rolling_labels.append(stats)
                else:
                    train_rolling_labels.append(np.zeros(16, dtype=np.float32))

        # Test: align with correct data indices
        test_base_offset = n_tr + n_val
        test_rolling_labels = []
        for wi in range(len(test_wins)):
            base_idx = (test_base_offset + wi) * STRIDE
            for pos in range(WINDOW):
                end_idx = base_idx + pos + 1
                if end_idx <= len(dn):
                    c = dn[base_idx:end_idx, 1]
                    v = dn[base_idx:end_idx, 4] if dn.shape[1] > 4 else None
                    try:
                        stats = compute_statistics_window(c, volume_series=v)
                    except Exception:
                        stats = np.zeros(16, dtype=np.float32)
                    test_rolling_labels.append(stats)
                else:
                    test_rolling_labels.append(np.zeros(16, dtype=np.float32))

        train_labels_arr = np.array(train_rolling_labels)
        test_labels_arr = np.array(test_rolling_labels)

        # Trim to match
        m_tr = min(len(train_acts_flat), len(train_labels_arr))
        m_te = min(len(test_acts_flat), len(test_labels_arr))
        all_train_acts.append(train_acts_flat[:m_tr])
        all_train_labels.append(train_labels_arr[:m_tr])
        all_test_acts.append(test_acts_flat[:m_te])
        all_test_labels.append(test_labels_arr[:m_te])
        stock_ids_train.extend([fi] * m_tr)
        stock_ids_test.extend([fi] * m_te)

        completed.add(ticker)
        if (fi + 1) % 20 == 0:
            print(f"[{len(completed)}/{len(all_csvs)}] {ticker}: collected all-token data")
            with open(STATE, "w") as f:
                json.dump({"completed": list(completed)}, f)

    if not all_train_acts:
        print("No data collected!")
        return

    # Concatenate all stocks
    X_train = np.concatenate(all_train_acts).astype(np.float32)
    y_train_all = np.concatenate(all_train_labels).astype(np.float32)
    X_test = np.concatenate(all_test_acts).astype(np.float32)
    y_test_all = np.concatenate(all_test_labels).astype(np.float32)
    stock_train = np.array(stock_ids_train)
    stock_test = np.array(stock_ids_test)

    print(f"\nTraining on {len(X_train)} token-level samples ({len(set(stock_train))} stocks)")
    print(f"Testing on {len(X_test)} token-level samples ({len(set(stock_test))} stocks)")

    # Train shared SAE on all token activations
    print("Training shared SAE on all-token activations...")
    sae = train_sae(X_train, n_steps=SAE_STEPS)
    lat_train = encode_sae(sae, X_train)
    lat_test = encode_sae(sae, X_test)

    # Label features by max |r| with any statistic (training set only)
    feature_labels = {}
    for j in range(lat_train.shape[1]):
        a = lat_train[:, j] != 0
        if a.sum() < 50:
            continue
        best_corr = -1
        best_k = -1
        for k in range(16):
            valid = a & (~np.isnan(y_train_all[:, k]))
            if valid.sum() < 10:
                continue
            corr = abs(np.corrcoef(lat_train[valid, j], y_train_all[valid, k])[0, 1])
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_k = k
        if best_corr > 0.10:
            feature_labels[j] = best_k

    print(f"Labeled {len(feature_labels)} features out of {lat_train.shape[1]}")

    # Group into families
    family_features = defaultdict(list)
    for j, k in feature_labels.items():
        for fname, indices in FAMILIES.items():
            if k in indices:
                family_features[fname].append(j)
                break

    # For each family, test feature-statistic association on held-out
    results = []
    for family, features in family_features.items():
        if len(features) < 3:
            continue

        # Get target statistics for this family
        target_idxs = FAMILIES[family]

        # Mean activation magnitude across family features
        family_act_test = np.abs(lat_test[:, features]).mean(axis=1)
        family_act_train = np.abs(lat_train[:, features]).mean(axis=1)

        for target_idx in target_idxs:
            if target_idx >= y_test_all.shape[1]:
                continue

            y_test = y_test_all[:, target_idx]
            y_train = y_train_all[:, target_idx]

            # Remove NaN
            valid_test = ~np.isnan(y_test) & ~np.isnan(family_act_test)
            if valid_test.sum() < 20:
                continue

            # Pearson correlation on test set
            r_test, _ = pearsonr(family_act_test[valid_test], y_test[valid_test])
            if np.isnan(r_test):
                r_test = 0.0

            # Block permutation: shuffle within stocks in blocks of 10 tokens
            null_rs = []
            rng = np.random.RandomState(42)
            for _ in range(N_PERMUTATIONS):
                perm_act = family_act_test.copy()
                for sid in np.unique(stock_test):
                    smask = stock_test == sid
                    idx = np.where(smask)[0]
                    if len(idx) < 20:
                        continue
                    # Shuffle in blocks of 10
                    n_blocks = len(idx) // 10 + 1
                    blocks = np.array_split(idx, n_blocks)
                    perm_order = rng.permutation(len(blocks))
                    perm_idx = np.concatenate([blocks[i] for i in perm_order])
                    perm_act[idx] = family_act_test[perm_idx]
                vr = perm_act[valid_test]
                vy = y_test[valid_test]
                nr, _ = pearsonr(vr, vy)
                null_rs.append(0.0 if np.isnan(nr) else abs(nr))

            p_val = (np.sum(np.abs(null_rs) >= abs(r_test)) + 1) / (N_PERMUTATIONS + 1)

            results.append({
                "family": family,
                "target_stat": LABEL_NAMES[target_idx],
                "n_features": len(features),
                "r_test": float(r_test),
                "p_permutation": float(p_val),
                "n_test_samples": int(valid_test.sum()),
            })

    # BH-FDR correction across families × statistics
    if results:
        pvals = [r["p_permutation"] for r in results]
        n = len(pvals)
        sorted_idx = np.argsort(pvals)
        bh_rejected = set()
        for rank, idx in enumerate(sorted_idx):
            threshold = ALPHA * (rank + 1) / n
            if pvals[idx] <= threshold:
                bh_rejected.add(idx)
        for i, r in enumerate(results):
            r["bh_significant"] = (i in bh_rejected)

    family_agg = defaultdict(list)
    for r in results:
        family_agg[r["family"]].append(r)

    final = {
        "experiment": "all_token_temporal_sae",
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_labeled_features": len(feature_labels),
        "n_families_with_features": len(family_features),
        "per_family": {f: {
            "n_features": len(feats),
            "results": [r for r in results if r["family"] == f],
            "mean_r_test": float(np.mean([r["r_test"] for r in results if r["family"] == f])),
            "n_bh_significant": sum(1 for r in results if r["family"] == f and r.get("bh_significant")),
        } for f, feats in family_features.items()},
        "bh_significant_count": int(sum(1 for r in results if r.get("bh_significant"))),
        "detail": results,
    }

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"Labeled features: {len(feature_labels)}")
    print(f"BH-FDR significant (q=0.05): {final['bh_significant_count']} / {len(results)}")
    for fam, agg in final["per_family"].items():
        print(f"  {fam}: mean r={agg['mean_r_test']:.4f}, {agg['n_bh_significant']} sig")


if __name__ == "__main__":
    main()
