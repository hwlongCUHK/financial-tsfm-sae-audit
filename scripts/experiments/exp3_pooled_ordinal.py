"""Experiment 3: Pooled Ordinal Concept Probing.
Cross-stock pooled probe with stock fixed effects, within-stock quantile
targets (low/middle/high), leave-sector-out + future-period evaluation.
Predicts ordinal class, evaluates macro-AUC.
"""
import torch, numpy as np, json, os, time, sys
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.sparse import hstack
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_exp_utils import *

OUTPUT = os.path.join(OUTPUT_DIR, "exp3_pooled_ordinal.json")
STATE = os.path.join(OUTPUT_DIR, "exp3_pooled_ordinal_state.json")

N_QUANTILES = 3  # low / middle / high
SECTORS = ["bank", "energy", "technology", "consumer"]


def main():
    model, tok, cfg, d_model, d_hidden = get_model()
    print("Experiment 3: Pooled Ordinal Concept Probing")

    if os.path.exists(STATE):
        with open(STATE) as f:
            s = json.load(f)
        completed = set(s.get("completed", []))
    else:
        completed = set()

    all_csvs = get_all_csvs()
    t0 = time.time()

    # Collect per-stock data
    stock_data = {}
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

        train_acts = extract_acts(wins, n_tr)
        test_wins = wins[n_tr+n_val:]
        all_labels = compute_all_labels(dn, len(wins))
        test_labels = all_labels[n_tr+n_val:]

        m_test = min(len(test_labels), n_test)
        test_acts = extract_acts(test_wins[:m_test], m_test)

        if len(train_acts) < 10 or len(test_acts) < 5:
            completed.add(ticker)
            continue

        # Train per-stock SAE
        sae = train_sae(train_acts)
        lat_train = encode_sae(sae, train_acts)
        lat_test = encode_sae(sae, test_acts)

        train_lbl = all_labels[:n_tr]
        m_tr = min(len(lat_train), len(train_lbl))
        m_te = min(len(lat_test), len(test_labels))

        sector = get_sector(ticker)
        stock_data[ticker] = {
            "lat_train": lat_train[:m_tr],
            "lat_test": lat_test[:m_te],
            "labels_train": train_lbl[:m_tr],
            "labels_test": test_labels[:m_te],
            "sector": sector,
        }
        completed.add(ticker)

        if (fi + 1) % 20 == 0:
            print(f"[{len(completed)}/{len(all_csvs)}] {ticker}")
            with open(STATE, "w") as f:
                json.dump({"completed": list(completed)}, f)

    if len(stock_data) < 10:
        print("Not enough stocks!")
        return

    print(f"Collected data for {len(stock_data)} stocks")

    # For each family, build pooled ordinal probe
    results = []
    for family, indices in FAMILIES.items():
        valid_indices = [i for i in indices if i < 16]
        if not valid_indices:
            continue

        # Leave-sector-out evaluation
        for test_sector in SECTORS:
            # Build train/test split
            X_train_list, y_train_list, stock_train_ids = [], [], []
            X_test_list, y_test_list, stock_test_ids = [], [], []

            for sid, (ticker, data) in enumerate(stock_data.items()):
                sector = data["sector"]
                lat = np.abs(data["lat_train"])  # Use absolute activation magnitudes
                lbl = data["labels_train"][:, valid_indices].mean(axis=1)

                valid = ~np.isnan(lbl)
                if valid.sum() < 5:
                    continue

                if sector == test_sector:
                    # Test on this sector's test windows
                    lat_te = np.abs(data["lat_test"])
                    lbl_te = data["labels_test"][:, valid_indices].mean(axis=1)
                    valid_te = ~np.isnan(lbl_te)
                    if valid_te.sum() < 3:
                        continue
                    X_test_list.append(lat_te[valid_te])
                    y_test_list.append(lbl_te[valid_te])
                    stock_test_ids.extend([sid] * valid_te.sum())
                else:
                    X_train_list.append(lat[valid])
                    y_train_list.append(lbl[valid])
                    stock_train_ids.extend([sid] * valid.sum())

            if not X_train_list or not X_test_list:
                continue

            X_train_raw = np.concatenate(X_train_list)
            y_train_raw = np.concatenate(y_train_list)
            X_test_raw = np.concatenate(X_test_list)
            y_test_raw = np.concatenate(y_test_list)
            stock_train = np.array(stock_train_ids)
            stock_test = np.array(stock_test_ids)

            # Convert to within-stock quantiles for training targets
            # Use stock-specific quantile bins
            y_train_ordinal = np.zeros(len(y_train_raw), dtype=int)
            for sid in np.unique(stock_train):
                smask = stock_train == sid
                if smask.sum() < 6:
                    y_train_ordinal[smask] = 1  # default to middle
                    continue
                vals = y_train_raw[smask]
                q33, q67 = np.percentile(vals, [33.33, 66.67])
                y_train_ordinal[smask] = np.where(vals <= q33, 0,
                                          np.where(vals <= q67, 1, 2))

            # For test, binarize high vs low (drop middle for AUC)
            y_test_binary = np.zeros(len(y_test_raw), dtype=int)
            for sid in np.unique(stock_test):
                smask = stock_test == sid
                if smask.sum() < 6:
                    continue
                vals = y_test_raw[smask]
                q33, q67 = np.percentile(vals, [33.33, 66.67])
                high_mask = vals >= q67
                low_mask = vals <= q33
                # Only keep high and low for binary eval
                y_test_binary[smask] = np.where(high_mask, 1, np.where(low_mask, 0, -1))

            binary_mask = y_test_binary >= 0
            if binary_mask.sum() < 10 or len(np.unique(y_test_binary[binary_mask])) < 2:
                continue

            X_test_bin = X_test_raw[binary_mask]
            y_test_bin = y_test_binary[binary_mask]

            # Add stock fixed effects
            stock_encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
            stock_train_enc = stock_encoder.fit_transform(stock_train.reshape(-1, 1))
            stock_test_enc = stock_encoder.transform(stock_test.reshape(-1, 1))
            stock_test_bin_enc = stock_encoder.transform(
                stock_test[binary_mask].reshape(-1, 1))

            # Scale latents
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_raw)
            X_test_s = scaler.transform(X_test_raw)
            X_test_bin_s = scaler.transform(X_test_bin)

            # Combine: SAE latents + stock fixed effects
            X_train_full = hstack([X_train_s, stock_train_enc])
            X_test_full = hstack([X_test_s, stock_test_enc])
            X_test_bin_full = hstack([X_test_bin_s, stock_test_bin_enc])

            # Train ordinal logistic regression
            try:
                clf = LogisticRegression(multi_class="multinomial", max_iter=1000,
                                         C=1.0, random_state=42)
                clf.fit(X_train_full, y_train_ordinal)

                # Predict on test (binary high vs low)
                y_prob = clf.predict_proba(X_test_bin_full)
                # AUC for high class vs low class using class 2 vs class 0 probabilities
                if y_prob.shape[1] >= 2:
                    # Use P(high) vs P(low) for binary AUC
                    prob_high = y_prob[:, 2] if y_prob.shape[1] > 2 else y_prob[:, 1]
                    prob_low = y_prob[:, 0]
                    # Probability ratio as score
                    score = prob_high / (prob_low + prob_high + 1e-8)
                    try:
                        auc = roc_auc_score(y_test_bin, score)
                    except ValueError:
                        auc = 0.5

                    y_pred = clf.predict(X_test_bin_full)
                    bal_acc = balanced_accuracy_score(y_test_bin, np.where(
                        y_pred == 1, 1, np.where(y_pred == 2, 1, 0)))

                    results.append({
                        "family": family,
                        "test_sector": test_sector,
                        "n_train": len(y_train_ordinal),
                        "n_test": len(y_test_bin),
                        "auc": float(auc),
                        "balanced_accuracy": float(bal_acc),
                    })
            except Exception as e:
                print(f"  Failed {family}/{test_sector}: {e}")
                continue

        print(f"  {family}: done")

    # Aggregate
    agg = defaultdict(list)
    for r in results:
        agg[r["family"]].append(r)

    family_summary = {}
    for family, fam_results in agg.items():
        aucs = [r["auc"] for r in fam_results]
        bals = [r["balanced_accuracy"] for r in fam_results]
        family_summary[family] = {
            "n_sector_folds": len(fam_results),
            "mean_auc": float(np.mean(aucs)),
            "mean_balanced_accuracy": float(np.mean(bals)),
            "auc_above_chance": int(sum(1 for a in aucs if a > 0.5)),
        }

    final = {
        "experiment": "pooled_ordinal_probing",
        "n_stocks": len(stock_data),
        "per_family": family_summary,
        "detail": results,
    }

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2)
    with open(STATE, "w") as f:
        json.dump({"completed": list(completed)}, f)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print("\nPooled Ordinal Results:")
    for family, s in sorted(family_summary.items()):
        print(f"  {family:20s}: AUC={s['mean_auc']:.4f} ({s['auc_above_chance']}/{s['n_sector_folds']} folds >0.5)")


if __name__ == "__main__":
    main()
