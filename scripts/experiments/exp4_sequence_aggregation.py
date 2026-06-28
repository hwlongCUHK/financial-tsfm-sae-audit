"""Experiment 4: Sequence-Level Aggregation Probe.
For each SAE feature, compute temporal aggregates across all 64 token positions
(mean, max, std, slope, last-first, firing proportion), then predict
16 statistics. Pooling selection done on training set only.
"""
import torch, numpy as np, json, os, time, sys
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_exp_utils import *

OUTPUT = os.path.join(OUTPUT_DIR, "exp4_sequence_aggregation.json")
STATE = os.path.join(OUTPUT_DIR, "exp4_sequence_aggregation_state.json")


def compute_aggregates(acts_3d):
    """acts_3d: (n_wins, 64, n_features or d_model)
    Returns (n_wins, n_features * 6) — 6 aggregate stats per feature.
    """
    n_wins, seq_len, n_feats = acts_3d.shape
    agg_list = []
    for j in range(n_feats):
        feat_seq = acts_3d[:, :, j]  # (n_wins, 64)
        # Mean
        agg_list.append(feat_seq.mean(axis=1, keepdims=True))
        # Max
        agg_list.append(feat_seq.max(axis=1, keepdims=True))
        # Std
        agg_list.append(feat_seq.std(axis=1, keepdims=True))
        # Activation slope (linear fit over positions)
        x = np.arange(seq_len, dtype=np.float32)
        x = x - x.mean()
        x_norm = (x * x).sum()
        if x_norm > 0:
            slope = ((feat_seq - feat_seq.mean(axis=1, keepdims=True)) * x).sum(axis=1) / x_norm
        else:
            slope = np.zeros(n_wins)
        agg_list.append(slope[:, np.newaxis])
        # Last minus first
        agg_list.append((feat_seq[:, -1] - feat_seq[:, 0])[:, np.newaxis])
        # Firing proportion (> 0)
        agg_list.append((feat_seq > 0).mean(axis=1)[:, np.newaxis])
    return np.concatenate(agg_list, axis=1)


def compute_best_aggregate(train_features_3d, train_labels, val_features_3d, val_labels):
    """For each feature, select the single best aggregate statistic using validation.
    Returns indices of selected aggregate for each original feature.
    """
    agg_names = ["mean", "max", "std", "slope", "last-first", "firing_rate"]
    n_feats = train_features_3d.shape[2]
    best_aggs = []
    for j in range(n_feats):
        best_r = -1
        best_agg = 0
        for ai in range(6):
            feat_seq_tr = train_features_3d[:, :, j]
            feat_seq_val = val_features_3d[:, :, j]
            if ai == 0:
                agg_tr = feat_seq_tr.mean(axis=1)
                agg_val = feat_seq_val.mean(axis=1)
            elif ai == 1:
                agg_tr = feat_seq_tr.max(axis=1)
                agg_val = feat_seq_val.max(axis=1)
            elif ai == 2:
                agg_tr = feat_seq_tr.std(axis=1)
                agg_val = feat_seq_val.std(axis=1)
            elif ai == 3:
                x = np.arange(feat_seq_tr.shape[1], dtype=np.float32)
                x = x - x.mean()
                x_norm = (x * x).sum()
                if x_norm > 0:
                    agg_tr = ((feat_seq_tr - feat_seq_tr.mean(axis=1, keepdims=True)) * x).sum(axis=1) / x_norm
                    agg_val = ((feat_seq_val - feat_seq_val.mean(axis=1, keepdims=True)) * x).sum(axis=1) / x_norm
                else:
                    agg_tr = np.zeros(feat_seq_tr.shape[0])
                    agg_val = np.zeros(feat_seq_val.shape[0])
            elif ai == 4:
                agg_tr = feat_seq_tr[:, -1] - feat_seq_tr[:, 0]
                agg_val = feat_seq_val[:, -1] - feat_seq_val[:, 0]
            else:
                agg_tr = (feat_seq_tr > 0).mean(axis=1)
                agg_val = (feat_seq_val > 0).mean(axis=1)

            valid = ~np.isnan(agg_val) & ~np.isnan(val_labels)
            if valid.sum() < 5:
                continue
            r = abs(pearsonr(agg_val[valid], val_labels[valid])[0])
            if not np.isnan(r) and r > best_r:
                best_r = r
                best_agg = ai
        best_aggs.append(best_agg)
    return best_aggs


def main():
    model, tok, cfg, d_model, d_hidden = get_model()
    print("Experiment 4: Sequence-Level Aggregation Probe")

    if os.path.exists(STATE):
        with open(STATE) as f:
            s = json.load(f)
        completed = set(s.get("completed", []))
        results = s.get("results", [])
    else:
        completed = set()
        results = []

    all_csvs = get_all_csvs()
    t0 = time.time()

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
        if n_tr < 10 or n_val < 3 or n_test < 5:
            completed.add(ticker)
            continue

        all_labels = compute_all_labels(dn, len(wins))

        # Extract all-token activations
        train_acts_3d = extract_all_token_acts(wins, n_tr)
        val_acts_3d = extract_all_token_acts(wins[n_tr:][:n_val], n_val)
        test_acts_3d = extract_all_token_acts(wins[n_tr+n_val:], min(n_test, 20))

        # Train SAE on final-token activations
        train_final = extract_acts(wins, n_tr)
        sae = train_sae(train_final)

        # Encode all-token activations through SAE
        train_lat_3d_list = []
        for b in range(0, train_acts_3d.shape[0], 64):
            batch = train_acts_3d[b:b+64].reshape(-1, d_model)  # flatten token dim
            lat = encode_sae(sae, batch)
            train_lat_3d_list.append(lat.reshape(-1, WINDOW, d_hidden))
        train_lat_3d = np.concatenate(train_lat_3d_list, axis=0) if train_lat_3d_list else train_acts_3d[:, :, :d_hidden]

        val_lat_3d_list = []
        for b in range(0, val_acts_3d.shape[0], 64):
            batch = val_acts_3d[b:b+64].reshape(-1, d_model)
            lat = encode_sae(sae, batch)
            val_lat_3d_list.append(lat.reshape(-1, WINDOW, d_hidden))
        val_lat_3d = np.concatenate(val_lat_3d_list, axis=0) if val_lat_3d_list else val_acts_3d[:, :, :d_hidden]

        test_lat_3d_list = []
        for b in range(0, test_acts_3d.shape[0], 64):
            batch = test_acts_3d[b:b+64].reshape(-1, d_model)
            lat = encode_sae(sae, batch)
            test_lat_3d_list.append(lat.reshape(-1, WINDOW, d_hidden))
        test_lat_3d = np.concatenate(test_lat_3d_list, axis=0) if test_lat_3d_list else test_acts_3d[:, :, :d_hidden]

        # For each family, compute aggregates and predict
        for family, indices in FAMILIES.items():
            valid_idx = [i for i in indices if i < 16]
            if not valid_idx:
                continue

            train_labels = all_labels[:n_tr][:, valid_idx].mean(axis=1)
            val_labels = all_labels[n_tr:n_tr+n_val][:, valid_idx].mean(axis=1)
            test_labels = all_labels[n_tr+n_val:][:, valid_idx].mean(axis=1)

            m_tr = min(train_lat_3d.shape[0], len(train_labels))
            m_val = min(val_lat_3d.shape[0], len(val_labels))
            m_te = min(test_lat_3d.shape[0], len(test_labels))

            # Select best aggregate per feature using validation
            best_aggs = compute_best_aggregate(
                train_lat_3d[:m_tr], train_labels[:m_tr],
                val_lat_3d[:m_val], val_labels[:m_val])

            # Build aggregated features for train and test
            X_tr_agg = []
            X_te_agg = []
            for j, ai in enumerate(best_aggs):
                feat_tr = train_lat_3d[:m_tr, :, j]
                feat_te = test_lat_3d[:m_te, :, j]
                if ai == 0:
                    X_tr_agg.append(feat_tr.mean(axis=1))
                    X_te_agg.append(feat_te.mean(axis=1))
                elif ai == 1:
                    X_tr_agg.append(feat_tr.max(axis=1))
                    X_te_agg.append(feat_te.max(axis=1))
                elif ai == 2:
                    X_tr_agg.append(feat_tr.std(axis=1))
                    X_te_agg.append(feat_te.std(axis=1))
                elif ai == 3:
                    x = np.arange(feat_tr.shape[1], dtype=np.float32)
                    x = x - x.mean()
                    xn = (x*x).sum()
                    X_tr_agg.append(((feat_tr-feat_tr.mean(axis=1,keepdims=True))*x).sum(axis=1)/xn if xn>0 else np.zeros(m_tr))
                    X_te_agg.append(((feat_te-feat_te.mean(axis=1,keepdims=True))*x).sum(axis=1)/xn if xn>0 else np.zeros(m_te))
                elif ai == 4:
                    X_tr_agg.append(feat_tr[:, -1] - feat_tr[:, 0])
                    X_te_agg.append(feat_te[:, -1] - feat_te[:, 0])
                else:
                    X_tr_agg.append((feat_tr > 0).mean(axis=1))
                    X_te_agg.append((feat_te > 0).mean(axis=1))

            X_tr = np.column_stack(X_tr_agg)
            X_te = np.column_stack(X_te_agg)
            y_tr = train_labels[:m_tr]
            y_te = test_labels[:m_te]

            valid_tr = ~np.isnan(y_tr)
            valid_te = ~np.isnan(y_te)
            if valid_tr.sum() < 10 or valid_te.sum() < 5:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr[valid_tr])
            X_te_s = scaler.transform(X_te[valid_te])

            probe = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            probe.fit(X_tr_s, y_tr[valid_tr])
            y_pred = probe.predict(X_te_s)

            ss_res = np.sum((y_te[valid_te] - y_pred)**2)
            ss_tot = np.sum((y_te[valid_te] - np.mean(y_te[valid_te]))**2)
            r2 = float(1 - ss_res / max(ss_tot, 1e-8))
            r_pearson = float(pearsonr(y_te[valid_te], y_pred)[0]) if len(y_te[valid_te]) > 1 else 0

            # Baseline: final-token only
            lat_final_tr = encode_sae(sae, extract_acts(wins, n_tr))
            lat_final_te = encode_sae(sae, extract_acts(wins[n_tr+n_val:], m_te))

            m_fr = min(len(lat_final_tr), m_tr)
            m_fe = min(len(lat_final_te), m_te)
            scaler_ft = StandardScaler()
            X_ft_s = scaler_ft.fit_transform(lat_final_tr[:m_fr])
            X_fe_s = scaler_ft.transform(lat_final_te[:m_fe])
            probe_ft = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            probe_ft.fit(X_ft_s, y_tr[:m_fr])
            yp_ft = probe_ft.predict(X_fe_s)
            ss_ft = np.sum((y_te[:m_fe] - yp_ft)**2)
            r2_final = float(1 - ss_ft / max(np.sum((y_te[:m_fe] - np.mean(y_te[:m_fe]))**2), 1e-8))

            results.append({
                "ticker": ticker,
                "family": family,
                "r2_aggregated": r2,
                "r2_final_only": r2_final,
                "pearson_r": r_pearson,
                "delta_r2": r2 - r2_final,
            })

        completed.add(ticker)
        if (fi + 1) % 20 == 0:
            print(f"[{len(completed)}/{len(all_csvs)}] {ticker}")
            with open(STATE, "w") as f:
                json.dump({"completed": list(completed), "results": results}, f)

    # Aggregate
    agg = defaultdict(list)
    for r in results:
        agg[r["family"]].append(r)

    family_summary = {}
    for family, fam_results in agg.items():
        deltas = [r["delta_r2"] for r in fam_results]
        family_summary[family] = {
            "n_stocks": len(fam_results),
            "mean_r2_aggregated": float(np.mean([r["r2_aggregated"] for r in fam_results])),
            "mean_r2_final_only": float(np.mean([r["r2_final_only"] for r in fam_results])),
            "mean_delta_r2": float(np.mean(deltas)),
            "pct_positive_delta": float(np.mean([1 if d > 0 else 0 for d in deltas])),
        }

    final = {
        "experiment": "sequence_aggregation_probe",
        "n_stocks": len(completed),
        "per_family": family_summary,
    }

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2)
    with open(STATE, "w") as f:
        json.dump({"completed": list(completed), "results": results}, f)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print("\nSequence Aggregation Results:")
    for family, s in sorted(family_summary.items()):
        direction = "BETTER" if s["mean_delta_r2"] > 0 else "WORSE"
        print(f"  {family:20s}: delta R²={s['mean_delta_r2']:.4f} ({direction}), {s['pct_positive_delta']:.0%} positive")


if __name__ == "__main__":
    main()
