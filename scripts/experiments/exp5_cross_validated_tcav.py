"""Experiment 5: Cross-Validated TCAV.
For each concept, create high/low sets from real windows, train CAV on
training stocks, test on held-out stocks. Evaluate:
- high/low classification accuracy
- cross-fold direction stability (cosine similarity)
- Kronos output directional derivative
- Compare against random concept vectors
"""
import torch, numpy as np, json, os, time, sys
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_exp_utils import *

OUTPUT = os.path.join(OUTPUT_DIR, "exp5_cross_validated_tcav.json")
STATE = os.path.join(OUTPUT_DIR, "exp5_cross_validated_tcav_state.json")

N_SPLITS = 5  # 5-fold cross-stock validation
N_RANDOM_CAV = 50  # random CAV comparisons
DT = 0.1  # step size for directional derivative


def compute_tcav_score(acts, cav):
    """Compute TCAV score: fraction of samples with positive projection onto CAV."""
    proj = acts @ cav
    return float((proj > 0).mean())


def directional_derivative(latents, cav, sae):
    """Perturb SAE latents along CAV direction, decode, measure change.
    latents: already-encoded SAE latents (n, d_hidden), not raw model acts.
    cav: concept activation vector in latent space.
    """
    with torch.no_grad():
        lat = torch.as_tensor(np.ascontiguousarray(latents), dtype=torch.float32, device=DEVICE)
        cav_t = torch.as_tensor(cav, dtype=torch.float32, device=DEVICE)
        cav_norm = cav_t / (torch.norm(cav_t) + 1e-8)
        lat_pert = lat + DT * cav_norm[None, :]
        orig_decoded = sae.decode(lat)
        pert_decoded = sae.decode(lat_pert)
        cos_sim = torch.nn.functional.cosine_similarity(orig_decoded, pert_decoded, dim=-1).mean()
    return float(1.0 - cos_sim)


def main():
    model, tok, cfg, d_model, d_hidden = get_model()
    print("Experiment 5: Cross-Validated TCAV")

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

    # Phase 1: Collect per-stock data (activations + labels)
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
        if n_tr < 10:
            completed.add(ticker)
            continue

        all_labels = compute_all_labels(dn, len(wins))
        train_labels = all_labels[:n_tr]
        train_acts = extract_acts(wins, n_tr)

        if len(train_acts) < 10:
            completed.add(ticker)
            continue

        sae = train_sae(train_acts)
        lat_train = encode_sae(sae, train_acts)
        m_tr = min(len(lat_train), len(train_labels))

        stock_data[ticker] = {
            "lat_train": lat_train[:m_tr],
            "labels_train": train_labels[:m_tr],
            "sae": sae,
            "sector": get_sector(ticker),
        }
        completed.add(ticker)

        if (fi + 1) % 20 == 0:
            print(f"[{len(completed)}/{len(all_csvs)}] {ticker}")
            with open(STATE, "w") as f:
                json.dump({"completed": list(completed), "results": results}, f)

    tickers_all = list(stock_data.keys())
    if len(tickers_all) < 20:
        print("Not enough stocks!")
        return

    rng = np.random.RandomState(42)

    # Cross-stock CV: split stocks into folds
    all_csvs_full = get_all_csvs()
    stocks_with_sectors = [(t, stock_data[t]["sector"]) for t in tickers_all]

    # Group by sector for stratified splitting
    sector_stocks = defaultdict(list)
    for t, s in stocks_with_sectors:
        sector_stocks[s].append(t)

    # Create folds balanced across sectors
    folds = [[] for _ in range(N_SPLITS)]
    for sector, stocks in sector_stocks.items():
        rng.shuffle(stocks)
        for i, t in enumerate(stocks):
            folds[i % N_SPLITS].append(t)

    print(f"\n{len(tickers_all)} stocks in {N_SPLITS} folds")
    for i, fold in enumerate(folds):
        print(f"  Fold {i}: {len(fold)} stocks")

    # For each concept family, run cross-validated TCAV
    for family, indices in FAMILIES.items():
        valid_idx = [i for i in indices if i < 16]
        if not valid_idx:
            continue

        for fold_i in range(N_SPLITS):
            test_stocks = set(folds[fold_i])
            train_stocks = [t for t in tickers_all if t not in test_stocks]

            # Build high/low sets from TRAIN stocks
            X_high, X_low = [], []
            for t in train_stocks:
                data = stock_data[t]
                lat = np.abs(data["lat_train"])
                lbl = data["labels_train"][:, valid_idx].mean(axis=1)
                valid = ~np.isnan(lbl)
                if valid.sum() < 10:
                    continue
                lv, la = lbl[valid], lat[valid]
                q75, q25 = np.percentile(lv, [75, 25])
                high_mask = lv >= q75
                low_mask = lv <= q25
                if high_mask.sum() >= 5:
                    X_high.append(la[high_mask])
                if low_mask.sum() >= 5:
                    X_low.append(la[low_mask])

            if not X_high or not X_low:
                continue
            X_high_all = np.concatenate(X_high)
            X_low_all = np.concatenate(X_low)

            # Train CAV: logistic regression weight vector separating high vs low
            X = np.concatenate([X_high_all, X_low_all])
            y = np.concatenate([np.ones(len(X_high_all)), np.zeros(len(X_low_all))])
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_s, y)
            cav = clf.coef_[0].copy()  # direction in latent space

            # Train random CAVs for baseline
            rand_cavs = [rng.randn(*cav.shape) for _ in range(N_RANDOM_CAV)]

            # Test on held-out stock SAEs
            test_class_acc = []
            test_cav_scores = []
            test_dir_derivs = []
            test_rand_scores = {i: [] for i in range(N_RANDOM_CAV)}

            for t in test_stocks:
                data = stock_data[t]
                lat = np.abs(data["lat_train"])
                lbl = data["labels_train"][:, valid_idx].mean(axis=1)
                valid = ~np.isnan(lbl)
                if valid.sum() < 10:
                    continue
                lv, la = lbl[valid], lat[valid]
                q75, q25 = np.percentile(lv, [75, 25])
                high_mask = lv >= q75
                low_mask = lv <= q25
                if high_mask.sum() < 3 or low_mask.sum() < 3:
                    continue

                # Scale test latents using training scaler
                la_s = scaler.transform(la)
                Xh_s = la_s[high_mask]
                Xl_s = la_s[low_mask]

                # Classification accuracy of CAV projection
                proj_h = Xh_s @ cav
                proj_l = Xl_s @ cav
                acc = float((np.mean(proj_h > np.median(np.concatenate([proj_h, proj_l]))) +
                             np.mean(proj_l <= np.median(np.concatenate([proj_h, proj_l])))) / 2)
                test_class_acc.append(acc)

                # TCAV score on all test activations
                tcav_score = compute_tcav_score(la_s, cav)
                test_cav_scores.append(tcav_score)

                # Directional derivative
                dd = directional_derivative(la, cav, data["sae"])
                test_dir_derivs.append(dd)

                # Random CAV scores
                for ri, rcav in enumerate(rand_cavs):
                    test_rand_scores[ri].append(compute_tcav_score(la_s, rcav))

            if not test_class_acc:
                continue

            # Random baseline: mean across random CAVs
            rand_cav_means = [np.mean(vals) for vals in test_rand_scores.values()]
            rand_mean = np.mean(rand_cav_means)
            rand_std = np.std(rand_cav_means)

            results.append({
                "family": family,
                "fold": fold_i,
                "n_train_stocks": len(train_stocks),
                "n_test_stocks": len(test_class_acc),
                "mean_classification_acc": float(np.mean(test_class_acc)),
                "mean_tcav_score": float(np.mean(test_cav_scores)),
                "mean_directional_derivative": float(np.mean(test_dir_derivs)),
                "random_cav_tcav_mean": float(rand_mean),
                "random_cav_tcav_std": float(rand_std),
                "tcav_vs_random_z": float((np.mean(test_cav_scores) - rand_mean) / max(rand_std, 1e-8)),
            })

        print(f"  {family}: {N_SPLITS} folds done")

    # Aggregate
    agg = defaultdict(list)
    for r in results:
        agg[r["family"]].append(r)

    family_summary = {}
    for family, fam_results in agg.items():
        accs = [r["mean_classification_acc"] for r in fam_results]
        tcav = [r["mean_tcav_score"] for r in fam_results]
        zs = [r["tcav_vs_random_z"] for r in fam_results]
        family_summary[family] = {
            "n_folds": len(fam_results),
            "mean_classification_acc": float(np.mean(accs)),
            "mean_tcav_score": float(np.mean(tcav)),
            "mean_tcav_random": float(np.mean([r["random_cav_tcav_mean"] for r in fam_results])),
            "mean_z_vs_random": float(np.mean(zs)),
            "n_folds_positive_z": int(sum(1 for z in zs if z > 0)),
        }

    final = {
        "experiment": "cross_validated_tcav",
        "n_stocks_total": len(tickers_all),
        "n_folds": N_SPLITS,
        "per_family": family_summary,
        "detail": results,
    }

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2)
    with open(STATE, "w") as f:
        json.dump({"completed": list(completed), "results": results}, f)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print("\nCross-Validated TCAV Results:")
    for family, s in sorted(family_summary.items()):
        print(f"  {family:20s}: Acc={s['mean_classification_acc']:.3f}, "
              f"TCAV={s['mean_tcav_score']:.3f} (random={s['mean_tcav_random']:.3f}), "
              f"Z={s['mean_z_vs_random']:.2f}, {s['n_folds_positive_z']}/{s['n_folds']} folds +")


if __name__ == "__main__":
    main()
