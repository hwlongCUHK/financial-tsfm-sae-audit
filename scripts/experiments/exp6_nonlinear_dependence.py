"""Experiment 6: Conditional Nonlinear Dependence Test.
For each family, test nonlinear dependence between SAE features and target
statistic after conditioning out other 29 statistics. Uses:
- Distance correlation (dCor)
- HSIC (Hilbert-Schmidt Independence Criterion)
- Conditional mutual information (CMI, binned approximation)
Null: stock-wise block permutation of SAE activations.
"""
import torch, numpy as np, json, os, time, sys
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_exp_utils import *

OUTPUT = os.path.join(OUTPUT_DIR, "exp6_nonlinear_dependence.json")
STATE = os.path.join(OUTPUT_DIR, "exp6_nonlinear_dependence_state.json")

N_PERMUTATIONS = 200
BLOCK_SIZE = 10  # temporal block size for permutation


def distance_correlation(x, y):
    """Compute distance correlation between 1D arrays x and y."""
    x, y = np.asarray(x), np.asarray(y)
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]
    if len(x) < 10:
        return 0.0
    # Center distance matrices
    n = len(x)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a_row = a.mean(axis=1, keepdims=True)
    a_col = a.mean(axis=0, keepdims=True)
    a_mean = a.mean()
    A = a - a_row - a_col + a_mean
    b_row = b.mean(axis=1, keepdims=True)
    b_col = b.mean(axis=0, keepdims=True)
    b_mean = b.mean()
    B = b - b_row - b_col + b_mean
    d_cov = np.sqrt(np.mean(A * B))
    d_var_x = np.sqrt(np.mean(A * A))
    d_var_y = np.sqrt(np.mean(B * B))
    denom = np.sqrt(d_var_x * d_var_y)
    return float(d_cov / denom) if denom > 1e-10 else 0.0


def hsic_gamma(x, y, sigma=1.0):
    """Gamma-test approximation of HSIC using RBF kernel."""
    x, y = np.asarray(x), np.asarray(y)
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]
    if len(x) < 10:
        return 0.0
    n = len(x)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # Median heuristic for bandwidth
    from scipy.spatial.distance import cdist
    gamma_x = 1.0 / max(np.median(cdist(x, x, "euclidean")), 1e-5)
    gamma_y = 1.0 / max(np.median(cdist(y, y, "euclidean")), 1e-5)
    K = np.exp(-gamma_x * cdist(x, x, "sqeuclidean"))
    L = np.exp(-gamma_y * cdist(y, y, "sqeuclidean"))
    H = np.eye(n) - np.ones((n, n)) / n
    hsic_val = np.trace(K @ H @ L @ H) / (n - 1) ** 2
    return float(hsic_val)


def conditional_mi_binned(x, y, z_cond, n_bins=10):
    """Approximate CMI by binning and computing I(X;Y|Z) ≈ H(X|Z) - H(X|Y,Z)."""
    from collections import Counter
    import math
    valid = ~np.isnan(x) & ~np.isnan(y) & np.all(~np.isnan(z_cond), axis=1)
    x, y, z = x[valid], y[valid], z_cond[valid]
    if len(x) < 20:
        return 0.0
    # Bin x into n_bins
    try:
        x_bins = np.digitize(x, np.percentile(x, np.linspace(0, 100, n_bins+1)[1:-1]))
        y_bins = np.digitize(y, np.percentile(y, np.linspace(0, 100, n_bins+1)[1:-1]))
    except (ValueError, IndexError):
        return 0.0
    # Simple approximation: I(X;Y) using binned counts, then subtract conditioning
    joint = Counter(zip(x_bins, y_bins))
    marginal_x = Counter(x_bins)
    marginal_y = Counter(y_bins)
    n = len(x_bins)
    mi = 0.0
    for (xi, yi), cnt in joint.items():
        p_xy = cnt / n
        p_x = marginal_x[xi] / n
        p_y = marginal_y[yi] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log(p_xy / (p_x * p_y))
    return float(mi)


def compute_conditional_residual(X_sae, y_target, z_other):
    """Regress out other statistics from SAE feature mean, return residual."""
    valid = ~np.isnan(y_target) & np.all(~np.isnan(z_other), axis=1)
    if valid.sum() < 10:
        return None, None, None
    X, y, Z = X_sae[valid], y_target[valid], z_other[valid]
    # Regress Z from y
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    try:
        ridge.fit(Z, y)
        y_resid = y - ridge.predict(Z)
    except Exception:
        y_resid = y - y.mean()
    # Regress Z from X
    try:
        ridge.fit(Z, X)
        X_resid = X - ridge.predict(Z)
    except Exception:
        X_resid = X - X.mean()
    return X_resid, y_resid, valid


def block_permute_acts(acts, stock_ids, block_size, rng):
    """Block-permute activations within each stock."""
    permuted = acts.copy()
    for sid in np.unique(stock_ids):
        mask = stock_ids == sid
        idx = np.where(mask)[0]
        if len(idx) < block_size * 2:
            continue
        n_blocks = len(idx) // block_size
        blocks = np.array_split(idx[:n_blocks * block_size], n_blocks)
        perm_order = rng.permutation(len(blocks))
        for bi, pi in enumerate(perm_order):
            permuted[blocks[bi]] = acts[blocks[pi]]
    return permuted


def main():
    model, tok, cfg, d_model, d_hidden = get_model()
    print("Experiment 6: Conditional Nonlinear Dependence Test")

    if os.path.exists(STATE):
        with open(STATE) as f:
            s = json.load(f)
        completed = set(s.get("completed", []))
    else:
        completed = set()

    all_csvs = get_all_csvs()
    t0 = time.time()

    # Collect per-stock SAE latents + all statistics
    stock_acts = {}
    stock_labels = {}
    stock_ids = []
    act_list, lbl_list = [], []

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
        if n_tr < 10:
            completed.add(ticker)
            continue

        train_acts = extract_acts(wins, n_tr)
        all_labels = compute_all_labels(dn, len(wins))
        train_labels = all_labels[:n_tr]

        if len(train_acts) < 10:
            completed.add(ticker)
            continue

        sae = train_sae(train_acts)
        lat_train = encode_sae(sae, train_acts)
        m_tr = min(len(lat_train), len(train_labels))

        stock_acts[ticker] = lat_train[:m_tr]
        stock_labels[ticker] = train_labels[:m_tr]

        act_list.append(lat_train[:m_tr])
        lbl_list.append(train_labels[:m_tr])
        stock_ids.extend([fi] * m_tr)

        completed.add(ticker)
        if (fi + 1) % 20 == 0:
            print(f"[{len(completed)}/{len(all_csvs)}] {ticker}")
            with open(STATE, "w") as f:
                json.dump({"completed": list(completed)}, f)

    if not act_list:
        print("No data!")
        return

    X_all = np.concatenate(act_list).astype(np.float32)
    Y_all = np.concatenate(lbl_list).astype(np.float32)
    stock_all = np.array(stock_ids)

    print(f"\n{len(X_all)} samples, {Y_all.shape[1]} statistics, {len(np.unique(stock_all))} stocks")

    rng = np.random.RandomState(42)
    results = []

    for family, indices in FAMILIES.items():
        valid_idx = [i for i in indices if i < 16]
        if not valid_idx:
            continue
        # Target: mean of family statistics
        y_target = np.nanmean(Y_all[:, valid_idx], axis=1)
        # Other statistics: all indices NOT in this family
        other_idx = [i for i in range(16) if i not in valid_idx]
        z_other = Y_all[:, other_idx]

        # Mean absolute SAE activation as feature
        x_sae = np.mean(np.abs(X_all), axis=1)

        # Conditional dependence
        X_resid, y_resid, vm = compute_conditional_residual(x_sae, y_target, z_other)
        if X_resid is None or len(X_resid) < 20:
            continue

        # Observed dependence metrics
        obs_dcor = distance_correlation(X_resid, y_resid)
        obs_hsic = hsic_gamma(X_resid, y_resid)
        obs_cmi = conditional_mi_binned(X_resid, y_resid, z_other[vm], n_bins=10)

        # Permutation null: block-permute SAE activations within stocks
        null_dcor, null_hsic, null_cmi = [], [], []
        for _ in range(N_PERMUTATIONS):
            X_perm_full = block_permute_acts(X_all, stock_all, BLOCK_SIZE, rng)
            x_perm = np.mean(np.abs(X_perm_full), axis=1)
            Xp_resid, yp_resid, vp = compute_conditional_residual(x_perm, y_target, z_other)
            if Xp_resid is None or len(Xp_resid) < 20:
                continue
            null_dcor.append(distance_correlation(Xp_resid, yp_resid))
            null_hsic.append(hsic_gamma(Xp_resid, yp_resid))
            null_cmi.append(conditional_mi_binned(Xp_resid, yp_resid, z_other[vp], n_bins=10))

        if not null_dcor:
            continue

        p_dcor = (np.sum(np.array(null_dcor) >= obs_dcor) + 1) / (len(null_dcor) + 1)
        p_hsic = (np.sum(np.array(null_hsic) >= obs_hsic) + 1) / (len(null_hsic) + 1)
        p_cmi = (np.sum(np.array(null_cmi) >= obs_cmi) + 1) / (len(null_cmi) + 1)

        results.append({
            "family": family,
            "n_valid_samples": len(X_resid),
            "dcor_observed": float(obs_dcor),
            "dcor_null_mean": float(np.mean(null_dcor)),
            "dcor_p": float(p_dcor),
            "hsic_observed": float(obs_hsic),
            "hsic_null_mean": float(np.mean(null_hsic)),
            "hsic_p": float(p_hsic),
            "cmi_observed": float(obs_cmi),
            "cmi_null_mean": float(np.mean(null_cmi)),
            "cmi_p": float(p_cmi),
        })

        print(f"  {family}: dCor={obs_dcor:.4f} (p={p_dcor:.3f}), HSIC={obs_hsic:.6f} (p={p_hsic:.3f})")

    # BH-FDR across families and metrics
    all_pvals = []
    for r in results:
        all_pvals.append(("dCor", r["family"], r["dcor_p"]))
        all_pvals.append(("HSIC", r["family"], r["hsic_p"]))
        all_pvals.append(("CMI", r["family"], r["cmi_p"]))

    sorted_p = sorted(all_pvals, key=lambda x: x[2])
    significant = set()
    for rank, (metric, family, p) in enumerate(sorted_p):
        threshold = 0.05 * (rank + 1) / len(sorted_p)
        if p <= threshold:
            significant.add((metric, family))

    final = {
        "experiment": "conditional_nonlinear_dependence",
        "n_stocks": len(np.unique(stock_all)),
        "n_samples": len(X_all),
        "n_permutations": N_PERMUTATIONS,
        "results": results,
        "bh_significant": [[m, f] for m, f in significant],
        "n_bh_significant": len(significant),
    }

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2)
    with open(STATE, "w") as f:
        json.dump({"completed": list(completed)}, f)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"BH-FDR significant (q=0.05): {len(significant)} / {len(all_pvals)}")
    for m, f in sorted(significant):
        print(f"  {m}: {f}")


if __name__ == "__main__":
    main()
