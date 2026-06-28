"""Experiment 2: Multi-Layer Concept Localization.
Extract activations at layers L in {0,3,6,9,11}, for each concept family
select the best layer on training set, then confirm on held-out test set.
"""
import torch, numpy as np, json, os, time, sys
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_exp_utils import *

OUTPUT = os.path.join(OUTPUT_DIR, "exp2_multilayer_localization.json")
STATE = os.path.join(OUTPUT_DIR, "exp2_multilayer_state.json")

LAYERS = [0, 3, 6, 9, 11]


def extract_acts_at_layer(wins_subset, n_win, layer):
    """Extract final-token activations at a specific layer from exactly n_win windows."""
    model, tok, _, _, _ = get_model()
    acts = []
    def hook_fn(m, i, o):
        a = o[0] if isinstance(o, tuple) else o
        acts.append(a[:, -1, :].detach().cpu().float().numpy())
    hook = model.transformer[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        for b in range(0, n_win, 64):
            end = min(b + 64, n_win)
            batch = torch.as_tensor(wins_subset[b:end].copy(), dtype=torch.float32, device=DEVICE)
            s1, s2 = tok.encode(batch, half=True)
            model(s1, s2)
    hook.remove()
    return np.concatenate(acts)


def safe_corr(x, y):
    """Compute |r| safely, return None on failure."""
    try:
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return None
        r, _ = pearsonr(x, y)
        if r is None or np.isnan(r):
            return None
        return abs(float(r))
    except Exception:
        return None


def label_features(lat_train, train_labels):
    """Assign each SAE feature to best-matching statistic. Returns {feature_idx: stat_idx}."""
    feature_stat = {}
    for j in range(lat_train.shape[1]):
        a = lat_train[:, j] != 0
        if a.sum() < 5:
            continue
        best_corr = -1.0
        best_k = -1
        for k in range(min(16, train_labels.shape[1])):
            valid_nan = ~np.isnan(train_labels[:, k])
            valid = a & valid_nan
            n_valid = valid.sum()
            if n_valid < 5:
                continue
            r = safe_corr(lat_train[valid, j], train_labels[valid, k])
            if r is not None and r > best_corr:
                best_corr = r
                best_k = k
        if best_corr > 0.10 and best_k >= 0:
            feature_stat[j] = best_k
    return feature_stat


def score_family(lat, labels, family_indices, feature_stat):
    """Score a family: mean |r| across family statistics. Returns None on failure."""
    family_feats = [j for j, k in feature_stat.items() if k in family_indices]
    if len(family_feats) < 3:
        return None
    family_act = np.abs(lat[:, family_feats]).mean(axis=1)
    scores = []
    for idx in family_indices:
        if idx >= labels.shape[1]:
            continue
        valid = ~np.isnan(labels[:, idx]) & ~np.isnan(family_act)
        if valid.sum() < 10:
            continue
        r = safe_corr(family_act[valid], labels[valid, idx])
        if r is not None:
            scores.append(r)
    return float(np.mean(scores)) if scores else None


def main():
    model, tok, cfg, d_model, d_hidden = get_model()
    print(f"Model loaded, layers: {LAYERS}")

    if os.path.exists(STATE):
        with open(STATE) as f:
            s = json.load(f)
        completed = set(s.get("completed", []))
    else:
        completed = set()

    all_csvs = get_all_csvs()
    t0 = time.time()

    # Phase 1: collect per-stock validation scores for each layer+family
    layer_family_scores = {layer: defaultdict(list) for layer in LAYERS}

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
        if n_tr < 10 or n_val < 5 or n_test < 5:
            completed.add(ticker)
            continue

        all_labels = compute_all_labels(dn, len(wins))
        train_labels = all_labels[:n_tr]
        val_labels = all_labels[n_tr:n_tr+n_val]

        # Extract per-layer activations
        for layer in LAYERS:
            acts_tr = extract_acts_at_layer(wins[:n_tr], n_tr, layer)
            acts_val = extract_acts_at_layer(wins[n_tr:n_tr+n_val], n_val, layer)

            if len(acts_tr) < 10 or len(acts_val) < 5:
                continue

            # Train SAE on this layer's activations
            sae = train_sae(acts_tr)
            lat_tr = encode_sae(sae, acts_tr)
            lat_val = encode_sae(sae, acts_val)

            m_tr = min(len(lat_tr), len(train_labels))
            m_val = min(len(lat_val), len(val_labels))

            feature_stat = label_features(lat_tr[:m_tr], train_labels[:m_tr])

            for family, indices in FAMILIES.items():
                sc = score_family(lat_val[:m_val], val_labels[:m_val], indices, feature_stat)
                if sc is not None:
                    layer_family_scores[layer][family].append(sc)

            del sae

        completed.add(ticker)
        if (fi + 1) % 20 == 0:
            print(f"[Phase1 {len(completed)}/{len(all_csvs)}] {ticker}")
            with open(STATE, "w") as f:
                json.dump({"completed": list(completed)}, f)

    # Select best layer per family
    best_layer = {}
    for family in FAMILIES:
        best_score = -1.0
        best_l = -1
        for layer in LAYERS:
            scores = layer_family_scores[layer].get(family, [])
            if scores:
                s = float(np.mean(scores))
                if s > best_score:
                    best_score = s
                    best_l = layer
        if best_l >= 0:
            best_layer[family] = best_l

    print(f"\nBest layers per family: {best_layer}")

    # Phase 2: Held-out test with best layer
    test_results = []
    for family, layer in best_layer.items():
        family_indices = FAMILIES[family]
        test_scores = []
        n_stocks_tested = 0

        for fname in all_csvs:
            ticker = fname.replace(".csv", "")
            loaded = load_stock(fname)
            if loaded is None:
                continue
            wins, dn = loaded
            n_tr2 = int(len(wins) * TRAIN_SPLIT)
            n_val2 = int(len(wins) * VAL_SPLIT)
            n_te2 = len(wins) - n_tr2 - n_val2
            if n_te2 < 5 or n_tr2 < 10:
                continue

            all_labels = compute_all_labels(dn, len(wins))
            train_labels = all_labels[:n_tr2]
            test_labels = all_labels[n_tr2+n_val2:]
            m_te = min(len(test_labels), n_te2)

            train_acts = extract_acts_at_layer(wins[:n_tr2], n_tr2, layer)
            test_wins = wins[n_tr2+n_val2:]
            test_acts = extract_acts_at_layer(test_wins, m_te, layer)

            if len(train_acts) < 10 or len(test_acts) < 5:
                continue

            try:
                sae = train_sae(train_acts)
                lat_tr = encode_sae(sae, train_acts)
                lat_te = encode_sae(sae, test_acts)
            except Exception:
                continue

            m_tr2 = min(len(lat_tr), len(train_labels))
            fs = label_features(lat_tr[:m_tr2], train_labels[:m_tr2])

            sc = score_family(lat_te[:m_te], test_labels[:m_te], family_indices, fs)
            if sc is not None:
                test_scores.append(sc)
                n_stocks_tested += 1

            del sae

        mean_score = float(np.mean(test_scores)) if test_scores else 0.0
        test_results.append({
            "family": family,
            "selected_layer": int(layer),
            "held_out_mean_abs_r": mean_score,
            "n_stocks_tested": n_stocks_tested,
            "n_scores": len(test_scores),
        })

    final = {
        "experiment": "multilayer_concept_localization",
        "best_layer_per_family": {f: int(l) for f, l in best_layer.items()},
        "held_out_results": test_results,
    }

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2)
    with open(STATE, "w") as f:
        json.dump({"completed": list(completed)}, f)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"{'Concept':20s} {'Layer':>5s} {'Held-out |r|':>14s} {'N':>5s}")
    print("-" * 48)
    for r in test_results:
        print(f"{r['family']:20s} {r['selected_layer']:>5d} {r['held_out_mean_abs_r']:>14.4f} {r['n_stocks_tested']:>5d}")


if __name__ == "__main__":
    main()
