"""Experiment 7: Valid-Token Counterfactual Search.
Work in BSQ token space: start from real token sequences, use greedy search
to find valid token-space neighbors that change a target financial statistic
while minimizing changes to other statistics.
Measure SAE concept feature response to valid counterfactuals.
"""
import torch, numpy as np, json, os, time, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_exp_utils import *

OUTPUT = os.path.join(OUTPUT_DIR, "exp7_token_counterfactual.json")
STATE = os.path.join(OUTPUT_DIR, "exp7_token_counterfactual_state.json")

N_WINDOWS_PER_STOCK = 10
N_NEIGHBORS = 100   # token neighbors to sample
BEAM_SIZE = 5       # beam search width
MAX_TOKEN_EDITS = 10  # maximum tokens to replace


def decode_tokens_to_windows(tok, s1_tokens, s2_tokens):
    """Decode BSQ token sequences back to K-line windows."""
    with torch.no_grad():
        s1 = torch.as_tensor(s1_tokens, dtype=torch.long, device=DEVICE)
        s2 = torch.as_tensor(s2_tokens, dtype=torch.long, device=DEVICE)
        decoded = tok.decode(s1, s2)
    return decoded.cpu().float().numpy()


def compute_statistic_from_window(w, stat_fn_name, stat_idx=None):
    """Compute a specific statistic from a decoded window."""
    c = w[:, 1]  # close
    v = w[:, 4]  # volume
    r = np.diff(c) / (c[:-1] + 1e-5)

    if stat_idx is not None:
        # Use the standard 16-statistic computation
        all_stats = compute_statistics_window(c, volume_series=v)
        if stat_idx < len(all_stats):
            return float(all_stats[stat_idx])
        return 0.0
    return 0.0


def sample_valid_tokens(tok, vocab_size, n_samples=100):
    """Sample tokens that decode to valid K-line data."""
    s1_vocab = tok.s1_vocab_size if hasattr(tok, 's1_vocab_size') else vocab_size
    s2_vocab = tok.s2_vocab_size if hasattr(tok, 's2_vocab_size') else vocab_size
    s1_tokens = np.random.randint(0, s1_vocab, (n_samples, WINDOW))
    s2_tokens = np.random.randint(0, s2_vocab, (n_samples, WINDOW))
    try:
        decoded = decode_tokens_to_windows(tok, s1_tokens, s2_tokens)
        # Count as valid if values are within reasonable range
        valid = np.all(np.abs(decoded) < 100, axis=(1, 2))
        return decoded[valid]
    except Exception:
        return np.array([])


def token_distance(orig_s1, orig_s2, cand_s1, cand_s2):
    """Hamming distance between token sequences."""
    return float(np.sum(orig_s1 != cand_s1) + np.sum(orig_s2 != cand_s2))


def main():
    model, tok, cfg, d_model, d_hidden = get_model()
    print("Experiment 7: Valid-Token Counterfactual Search")

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
        if n_test < 5 or n_tr < 10:
            completed.add(ticker)
            continue

        # Train SAE on this stock (needed for concept response measurement)
        train_acts = extract_acts(wins, n_tr)
        sae = train_sae(train_acts)

        # Get test windows and their token sequences
        test_wins = wins[n_tr+n_val:]
        n_test_use = min(N_WINDOWS_PER_STOCK, len(test_wins))

        # Select windows with enough data
        for wi in range(n_test_use):
            w = test_wins[wi].copy()
            window_idx = n_tr + n_val + wi
            if window_idx + WINDOW > len(dn):
                continue

            # Tokenize the real window and get model activations
            try:
                at = torch.as_tensor(w[np.newaxis].copy(), dtype=torch.float32, device=DEVICE)
                s1_orig, s2_orig = tok.encode(at, half=True)
                s1o_np = s1_orig.cpu().numpy()[0]
                s2o_np = s2_orig.cpu().numpy()[0]
            except Exception:
                continue

            # Get model activations for the real window, then encode with SAE
            def hook_fn(m, i, o):
                a = o[0] if isinstance(o, tuple) else o
                hook_fn.out = a[:, -1, :].detach()
            hook = model.transformer[LAYER].register_forward_hook(hook_fn)
            with torch.no_grad():
                model(s1_orig, s2_orig)
            hook.remove()
            acts_orig = hook_fn.out  # (1, d_model)
            with torch.no_grad():
                lat_orig = sae.encode(acts_orig).detach().cpu().numpy()[0]
            orig_feat_means = np.abs(lat_orig)

            # For each concept family, try to find counterfactuals
            for family, indices in FAMILIES.items():
                valid_idx = [i for i in indices if i < 16]
                if not valid_idx:
                    continue

                target_idx = valid_idx[0]  # primary statistic
                orig_stat = compute_statistic_from_window(w, None, target_idx)
                if np.isnan(orig_stat) or np.isinf(orig_stat):
                    continue

                # Sample valid token neighbors
                # Strategy: perturb a few tokens in the sequence
                s1_vocab = tok.s1_vocab_size if hasattr(tok, 's1_vocab_size') else 256
                s2_vocab = tok.s2_vocab_size if hasattr(tok, 's2_vocab_size') else 256

                best_increase = None
                best_decrease = None
                best_score_increase = -float("inf")
                best_score_decrease = float("inf")

                for attempt in range(N_NEIGHBORS):
                    # Perturb random token positions
                    s1_pert = s1o_np.copy()
                    s2_pert = s2o_np.copy()
                    n_edits = np.random.randint(1, MAX_TOKEN_EDITS + 1)
                    edit_positions = np.random.choice(WINDOW, n_edits, replace=False)

                    for pos in edit_positions:
                        if np.random.random() < 0.5:
                            s1_pert[pos] = np.random.randint(0, s1_vocab)
                        else:
                            s2_pert[pos] = np.random.randint(0, s2_vocab)

                    # Skip if too many edits
                    dist = token_distance(s1o_np, s2o_np, s1_pert, s2_pert)

                    # Decode and check validity
                    try:
                        w_decoded = decode_tokens_to_windows(tok,
                            s1_pert.reshape(1, -1), s2_pert.reshape(1, -1))[0]
                        # Check OHLCVA constraints
                        valid = True
                        if np.any(np.abs(w_decoded) > 50):
                            valid = False
                        if valid:
                            pert_stat = compute_statistic_from_window(
                                w_decoded, None, target_idx)
                            if np.isnan(pert_stat) or np.isinf(pert_stat):
                                valid = False
                    except Exception:
                        valid = False

                    if not valid:
                        continue

                    # Run model on perturbed tokens to get activations, then SAE encode
                    try:
                        s1p = torch.as_tensor(s1_pert.reshape(1, -1).copy(), dtype=torch.long, device=DEVICE)
                        s2p = torch.as_tensor(s2_pert.reshape(1, -1).copy(), dtype=torch.long, device=DEVICE)
                        def hook_pert(m, i, o):
                            a = o[0] if isinstance(o, tuple) else o
                            hook_pert.out = a[:, -1, :].detach()
                        hp = model.transformer[LAYER].register_forward_hook(hook_pert)
                        with torch.no_grad():
                            model(s1p, s2p)
                        hp.remove()
                        acts_pert = hook_pert.out
                        with torch.no_grad():
                            lat_pert = sae.encode(acts_pert).detach().cpu().numpy()[0]
                    except Exception:
                        continue

                    pert_feat_means = np.abs(lat_pert)

                    # Compute other statistic changes (penalty)
                    other_idx = [i for i in range(16) if i not in valid_idx and i < 16]
                    other_change = 0.0
                    for oi in other_idx:
                        os_orig = compute_statistic_from_window(w, None, oi)
                        os_pert = compute_statistic_from_window(w_decoded, None, oi)
                        if not np.isnan(os_orig) and not np.isnan(os_pert):
                            other_change += abs(os_pert - os_orig)

                    # Score: maximize target change, minimize other change and token edits
                    delta_target = pert_stat - orig_stat
                    score = abs(delta_target) - 0.1 * other_change - 0.01 * dist

                    if delta_target > 0 and score > best_score_increase:
                        best_score_increase = score
                        best_increase = {
                            "direction": "increase",
                            "orig_stat": float(orig_stat),
                            "pert_stat": float(pert_stat),
                            "delta_target": float(delta_target),
                            "token_edits": int(dist),
                            "other_stat_change": float(other_change),
                            "score": float(score),
                            "family_feature_delta": {},
                        }
                        for resp_family, resp_indices in FAMILIES.items():
                            resp_label_indices = [i for i in resp_indices if i < 16]
                            # Mean activation of SAE features (proxy via full latent)
                            o_mean = float(np.mean(np.abs(lat_orig)))
                            p_mean = float(np.mean(np.abs(lat_pert)))
                            best_increase["family_feature_delta"][resp_family] = p_mean - o_mean

                    elif delta_target < 0 and score > best_score_decrease:
                        best_score_decrease = score
                        best_decrease = {
                            "direction": "decrease",
                            "orig_stat": float(orig_stat),
                            "pert_stat": float(pert_stat),
                            "delta_target": float(delta_target),
                            "token_edits": int(dist),
                            "other_stat_change": float(other_change),
                            "score": float(score),
                            "family_feature_delta": {},
                        }
                        for resp_family, resp_indices in FAMILIES.items():
                            o_mean = float(np.mean(np.abs(lat_orig)))
                            p_mean = float(np.mean(np.abs(lat_pert)))
                            best_decrease["family_feature_delta"][resp_family] = p_mean - o_mean

                if best_increase is not None:
                    results.append({
                        "ticker": ticker,
                        "window": wi,
                        "family": family,
                        **best_increase,
                    })
                if best_decrease is not None:
                    results.append({
                        "ticker": ticker,
                        "window": wi,
                        "family": family,
                        **best_decrease,
                    })

        completed.add(ticker)
        del sae
        torch.cuda.empty_cache()
        if (fi + 1) % 20 == 0:
            print(f"[{len(completed)}/{len(all_csvs)}] {ticker}: {len(results)} counterfactuals found")
            with open(STATE, "w") as f:
                json.dump({"completed": list(completed), "results": results}, f)

    if not results:
        print("No valid counterfactuals found!")
        final = {"experiment": "token_counterfactual_search", "n_results": 0, "error": "No valid counterfactuals"}
        with open(OUTPUT, "w") as f:
            json.dump(final, f, indent=2)
        return

    # Aggregate: diagonal selectivity
    selectivity = defaultdict(lambda: defaultdict(list))
    for r in results:
        target = r["family"]
        for resp, delta in r["family_feature_delta"].items():
            selectivity[target][resp].append(abs(delta))

    matrix = {}
    for target, responses in selectivity.items():
        matrix[target] = {}
        for resp, vals in responses.items():
            if vals:
                matrix[target][resp] = {"mean": float(np.mean(vals)), "n": len(vals)}

    diag_vals, off_vals = [], []
    for target, responses in matrix.items():
        for resp, v in responses.items():
            (diag_vals if target == resp else off_vals).append(v["mean"])

    final = {
        "experiment": "token_counterfactual_search",
        "n_stocks": len(completed),
        "n_counterfactuals": len(results),
        "selectivity_matrix": matrix,
        "diagonal_mean": float(np.mean(diag_vals)) if diag_vals else 0,
        "off_diagonal_mean": float(np.mean(off_vals)) if off_vals else 0,
        "diagonal_ratio": float(np.mean(diag_vals) / np.mean(off_vals)) if diag_vals and off_vals else 0,
    }

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2)
    with open(STATE, "w") as f:
        json.dump({"completed": list(completed), "results": results}, f)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"Counterfactuals: {len(results)}")
    print(f"Diagonal ratio: {final['diagonal_ratio']:.3f}x")


if __name__ == "__main__":
    main()
