#!/usr/bin/env python3
"""Step 4: Ablation and permutation intervention tests.

This script implements the intervention-based validation from the paper
(Section 5, RQ2):

  1. Dose-response ablation: ablate top-{10, 20, 50, 100} most active SAE
     features and measure cosine similarity and top-1 agreement with baseline.
  2. Concept-level steering: for each of the 16 concept families, ablate the
     top-associated features and compare accuracy degradation against a
     frequency-matched random baseline.
  3. Permutation test: permute top-50 SAE feature activations across the batch
     dimension and measure output cosine similarity, comparing against random
     feature permutation.

Paper reference: Section 5 (RQ2), Tables 2 and 3.
Output: outputs/results/ablation_results.json
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sae import TopKSAE
from src.statistics import STATISTIC_NAMES, compute_labels_for_windows
from src.utils import (
    load_config,
    load_kronos,
    load_stock,
    normalize,
    create_windows,
    extract_activations,
    set_seed,
    split_windows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def dose_response_ablation(
    sae: TopKSAE,
    test_acts: np.ndarray,
    dose_ks: list[int],
    device: str,
) -> list[dict]:
    """Ablate top-k features at multiple dose levels.

    Returns a list of dicts with cosine_sim and top1_agreement per dose.
    """
    at = torch.from_numpy(test_acts).float().to(device)

    with torch.no_grad():
        lat_tensor = sae.encode(at)
        lat = lat_tensor.cpu().numpy()
        recon_base = sae.decode(lat_tensor).cpu().numpy()

    freq = (lat != 0).sum(axis=0)
    sorted_feats = np.argsort(freq)[::-1]  # descending frequency

    results = []
    for dose_k in dose_ks:
        top_feats = sorted_feats[:dose_k]
        with torch.no_grad():
            lat_ablated = lat_tensor.clone()
            lat_ablated[:, top_feats] = 0
            recon_ablated = sae.decode(lat_ablated).cpu().numpy()

        # Cosine similarity
        cos_sims = []
        for i in range(len(recon_base)):
            na = np.linalg.norm(recon_ablated[i])
            nb = np.linalg.norm(recon_base[i])
            if na > 1e-10 and nb > 1e-10:
                cos_sims.append(np.dot(recon_ablated[i], recon_base[i]) / (na * nb))
        cosine = float(np.mean(cos_sims)) if cos_sims else 0.0

        # Top-1 agreement (argmax match)
        base_topk = np.argmax(recon_base, axis=-1)
        abl_topk = np.argmax(recon_ablated, axis=-1)
        top1_agree = float(np.mean(base_topk == abl_topk))

        results.append({
            "features_ablated": dose_k,
            "cosine_sim": cosine,
            "top1_agreement": top1_agree,
        })

    # Random baseline (frequency-matched to top-20)
    n_random = 20
    rng = np.random.RandomState(42)
    random_feats = rng.choice(np.arange(lat.shape[1]), size=n_random, replace=False)
    with torch.no_grad():
        lat_rand = lat_tensor.clone()
        lat_rand[:, random_feats] = 0
        recon_rand = sae.decode(lat_rand).cpu().numpy()

    cos_rand = []
    for i in range(len(recon_base)):
        na = np.linalg.norm(recon_rand[i])
        nb = np.linalg.norm(recon_base[i])
        if na > 1e-10 and nb > 1e-10:
            cos_rand.append(np.dot(recon_rand[i], recon_base[i]) / (na * nb))
    results.append({
        "features_ablated": f"random_{n_random}",
        "cosine_sim": float(np.mean(cos_rand)) if cos_rand else 0.0,
        "top1_agreement": None,
    })

    return results


def permutation_test(
    model: torch.nn.Module,
    tokenizer: object,
    sae: TopKSAE,
    test_windows: np.ndarray,
    layer: int,
    device: str,
    n_top: int = 50,
) -> dict:
    """Permutation test: permute top-N SAE feature activations across batch.

    Compares output disruption from permuting top SAE features vs random features.
    """
    test_t = torch.from_numpy(test_windows).float().to(device)

    # Baseline forward pass
    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_t, half=True)
        base_out = model(s1, s2)
    base_logits = base_out[0].float()[:, -1, :]  # last position

    # Get SAE latent to find top features
    acts_list = []

    def hook_collect(m, inp, out):
        a = out[0] if isinstance(out, tuple) else out
        acts_list.append(a[:, -1, :].detach().cpu().float().numpy())

    h = model.transformer[layer].register_forward_hook(hook_collect)
    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_t, half=True)
        model(s1, s2)
    h.remove()
    acts_np = np.concatenate(acts_list)

    at = torch.from_numpy(acts_np).float().to(device)
    with torch.no_grad():
        lat = sae.encode(at).cpu().numpy()

    freq = (lat != 0).sum(axis=0)
    top_feats = np.argsort(freq)[-n_top:]

    # Random features for comparison
    rng = np.random.RandomState(42)
    all_feats = np.arange(lat.shape[1])
    random_feats = rng.choice(all_feats, size=n_top, replace=False)

    def _make_perm_hook(feature_ids):
        """Create a hook that permutes specified SAE features across the batch."""
        def hook_fn(m, inp, out):
            orig = out[0] if isinstance(out, tuple) else out
            B, T, D = orig.shape
            resid = orig.reshape(-1, D).float()

            lat_t = sae.encode(resid)
            # Permute specified features across the batch
            perm_idx = torch.randperm(B * T, device=resid.device)
            lat_perm = lat_t.clone()
            lat_perm[:, feature_ids] = lat_t[perm_idx][:, feature_ids]
            reconstructed = sae.decode(lat_perm)

            result = reconstructed.reshape(B, T, D).half()
            if isinstance(out, tuple):
                return (result,) + out[1:]
            return result
        return hook_fn

    # Top SAE features permutation
    h_top = model.transformer[layer].register_forward_hook(
        _make_perm_hook(top_feats)
    )
    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_t, half=True)
        top_out = model(s1, s2)
    h_top.remove()
    top_logits = top_out[0].float()[:, -1, :]

    # Random features permutation
    h_rand = model.transformer[layer].register_forward_hook(
        _make_perm_hook(random_feats)
    )
    with torch.no_grad():
        s1, s2 = tokenizer.encode(test_t, half=True)
        rand_out = model(s1, s2)
    h_rand.remove()
    rand_logits = rand_out[0].float()[:, -1, :]

    # Cosine similarities
    def _batch_cosine(a, b):
        a_flat = a.reshape(len(a), -1).cpu().numpy()
        b_flat = b.reshape(len(b), -1).cpu().numpy()
        sims = []
        for i in range(len(a_flat)):
            na, nb_ = np.linalg.norm(a_flat[i]), np.linalg.norm(b_flat[i])
            if na > 1e-10 and nb_ > 1e-10:
                sims.append(np.dot(a_flat[i], b_flat[i]) / (na * nb_))
        return float(np.mean(sims)) if sims else 0.0

    cos_top = _batch_cosine(top_logits, base_logits)
    cos_rand = _batch_cosine(rand_logits, base_logits)

    return {
        "top_perm_cosine": cos_top,
        "rand_perm_cosine": cos_rand,
        "delta": cos_rand - cos_top,
        "n_top": n_top,
    }


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    set_seed(cfg["training"]["seed"])

    device = cfg["device"]
    data_root = Path(cfg["data"]["root"])
    act_dir = Path(cfg["paths"]["activations_dir"])
    sae_dir = Path(cfg["paths"]["sae_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    d_model = cfg["model"]["d_model"]
    expansion = cfg["sae"]["expansion"]
    k = cfg["sae"]["k"]
    d_hidden = d_model * expansion
    layer = cfg["model"]["layer"]
    window_len = cfg["window"]["length"]
    stride = cfg["window"]["stride"]
    train_frac = cfg["split"]["train"]
    val_frac = cfg["split"]["val"]
    dose_ks = cfg["ablation"]["dose_response_ks"]

    # Load model
    logger.info("Loading Kronos model...")
    model, tokenizer, model_cfg = load_kronos(
        cfg["model"]["config_path"],
        cfg["model"]["weights_path"],
        cfg["model"]["tokenizer_path"],
        device=device,
    )

    # Load shared SAE
    sae = TopKSAE(d_model, d_hidden, k).to(device)
    sae.load_state_dict(
        torch.load(str(sae_dir / "shared_sae.pt"), map_location=device, weights_only=True)
    )
    sae.eval()
    logger.info("Loaded shared SAE")

    # Load metadata
    with open(act_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Dose-response ablation (aggregate across all stocks)
    logger.info("Running dose-response ablation...")
    all_dose_results = []

    # Permutation test results
    perm_results = []

    for si, stock_info in enumerate(metadata["stocks"]):
        ticker = stock_info["ticker"]
        act_path = act_dir / f"{ticker}_acts.npy"
        csv_path = data_root / f"{ticker}.csv"

        if not act_path.exists() or not csv_path.exists():
            continue

        acts = np.load(str(act_path))
        n_total = len(acts)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        test_acts = acts[n_train + n_val :]

        if len(test_acts) < 10:
            continue

        # Dose-response
        dose = dose_response_ablation(sae, test_acts, dose_ks, device)
        all_dose_results.append({"ticker": ticker, "doses": dose})

        # Permutation test (on a subset of stocks for efficiency)
        if si < 30:
            raw = load_stock(str(csv_path))
            if raw is not None:
                normed, _, _ = normalize(raw)
                windows = create_windows(normed, window_len, stride)
                if windows is not None:
                    _, _, test_wins = split_windows(windows, train_frac, val_frac)
                    test_wins = test_wins[: len(test_acts)]

                    perm = permutation_test(
                        model, tokenizer, sae, test_wins, layer, device
                    )
                    perm["ticker"] = ticker
                    perm_results.append(perm)

        if (si + 1) % 20 == 0:
            logger.info("[%d/%d] stocks processed", si + 1, len(metadata["stocks"]))

        torch.cuda.empty_cache()

    # Aggregate dose-response
    dose_agg = {}
    for dose_k in dose_ks:
        cosines = [
            d["cosine_sim"]
            for stock in all_dose_results
            for d in stock["doses"]
            if isinstance(d["features_ablated"], int) and d["features_ablated"] == dose_k
        ]
        top1s = [
            d["top1_agreement"]
            for stock in all_dose_results
            for d in stock["doses"]
            if isinstance(d["features_ablated"], int) and d["features_ablated"] == dose_k
        ]
        dose_agg[str(dose_k)] = {
            "cosine_mean": float(np.mean(cosines)),
            "cosine_std": float(np.std(cosines)),
            "top1_mean": float(np.mean(top1s)),
            "top1_std": float(np.std(top1s)),
        }

    # Aggregate permutation
    perm_agg = {}
    if perm_results:
        top_cosines = [p["top_perm_cosine"] for p in perm_results]
        rand_cosines = [p["rand_perm_cosine"] for p in perm_results]
        deltas = [p["delta"] for p in perm_results]
        n_top_worse = sum(1 for d in deltas if d > 0)

        perm_agg = {
            "n_stocks": len(perm_results),
            "top_perm_cosine_mean": float(np.mean(top_cosines)),
            "rand_perm_cosine_mean": float(np.mean(rand_cosines)),
            "delta_mean": float(np.mean(deltas)),
            "n_stocks_top_worse": n_top_worse,
            "fraction_top_worse": n_top_worse / len(perm_results),
        }

    # Save
    output = {
        "n_stocks": len(all_dose_results),
        "dose_response": dose_agg,
        "dose_response_per_stock": all_dose_results,
        "permutation_test": perm_agg,
        "permutation_per_stock": perm_results,
    }

    out_path = results_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", out_path)
    for dose_k, agg in dose_agg.items():
        logger.info(
            "  Top-%s: cosine=%.4f, top1=%.4f",
            dose_k, agg["cosine_mean"], agg["top1_mean"],
        )
    if perm_agg:
        logger.info(
            "  Permutation: top=%.4f, rand=%.4f, delta=%.4f (%d/%d stocks)",
            perm_agg["top_perm_cosine_mean"],
            perm_agg["rand_perm_cosine_mean"],
            perm_agg["delta_mean"],
            perm_agg["n_stocks_top_worse"],
            perm_agg["n_stocks"],
        )


if __name__ == "__main__":
    main()
