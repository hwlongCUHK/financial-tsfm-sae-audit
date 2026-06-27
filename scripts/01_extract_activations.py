#!/usr/bin/env python3
"""Step 1: Extract Kronos layer-6 residual stream activations.

For each of the 120 stock CSVs, this script:
  1. Loads and z-score normalizes the OHLCVA data.
  2. Creates 64-period sliding windows with stride 32.
  3. Runs each window through Kronos and hooks the layer-6 residual stream.
  4. Saves the final-token activations to .npy files.

Paper reference: Section 3 (Method -- Model and Data).
Output: outputs/activations/{ticker}_acts.npy  (one per stock)
        outputs/activations/metadata.json       (stock list and shapes)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    set_seed(cfg["training"]["seed"])

    device = cfg["device"]
    data_root = Path(cfg["data"]["root"])
    output_dir = Path(cfg["paths"]["activations_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    layer = cfg["model"]["layer"]
    window_len = cfg["window"]["length"]
    stride = cfg["window"]["stride"]
    train_frac = cfg["split"]["train"]
    val_frac = cfg["split"]["val"]

    # Load Kronos model
    logger.info("Loading Kronos model...")
    model, tokenizer, model_cfg = load_kronos(
        cfg["model"]["config_path"],
        cfg["model"]["weights_path"],
        cfg["model"]["tokenizer_path"],
        device=device,
    )
    d_model = model_cfg["d_model"]
    logger.info("d_model=%d, layer=%d", d_model, layer)

    # Process all stocks
    all_csvs = sorted([f for f in os.listdir(str(data_root)) if f.endswith(".csv")])
    logger.info("Found %d CSV files in %s", len(all_csvs), data_root)

    metadata = {"d_model": d_model, "layer": layer, "stocks": []}
    t_start = time.time()

    for i, fname in enumerate(all_csvs):
        ticker = fname.replace(".csv", "")
        out_path = output_dir / f"{ticker}_acts.npy"

        # Skip if already extracted
        if out_path.exists():
            logger.info("[%d/%d] %s: already exists, skipping", i + 1, len(all_csvs), ticker)
            metadata["stocks"].append({"ticker": ticker, "status": "cached"})
            continue

        # Load and normalize
        raw = load_stock(str(data_root / fname))
        if raw is None:
            logger.info("[%d/%d] %s: SKIP (insufficient data)", i + 1, len(all_csvs), ticker)
            continue

        normed, mean, std = normalize(raw)
        windows = create_windows(normed, window_len, stride)
        if windows is None:
            logger.info("[%d/%d] %s: SKIP (too few windows)", i + 1, len(all_csvs), ticker)
            continue

        # Split (we save all activations but record split indices)
        train_wins, val_wins, test_wins = split_windows(windows, train_frac, val_frac)
        n_train = len(train_wins)
        n_val = len(val_wins)
        n_test = len(test_wins)

        # Extract activations for ALL windows
        acts = extract_activations(model, tokenizer, windows, layer=layer, device=device)

        # Save
        np.save(str(out_path), acts)
        metadata["stocks"].append({
            "ticker": ticker,
            "n_windows": len(windows),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "shape": list(acts.shape),
            "status": "extracted",
        })

        logger.info(
            "[%d/%d] %s: %d windows -> acts shape %s",
            i + 1, len(all_csvs), ticker, len(windows), acts.shape,
        )
        torch.cuda.empty_cache()

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t_start
    logger.info(
        "Done. %d stocks processed in %.0fs. Saved to %s",
        len(metadata["stocks"]), elapsed, output_dir,
    )


if __name__ == "__main__":
    main()
