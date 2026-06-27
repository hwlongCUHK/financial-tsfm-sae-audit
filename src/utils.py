"""Utilities for data loading, normalization, windowing, and activation extraction.

All functions operate on individual stock CSV files in the OHLCVA format
(open, high, low, close, volume, amount) used by the Kronos tokenizer.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)

OHLCVA_COLUMNS = ["open", "close", "high", "low", "volume", "amount"]


# =========================================================================
# Configuration
# =========================================================================


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# =========================================================================
# Reproducibility
# =========================================================================


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================
# Data loading and normalization
# =========================================================================


def load_stock(
    filepath: str,
    columns: Optional[list[str]] = None,
) -> Optional[np.ndarray]:
    """Load a stock CSV file and return raw OHLCVA data.

    Missing columns are filled with zeros. Rows with NaN values are dropped.
    Returns None if the file has fewer than 100 valid rows.

    Args:
        filepath: Path to the CSV file.
        columns: Column names to use. Defaults to OHLCVA_COLUMNS.

    Returns:
        Array of shape ``(n_days, 6)`` with dtype float32, or None.
    """
    columns = columns or OHLCVA_COLUMNS

    df = pd.read_csv(filepath)
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0

    data = df[columns].values.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]

    if len(data) < 100:
        logger.warning("%s has only %d rows, skipping", filepath, len(data))
        return None

    return data


def normalize(
    data: np.ndarray,
    clip_range: tuple[float, float] = (-5.0, 5.0),
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize per column, clamp to range.

    Uses only the provided data for computing mean and std (the caller is
    responsible for ensuring this is the training portion to avoid leakage).

    Args:
        data: Array of shape ``(n_samples, n_features)``.
        clip_range: Min and max values for clamping.
        epsilon: Small constant for numerical stability.

    Returns:
        Tuple of (normalized_data, mean, std).
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    normalized = np.clip((data - mean) / (std + epsilon), clip_range[0], clip_range[1])
    return normalized, mean, std


def create_windows(
    data: np.ndarray,
    window_length: int = 64,
    stride: int = 32,
    max_windows: int = 2000,
) -> Optional[np.ndarray]:
    """Create sliding windows from a 2-D array.

    Args:
        data: Array of shape ``(n_timesteps, n_features)``.
        window_length: Number of timesteps per window.
        stride: Step size between consecutive windows.
        max_windows: Maximum number of windows to create.

    Returns:
        Array of shape ``(n_windows, window_length, n_features)``, or None
        if fewer than 25 windows can be created.
    """
    n_possible = (len(data) - window_length) // stride
    n_windows = min(max_windows, n_possible)

    if n_windows < 25:
        logger.warning("Only %d windows available (< 25), skipping", n_windows)
        return None

    windows = np.stack(
        [data[i * stride : i * stride + window_length] for i in range(n_windows)]
    )
    return windows


# =========================================================================
# Activation extraction
# =========================================================================


def extract_activations(
    model: torch.nn.Module,
    tokenizer: object,
    windows: np.ndarray,
    layer: int = 6,
    device: str = "cuda:0",
    batch_size: int = 64,
) -> np.ndarray:
    """Extract residual stream activations at a given layer.

    Hooks into the specified transformer layer and collects the final-token
    hidden state for each input window.

    Args:
        model: The Kronos model (in eval mode on device).
        tokenizer: The KronosTokenizer (on device).
        windows: Array of shape ``(n_windows, window_length, n_features)``.
        layer: Transformer layer index to hook.
        device: Target device string.
        batch_size: Batch size for forward passes.

    Returns:
        Array of shape ``(n_windows, d_model)`` with dtype float32.
    """
    activations: list[np.ndarray] = []

    def hook_fn(module, inputs, output):
        act = output[0] if isinstance(output, tuple) else output
        activations.append(act[:, -1, :].detach().cpu().float().numpy())

    hook = model.transformer[layer].register_forward_hook(hook_fn)

    with torch.no_grad():
        for b_start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(
                windows[b_start : b_start + batch_size]
            ).float().to(device)
            s1, s2 = tokenizer.encode(batch, half=True)
            model(s1, s2)

    hook.remove()
    return np.concatenate(activations, axis=0)


# =========================================================================
# Model loading
# =========================================================================


def load_kronos(
    config_path: str,
    weights_path: str,
    tokenizer_path: str,
    device: str = "cuda:0",
) -> tuple:
    """Load the Kronos model and tokenizer.

    This function imports Kronos and KronosTokenizer from the Kronos package.
    The model is loaded in half precision (fp16) in eval mode.

    Args:
        config_path: Path to Kronos config.json.
        weights_path: Path to model.safetensors.
        tokenizer_path: Path to the tokenizer directory.
        device: Target device.

    Returns:
        Tuple of (model, tokenizer, config_dict).
    """
    from model.kronos import Kronos, KronosTokenizer
    from safetensors.torch import load_file

    # Load tokenizer
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(device).eval()

    # Load model config
    with open(config_path) as f:
        cfg = json.load(f)

    # Build model
    model = Kronos(
        s1_bits=cfg["s1_bits"],
        s2_bits=cfg["s2_bits"],
        n_layers=cfg["n_layers"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        ff_dim=cfg["ff_dim"],
        ffn_dropout_p=cfg["ffn_dropout_p"],
        attn_dropout_p=cfg["attn_dropout_p"],
        resid_dropout_p=cfg["resid_dropout_p"],
        token_dropout_p=cfg["token_dropout_p"],
        learn_te=cfg["learn_te"],
    )

    # Load weights
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).half().eval()

    logger.info(
        "Loaded Kronos: d_model=%d, n_layers=%d, device=%s",
        cfg["d_model"],
        cfg["n_layers"],
        device,
    )

    return model, tokenizer, cfg


# =========================================================================
# Split helpers
# =========================================================================


def split_windows(
    windows: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split windows into train / val / test.

    Args:
        windows: Array of shape ``(n_windows, ...)``.
        train_frac: Fraction of windows for training.
        val_frac: Fraction of windows for validation (purge gap).

    Returns:
        Tuple of (train_windows, val_windows, test_windows).
    """
    n = len(windows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return windows[:n_train], windows[n_train : n_train + n_val], windows[n_train + n_val :]
