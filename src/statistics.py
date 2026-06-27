"""Financial statistics computation for SAE feature labeling.

For each 64-period K-line window, we compute 30+ precisely defined financial
measures spanning six categories: Momentum, Volatility, Autocorrelation,
Tail Risk, Price Structure, and Volume. These statistics serve as the
ground-truth labels for correlation-based SAE feature labeling.

The statistics are computed on the *normalized* window data (z-scored,
clipped to [-5, 5]). Column indices follow the OHLCVA convention:
    0=open, 1=close, 2=high, 3=low, 4=volume, 5=amount.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The 16 primary statistic families used in the paper's concept labeling.
# These are a subset of the 30+ statistics, selected as the best-correlated
# representative of each concept family.
STATISTIC_NAMES: list[str] = [
    "momentum_5",
    "trend",
    "volatility",
    "vol_persistence",
    "autocorr_lag1",
    "autocorr_lag5",
    "max_drawdown",
    "var_95",
    "max_1day_gain",
    "max_1day_loss",
    "skewness",
    "kurtosis",
    "price_range",
    "vol_clustering",
    "volume_trend",
    "volume_price_corr",
]

# Full list of all 30+ statistics computed per window.
ALL_STATISTIC_NAMES: list[str] = [
    # Momentum (6)
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "momentum_64",
    "ma_crossover",
    "rsi_like",
    # Volatility (6)
    "volatility",
    "parkinson_vol",
    "garman_klass_vol",
    "vol_of_vol",
    "vol_persistence",
    "vol_clustering",
    # Autocorrelation (3)
    "autocorr_lag1",
    "autocorr_lag5",
    "hurst_estimate",
    # Tail Risk (6)
    "var_95",
    "cvar_95",
    "max_1day_gain",
    "max_1day_loss",
    "skewness",
    "kurtosis",
    # Price Structure (5)
    "trend",
    "max_drawdown",
    "price_range",
    "close_to_close_range",
    "jarque_bera",
    # Volume (3)
    "volume_trend",
    "volume_volatility",
    "volume_price_corr",
]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient, returning 0 on degenerate input."""
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def compute_financial_statistics(
    window: np.ndarray,
    return_all: bool = False,
) -> np.ndarray:
    """Compute financial statistics for a single K-line window.

    Args:
        window: Array of shape ``(window_length, 6)`` with columns
            [open, close, high, low, volume, amount].
        return_all: If True, return all 30+ statistics. If False (default),
            return only the 16 primary statistics used for concept labeling.

    Returns:
        1-D array of statistic values.
    """
    close = window[:, 1]
    high = window[:, 2]
    low = window[:, 3]
    volume = window[:, 4]
    n = len(close)

    # Returns (close-to-close)
    returns = np.diff(close) / (close[:-1] + 1e-5)
    abs_returns = np.abs(returns)

    # =====================================================================
    # Momentum
    # =====================================================================

    def momentum_5() -> float:
        """5-period momentum: close[-1] / close[-6] - 1."""
        if n >= 6:
            return float(close[-1] / close[-6] - 1.0)
        return 0.0

    def momentum_10() -> float:
        """10-period momentum."""
        if n >= 11:
            return float(close[-1] / close[-11] - 1.0)
        return 0.0

    def momentum_20() -> float:
        """20-period momentum."""
        if n >= 21:
            return float(close[-1] / close[-21] - 1.0)
        return 0.0

    def momentum_64() -> float:
        """Full-window momentum: close[-1] / close[0] - 1."""
        return float(close[-1] / (close[0] + 1e-5) - 1.0)

    def ma_crossover() -> float:
        """Moving average crossover signal: MA(5) - MA(20), normalized."""
        if n >= 20:
            ma5 = np.mean(close[-5:])
            ma20 = np.mean(close[-20:])
            return float((ma5 - ma20) / (ma20 + 1e-5))
        return 0.0

    def rsi_like() -> float:
        """RSI-like oscillator: fraction of positive returns over the window."""
        if len(returns) == 0:
            return 0.5
        return float(np.mean(returns > 0))

    # =====================================================================
    # Volatility
    # =====================================================================

    def volatility() -> float:
        """Realized volatility: standard deviation of returns."""
        return float(np.std(returns))

    def parkinson_vol() -> float:
        """Parkinson volatility estimator using high-low range."""
        log_hl = np.log(high / (low + 1e-5) + 1e-5)
        return float(np.sqrt(np.mean(log_hl ** 2) / (4.0 * np.log(2.0))))

    def garman_klass_vol() -> float:
        """Garman-Klass volatility estimator."""
        o = window[:, 0]
        c = window[:, 1]
        log_hl = np.log(high / (low + 1e-5) + 1e-5)
        log_co = np.log(c / (o + 1e-5) + 1e-5)
        gk = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
        return float(np.sqrt(np.mean(gk)))

    def vol_of_vol() -> float:
        """Volatility of volatility: std of rolling 5-period volatility."""
        if len(returns) < 10:
            return 0.0
        rolling_vol = pd.Series(returns).rolling(5).std().dropna().values
        return float(np.std(rolling_vol)) if len(rolling_vol) > 1 else 0.0

    def vol_persistence() -> float:
        """Volatility persistence: autocorrelation of |returns|."""
        if len(returns) < 3:
            return 0.0
        return _safe_corr(abs_returns[1:], abs_returns[:-1])

    def vol_clustering() -> float:
        """Volatility clustering: mean(r^2) / var(r)."""
        var = np.var(returns)
        return float(np.mean(returns ** 2) / (var + 1e-10))

    # =====================================================================
    # Autocorrelation
    # =====================================================================

    def autocorr_lag1() -> float:
        """Lag-1 return autocorrelation."""
        if len(returns) < 3:
            return 0.0
        return _safe_corr(returns[1:], returns[:-1])

    def autocorr_lag5() -> float:
        """Lag-5 return autocorrelation."""
        if len(returns) < 7:
            return 0.0
        return _safe_corr(returns[5:], returns[:-5])

    def hurst_estimate() -> float:
        """Simple Hurst exponent estimate via rescaled range."""
        if len(returns) < 10:
            return 0.5
        series = returns - np.mean(returns)
        cumdev = np.cumsum(series)
        r = np.max(cumdev) - np.min(cumdev)
        s = np.std(returns)
        if s < 1e-10 or r < 1e-10:
            return 0.5
        return float(np.log(r / s) / np.log(len(returns)))

    # =====================================================================
    # Tail Risk
    # =====================================================================

    def var_95() -> float:
        """Value at Risk at 95% confidence: 5th percentile of returns."""
        return float(np.percentile(returns, 5))

    def cvar_95() -> float:
        """Conditional VaR (Expected Shortfall) at 95%."""
        threshold = np.percentile(returns, 5)
        tail = returns[returns <= threshold]
        return float(np.mean(tail)) if len(tail) > 0 else float(threshold)

    def max_1day_gain() -> float:
        """Maximum single-period return."""
        return float(np.max(returns))

    def max_1day_loss() -> float:
        """Minimum single-period return (worst loss)."""
        return float(np.min(returns))

    def skewness() -> float:
        """Return distribution skewness."""
        if len(returns) < 3:
            return 0.0
        return float(pd.Series(returns).skew())

    def kurtosis() -> float:
        """Return distribution excess kurtosis."""
        if len(returns) < 4:
            return 0.0
        return float(pd.Series(returns).kurtosis())

    # =====================================================================
    # Price Structure
    # =====================================================================

    def trend() -> float:
        """Trend direction: slope of linear fit to close prices."""
        return float(np.polyfit(np.arange(n), close, 1)[0])

    def max_drawdown() -> float:
        """Maximum drawdown from running peak."""
        running_max = np.maximum.accumulate(close)
        drawdowns = close / running_max - 1.0
        return float(np.min(drawdowns))

    def price_range() -> float:
        """Normalized price range: (max - min) / mean."""
        mean_price = np.mean(close)
        return float((np.max(close) - np.min(close)) / max(mean_price, 1e-5))

    def close_to_close_range() -> float:
        """Close-to-close range: max(close) - min(close)."""
        return float(np.max(close) - np.min(close))

    def jarque_bera() -> float:
        """Jarque-Bera test statistic for normality of returns."""
        if len(returns) < 4:
            return 0.0
        s = float(pd.Series(returns).skew())
        k = float(pd.Series(returns).kurtosis())
        jb = (len(returns) / 6.0) * (s ** 2 + (k ** 2) / 4.0)
        return float(jb)

    # =====================================================================
    # Volume
    # =====================================================================

    def volume_trend() -> float:
        """Volume trend: mean of volume changes."""
        vol_changes = np.diff(volume) / (volume[:-1] + 1e-5)
        return float(np.mean(vol_changes))

    def volume_volatility() -> float:
        """Volume volatility: standard deviation of volume changes."""
        vol_changes = np.diff(volume) / (volume[:-1] + 1e-5)
        return float(np.std(vol_changes))

    def volume_price_corr() -> float:
        """Correlation between returns and volume changes."""
        if len(returns) < 3:
            return 0.0
        vol_changes = np.diff(volume)[: len(returns)] / (
            volume[:-1][: len(returns)] + 1e-5
        )
        return _safe_corr(returns, vol_changes)

    # =====================================================================
    # Build output
    # =====================================================================

    if return_all:
        return np.array(
            [
                # Momentum
                momentum_5(),
                momentum_10(),
                momentum_20(),
                momentum_64(),
                ma_crossover(),
                rsi_like(),
                # Volatility
                volatility(),
                parkinson_vol(),
                garman_klass_vol(),
                vol_of_vol(),
                vol_persistence(),
                vol_clustering(),
                # Autocorrelation
                autocorr_lag1(),
                autocorr_lag5(),
                hurst_estimate(),
                # Tail Risk
                var_95(),
                cvar_95(),
                max_1day_gain(),
                max_1day_loss(),
                skewness(),
                kurtosis(),
                # Price Structure
                trend(),
                max_drawdown(),
                price_range(),
                close_to_close_range(),
                jarque_bera(),
                # Volume
                volume_trend(),
                volume_volatility(),
                volume_price_corr(),
            ],
            dtype=np.float32,
        )

    # Default: 16 primary statistics used in concept labeling
    return np.array(
        [
            momentum_5(),
            trend(),
            volatility(),
            vol_persistence(),
            autocorr_lag1(),
            autocorr_lag5(),
            max_drawdown(),
            var_95(),
            max_1day_gain(),
            max_1day_loss(),
            skewness(),
            kurtosis(),
            price_range(),
            vol_clustering(),
            volume_trend(),
            volume_price_corr(),
        ],
        dtype=np.float32,
    )


def compute_labels_for_windows(
    windows: np.ndarray,
    return_all: bool = False,
) -> np.ndarray:
    """Compute financial statistics for every window in an array.

    Args:
        windows: Array of shape ``(n_windows, window_length, 6)``.
        return_all: If True, compute all 30+ statistics per window.

    Returns:
        Array of shape ``(n_windows, n_statistics)``.
    """
    labels = [compute_financial_statistics(w, return_all=return_all) for w in windows]
    return np.array(labels, dtype=np.float32)
