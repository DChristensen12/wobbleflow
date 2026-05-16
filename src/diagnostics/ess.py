# Markov Chain Monte Carlo diagnostics: autocorrelation and effective sample size.

from __future__ import annotations
from typing import Optional
import numpy as np


def autocorr_1d(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Samples autocorrelation of a 1D chain at lags 0..max_lag-1.
    Default max_lag is N/4, which is a common practical compromise between
    seeing the autocorrelation decay and avoiding the noisy tail.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    var = np.dot(x, x) / n
    rho = np.empty(max_lag)
    for k in range(max_lag):
        rho[k] = np.dot(x[: n - k], x[k:]) / ((n - k) * var)
    return rho


def ess_1d(x: np.ndarray) -> float:
    """Effective sample size for a 1D MCMC chain.
    Sums the autocorrelations up to the first negative lag (initial monotone
    sequence truncation), then applies ESS = N / (1 + 2 sum rho_k).
    """
    rho = autocorr_1d(x, max_lag=min(len(x) // 4, 500))
    neg_idx = np.argmax(rho < 0)
    if neg_idx == 0 and rho[0] >= 0:
        cut = len(rho)
    else:
        cut = neg_idx
    tau = 1.0 + 2.0 * np.sum(rho[1:cut])
    return len(x) / max(tau, 1.0)


def posterior_summary(samples: np.ndarray) -> dict:
    """Return mean, std, median, and 95% HDI for a 1D sample array."""
    return {
        "mean":   float(np.mean(samples)),
        "std":    float(np.std(samples)),
        "median": float(np.median(samples)),
        "hdi_lo": float(np.percentile(samples, 2.5)),
        "hdi_hi": float(np.percentile(samples, 97.5)),
    }
