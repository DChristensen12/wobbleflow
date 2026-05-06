# Effective sample size and mixing diagnostics; MCMC diagnostics: autocorrelation and effective sample size.


from __future__ import annotations
from typing import Optional
import numpy as np


def autocorr_1d(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Sample autocorrelation function of a 1d array up to `max_lag`.
    Computes rho_k = (1 / (n - k)) sum_{i} (x_i - mean)(x_{i+k} - mean) / var.
    The default max_lag is N/4, which is a common practical compromise.
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
    """Effective sample size of a 1d MCMC chain.
    Sums the autocorrelations up to the first negative lag, then applies
    the standard ESS formula. `max(tau, 1)` guards against pathological
    chains where the truncated sum somehow goes below 1.
    """
    rho = autocorr_1d(x, max_lag=min(len(x) // 4, 500))

    # Find the first negative lag; if every lag is nonneg, just sum all of them.
    neg_idx = np.argmax(rho < 0)
    if neg_idx == 0 and rho[0] >= 0:
        cut = len(rho)
    else:
        cut = neg_idx

    tau = 1.0 + 2.0 * np.sum(rho[1:cut])
    return len(x) / max(tau, 1.0)
