# Astronomical priors on orbital parameters

from __future__ import annotations
import math
import torch
from torch import Tensor


# Hyperparameters for the priors. These are project-level knobs and could
# be exposed as arguments later; they are kept here as constants for
# clarity. All values are in their stated units (days, m/s, etc.).

# Period prior: log-uniform on [P_MIN, P_MAX] days. The bounds bracket the
# K2-24 system (transit-derived periods ~21 and ~42 days) by ~2 orders of
# magnitude, leaving room for the multimodal alias structure to manifest.
P_MIN, P_MAX = 1.0, 1000.0

# Time-of-periastron prior: uniform on [TP_MIN, TP_MAX]. Set wider than the
# data window so the prior never truncates the posterior.
TP_MIN, TP_MAX = 2300.0, 2500.0

# Eccentricity prior: Beta(KIPPING_A, KIPPING_B), the empirical fit of
# Kipping (2013) to the long-period exoplanet population.
KIPPING_A, KIPPING_B = 0.867, 3.03

# Semi-amplitude prior: half-normal with scale K_SCALE. Weakly informative
# on positive RVs; the K2-24 RV swing is ~10 m/s, so scale 50 is generous.
K_SCALE = 50.0

# Systemic velocity prior: zero-mean Gaussian. The data is mean-subtracted
# in practice so v0 should be small, but we keep a wide prior.
V0_SCALE = 50.0

# Log-jitter prior: uniform on [LOG_JIT_MIN, LOG_JIT_MAX]. This corresponds
# to jitter values in roughly [exp(-5), exp(3)] = [0.007, 20] m/s.
LOG_JIT_MIN, LOG_JIT_MAX = -5.0, 3.0


def _log_uniform_period(P: Tensor) -> Tensor:
    """
    Log of the log-uniform period prior on [P_MIN, P_MAX].
    """
    in_range = (P >= P_MIN) & (P <= P_MAX)
    log_norm = math.log(math.log(P_MAX / P_MIN))
    val = -torch.log(P) - log_norm
    return torch.where(in_range, val, torch.full_like(P, float("-inf")))


def _log_uniform(x: Tensor, low: float, high: float) -> Tensor:
    """Log of a uniform density on [low, high]."""
    in_range = (x >= low) & (x <= high)
    log_norm = math.log(high - low)
    return torch.where(in_range, torch.full_like(x, -log_norm), torch.full_like(x, float("-inf")))


def _log_beta(x: Tensor, a: float, b: float) -> Tensor:
    """Log of the Beta(a, b) density on [0, 1], evaluated at x.
    Uses torch.lgamma for the normalizing constant. Values outside [0, 1] return -inf."""

    in_range = (x > 0) & (x < 1)
    log_norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    val = (a - 1.0) * torch.log(x) + (b - 1.0) * torch.log1p(-x) - log_norm
    return torch.where(in_range, val, torch.full_like(x, float("-inf")))


def _log_half_normal(x: Tensor, scale: float) -> Tensor:
    """Log of a half-normal density on x >= 0 with scale parameter `scale`."""
    in_range = x >= 0
    log_norm = 0.5 * math.log(2.0 / math.pi) - math.log(scale)
    val = log_norm - 0.5 * (x / scale) ** 2
    return torch.where(in_range, val, torch.full_like(x, float("-inf")))


def _log_normal(x: Tensor, mean: float, scale: float) -> Tensor:
    """Log of a Normal(mean, scale^2) density."""
    return -0.5 * ((x - mean) / scale) ** 2 - math.log(scale) - 0.5 * math.log(2.0 * math.pi)


def log_prior_two_planet(theta: Tensor, log_jitter: Tensor) -> Tensor:
    """
    Total log prior for the two-planet Keplerian model.

    Sums independent log-priors over each parameter:
        - P1, P2: log-uniform on [P_MIN, P_MAX]
        - tp1, tp2: uniform on [TP_MIN, TP_MAX]
        - e1, e2: Beta(KIPPING_A, KIPPING_B)
        - omega1, omega2: uniform on [0, 2*pi]
        - K1, K2: half-normal with scale K_SCALE
        - v0: Normal(0, V0_SCALE^2)
        - log_jitter: uniform on [LOG_JIT_MIN, LOG_JIT_MAX]

    Parameters:
    theta : Tensor of shape (11,)
        [P1, tp1, e1, omega1, K1, P2, tp2, e2, omega2, K2, v0].
    log_jitter : Tensor (scalar)
        Log of jitter standard deviation.

    and it returns Tensor (scalar), Sum of log-priors.
    """
    P1, tp1, e1, w1, K1 = theta[0], theta[1], theta[2], theta[3], theta[4]
    P2, tp2, e2, w2, K2 = theta[5], theta[6], theta[7], theta[8], theta[9]
    v0 = theta[10]

    lp = (
        _log_uniform_period(P1)
        + _log_uniform(tp1, TP_MIN, TP_MAX)
        + _log_beta(e1, KIPPING_A, KIPPING_B)
        + _log_uniform(w1, 0.0, 2.0 * math.pi)
        + _log_half_normal(K1, K_SCALE)
        + _log_uniform_period(P2)
        + _log_uniform(tp2, TP_MIN, TP_MAX)
        + _log_beta(e2, KIPPING_A, KIPPING_B)
        + _log_uniform(w2, 0.0, 2.0 * math.pi)
        + _log_half_normal(K2, K_SCALE)
        + _log_normal(v0, 0.0, V0_SCALE)
        + _log_uniform(log_jitter, LOG_JIT_MIN, LOG_JIT_MAX)
    )
    return lp


def log_posterior_two_planet(
    theta: Tensor,
    log_jitter: Tensor,
    t: Tensor,
    rv_obs: Tensor,
    rv_err: Tensor,
) -> Tensor:
    """
    Unnormalized log posterior for the two-planet Keplerian model.
    This combines the Gaussian log likelihood with the joint log prior. This
    is the function MCMC and VI both target.
    """
    from .kepler import log_likelihood_two_planet
    return (
        log_likelihood_two_planet(theta, t, rv_obs, rv_err, log_jitter)
        + log_prior_two_planet(theta, log_jitter)
    )