# Astronomical priors and the unnormalized log posterior.

# Needed Libraries
from __future__ import annotations
import math
import torch
from torch import Tensor

# Period prior: log-uniform on [P_MIN, P_MAX] days. The bounds bracket the
# K2-24 system (transit-derived periods ~21 and ~42 days) by ~2 orders of
# magnitude on each side, which leaves room for the multimodal alias
# structure to manifest in the posterior.
P_MIN, P_MAX = 1.0, 1000.0

# Time-of-periastron prior: uniform on [TP_MIN, TP_MAX]. Set wider than the
# data window so the prior never truncates the posterior.
TP_MIN, TP_MAX = 2300.0, 2500.0

# Eccentricity prior: Beta(KIPPING_A, KIPPING_B). Mildly favors low e, which
# matches what we know empirically about long-period exoplanets.
KIPPING_A, KIPPING_B = 0.867, 3.03

# Semi-amplitude prior: half-normal with scale K_SCALE. K2-24's RV swing is
# about 10 m/s, so a scale of 50 is generous and weakly informative.
K_SCALE = 50.0

# Systemic velocity prior: zero-mean Gaussian. The data is mean-subtracted in
# practice so v0 should be near zero, but we keep a wide prior.
V0_SCALE = 50.0

# Log-jitter prior: uniform on [LOG_JIT_MIN, LOG_JIT_MAX]. This corresponds
# to jitter values in roughly [exp(-5), exp(3)] = [0.007, 20] m/s.
LOG_JIT_MIN, LOG_JIT_MAX = -5.0, 3.0


def _log_uniform_period(P: Tensor) -> Tensor:
    """Log of the log-uniform prior on period, restricted to [P_MIN, P_MAX].
    p(P) ~ 1/P on the support, so log p(P) = -log P - log log(P_MAX/P_MIN)."""
    in_range = (P >= P_MIN) & (P <= P_MAX)
    log_norm = math.log(math.log(P_MAX / P_MIN))
    val = -torch.log(P) - log_norm
    return torch.where(in_range, val, torch.full_like(P, float("-inf")))


def _log_uniform(x: Tensor, low: float, high: float) -> Tensor:
    """Log of a uniform density on [low, high]; -inf outside."""
    in_range = (x >= low) & (x <= high)
    log_norm = math.log(high - low)
    return torch.where(in_range,
                       torch.full_like(x, -log_norm),
                       torch.full_like(x, float("-inf")))


def _log_beta(x: Tensor, a: float, b: float) -> Tensor:
    """Log of the Beta(a, b) density on (0, 1).
    The clamps below are defensive and only ever activate at the open
    boundary, where the density is anyway -inf or +inf. They keep autograd
    from producing NaNs from log(0)."""
    in_range = (x > 0) & (x < 1)
    log_norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    val = ((a - 1.0) * torch.log(x.clamp(min=1e-30))
           + (b - 1.0) * torch.log1p(-x.clamp(max=1.0 - 1e-12))
           - log_norm)
    return torch.where(in_range, val, torch.full_like(x, float("-inf")))


def _log_half_normal(x: Tensor, scale: float) -> Tensor:
    """Log of a half-normal density on x >= 0 with the given scale."""
    in_range = x >= 0
    log_norm = 0.5 * math.log(2.0 / math.pi) - math.log(scale)
    val = log_norm - 0.5 * (x / scale) ** 2
    return torch.where(in_range, val, torch.full_like(x, float("-inf")))


def _log_normal(x: Tensor, mean: float, scale: float) -> Tensor:
    """Log of a Normal(mean, scale^2) density."""
    return (-0.5 * ((x - mean) / scale) ** 2
            - math.log(scale)
            - 0.5 * math.log(2.0 * math.pi))


def log_prior_two_planet(theta: Tensor, log_jitter: Tensor) -> Tensor:
    """Sum of independent log-priors over each orbital parameter.

    Parameters
    ----------
    theta : Tensor of shape (11,)
        [P1, tp1, e1, omega1, K1, P2, tp2, e2, omega2, K2, v0].
    log_jitter : Tensor (scalar)
        Log of the jitter standard deviation in m/s.

    Returns
    -------
    Scalar Tensor: sum of log priors. -inf if any parameter is outside its
    support.
    """
    P1, tp1, e1, w1, K1 = theta[0], theta[1], theta[2], theta[3], theta[4]
    P2, tp2, e2, w2, K2 = theta[5], theta[6], theta[7], theta[8], theta[9]
    v0 = theta[10]

    return (
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


def log_posterior_two_planet(theta: Tensor,
                             log_jitter: Tensor,
                             t: Tensor,
                             rv_obs: Tensor,
                             rv_err: Tensor) -> Tensor:
    """Unnormalized log posterior in the natural (constrained) parameter space.
    log pi(theta, log_jitter | data) = log L + log prior, up to an additive
    constant. The reason I put the import of the likelihood inside this function is so that we
    don't create a circular import between this module and `kepler`.
    """
    from .kepler import log_likelihood_two_planet
    return (
        log_likelihood_two_planet(theta, t, rv_obs, rv_err, log_jitter)
        + log_prior_two_planet(theta, log_jitter)
    )

