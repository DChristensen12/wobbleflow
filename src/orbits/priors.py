#Astronomical priors and the unnormalized log posterior.

from __future__ import annotations
import math
import torch
from torch import Tensor


# Period prior: Normal on log P. Mean log(50) puts the bulk of the prior
# between ~7 and ~370 days, generous around the K2-24 transit-derived
# periods of 21 and 42 days.
LOG_P_MEAN, LOG_P_SCALE = math.log(50.0), 2.0

# Time-of-periastron prior: Normal centered at the data midpoint.
TP_MEAN, TP_SCALE = 2400.0, 50.0

# Eccentricity prior: Beta(KIPPING_A, KIPPING_B), Kipping (2013).
KIPPING_A, KIPPING_B = 0.867, 3.03

# Semi-amplitude prior: half-normal with scale K_SCALE.
K_SCALE = 50.0

# Systemic velocity prior: zero-mean Gaussian.
V0_SCALE = 50.0

# Log-jitter prior: Normal centered at 0 (jitter ~ 1 m/s a priori).
LOG_JIT_MEAN, LOG_JIT_SCALE = 0.0, 2.0


def _log_normal(x: Tensor, mean: float, scale: float) -> Tensor:
    return (-0.5 * ((x - mean) / scale) ** 2
            - math.log(scale) - 0.5 * math.log(2.0 * math.pi))


def _log_normal_log_period(P: Tensor) -> Tensor:
    """Normal prior on log P, applied to P > 0."""
    log_P = torch.log(P.clamp(min=1e-30))
    return _log_normal(log_P, LOG_P_MEAN, LOG_P_SCALE)


def _log_beta(x: Tensor, a: float, b: float) -> Tensor:
    in_range = (x > 0) & (x < 1)
    log_norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    val = ((a - 1.0) * torch.log(x.clamp(min=1e-30))
           + (b - 1.0) * torch.log1p(-x.clamp(max=1.0 - 1e-12))
           - log_norm)
    return torch.where(in_range, val, torch.full_like(x, float("-inf")))


def _log_half_normal(x: Tensor, scale: float) -> Tensor:
    in_range = x >= 0
    log_norm = 0.5 * math.log(2.0 / math.pi) - math.log(scale)
    val = log_norm - 0.5 * (x / scale) ** 2
    return torch.where(in_range, val, torch.full_like(x, float("-inf")))


def log_prior_two_planet(theta: Tensor, log_jitter: Tensor) -> Tensor:
    """Sum of independent log-priors over each orbital parameter.

    Note: omega (argument of periastron) carries no explicit prior. Since
    the likelihood is 2 pi periodic in omega, an improper uniform prior is
    equivalent to a constant and drops out of the posterior.
    """
    P1, tp1, e1, _w1, K1 = theta[0], theta[1], theta[2], theta[3], theta[4]
    P2, tp2, e2, _w2, K2 = theta[5], theta[6], theta[7], theta[8], theta[9]
    v0 = theta[10]

    return (
        _log_normal_log_period(P1)
        + _log_normal(tp1, TP_MEAN, TP_SCALE)
        + _log_beta(e1, KIPPING_A, KIPPING_B)
        + _log_half_normal(K1, K_SCALE)
        + _log_normal_log_period(P2)
        + _log_normal(tp2, TP_MEAN, TP_SCALE)
        + _log_beta(e2, KIPPING_A, KIPPING_B)
        + _log_half_normal(K2, K_SCALE)
        + _log_normal(v0, 0.0, V0_SCALE)
        + _log_normal(log_jitter, LOG_JIT_MEAN, LOG_JIT_SCALE)
    )


def log_posterior_two_planet(theta, log_jitter, t, rv_obs, rv_err):
    from .kepler import log_likelihood_two_planet
    return (
        log_likelihood_two_planet(theta, t, rv_obs, rv_err, log_jitter)
        + log_prior_two_planet(theta, log_jitter)
    )