""" Note: This has smooth priors with no hard boundaries. Variational inference samples from
    an unbounded Gaussian family, so any uniform prior with a sharp cutoff would
    hand back -inf gradients the moment the variational distribution crossed
    the boundary; we use broad Normal/half-Normal/Beta priors throughout.
"""

from __future__ import annotations
import math
import torch
from torch import Tensor
from .kepler import log_likelihood_two_planet

# Period prior: Normal on log P. Mean log(50) puts the bulk of the prior mass
# between ~7 and ~370 days, generous on both sides of the K2-24 transit
# periods (21 and 42 days).
LOG_P_MEAN, LOG_P_SCALE = math.log(50.0), 2.0

# Time of periastron prior: Normal centered at the data midpoint.
TP_MEAN, TP_SCALE = 2400.0, 50.0

# Kipping (2013) empirical Beta prior on eccentricity.
KIPPING_A, KIPPING_B = 0.867, 3.03

# Semi-amplitude prior: half-Normal. K2-24 has an RV swing of about 10 m/s,
# so scale 50 is weakly informative on the positive K axis.
K_SCALE = 50.0

# Systemic velocity prior: zero-mean Gaussian.
V0_SCALE = 50.0

# Log-jitter prior: Normal centered at 0 (1 m/s a priori).
LOG_JIT_MEAN, LOG_JIT_SCALE = 0.0, 2.0


def _log_normal(x: Tensor, mean: float, scale: float) -> Tensor:
    """Log of a Normal(mean, scale^2) density evaluated at x."""
    return (-0.5 * ((x - mean) / scale) ** 2
            - math.log(scale) - 0.5 * math.log(2.0 * math.pi))


def _log_normal_log_period(P: Tensor) -> Tensor:
    """Log of the Normal-on-log-P prior, evaluated at P > 0."""
    log_P = torch.log(P.clamp(min=1e-30))
    return _log_normal(log_P, LOG_P_MEAN, LOG_P_SCALE)


def _log_beta(x: Tensor, a: float, b: float) -> Tensor:
    """Log Beta(a, b) density at x.
    This clamps x strictly inside (eps, 1-eps) so the backward pass never hits
    log(0). Earlier versions used torch.where(in_range, val, -inf), but
    that pattern leaks NaN gradients through the masked branch and kills
    flow VI as soon as the planar layers push eta to a region where
    sigmoid saturates to exactly 1.0 in float32.
    """
    eps = 1e-6
    x_safe = x.clamp(min=eps, max=1.0 - eps)
    log_norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return (a - 1.0) * torch.log(x_safe) + (b - 1.0) * torch.log1p(-x_safe) - log_norm


def _log_half_normal(x: Tensor, scale: float) -> Tensor:
    """Log half-Normal density on x >= 0 with the given scale."""
    in_range = x >= 0
    log_norm = 0.5 * math.log(2.0 / math.pi) - math.log(scale)
    val = log_norm - 0.5 * (x / scale) ** 2
    return torch.where(in_range, val, torch.full_like(x, float("-inf")))


def log_prior_two_planet(theta: Tensor, log_jitter: Tensor) -> Tensor:
    """This is sum of independent log-priors on the orbital parameters.
    theta : 11-vector with layout [P1, tp1, e1, w1, K1, P2, tp2, e2, w2, K2, v0].
    log_jitter : scalar log-jitter in m/s.

    Note: omega1 and omega2 carry no explicit prior: the RV likelihood
    is 2*pi periodic in omega, so an improper uniform prior on R is
    equivalent to a constant and drops out of the unnormalized posterior.
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


def log_posterior_two_planet(theta: Tensor, log_jitter: Tensor,
                             t: Tensor, rv_obs: Tensor, rv_err: Tensor) -> Tensor:
    """Unnormalized log posterior in the natural (constrained) parameter space."""
    return (
        log_likelihood_two_planet(theta, t, rv_obs, rv_err, log_jitter)
        + log_prior_two_planet(theta, log_jitter)
    )
