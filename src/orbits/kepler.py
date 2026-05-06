
# Two-planet Keplerian radial velocity model.

# Import Libraries
from __future__ import annotations
import math
import torch
from torch import Tensor


def solve_kepler_equation(M: Tensor, e: Tensor,
                          n_iter: int = 50,
                          tol: float = 1e-10) -> Tensor:
    """Solve M = E - e sin(E) for E by Newton's method.
    The starting guess E0 = M is good enough for the moderate eccentricities
    encountered in RV fits (e < ~0.9). Wrapping M to [-pi, pi] keeps the
    Newton iterates from drifting across multiple orbital cycles.

    Parameters
    ----------
    M : Tensor
        Mean anomaly (radians). Any shape.
    e : Tensor
        Eccentricity, in [0, 1). Broadcastable against M.
    n_iter : int
        Maximum number of Newton iterations. We exit early if the residual
        falls below `tol`.
    tol : float
        Convergence tolerance on |E - e sin(E) - M|.

    Returns
    -------
    Tensor with the same shape as M, holding the eccentric anomaly E.
    """
    M = torch.remainder(M + math.pi, 2 * math.pi) - math.pi
    E = M.clone()
    for _ in range(n_iter):
        f = E - e * torch.sin(E) - M
        fp = 1.0 - e * torch.cos(E)
        E = E - f / fp
        if torch.max(torch.abs(f)) < tol:
            break
    return E


def true_anomaly(M: Tensor, e: Tensor) -> Tensor:
    """Convert mean anomaly M to true anomaly nu given eccentricity e.
    Uses the standard half-angle identity
        tan(nu / 2) = sqrt((1 + e) / (1 - e)) * tan(E / 2)
    rewritten in atan2 form, which gives the correct quadrant of nu without
    the periodic discontinuities of arctan."""
    E = solve_kepler_equation(M, e)
    sin_half = torch.sqrt(1.0 + e) * torch.sin(E / 2.0)
    cos_half = torch.sqrt(1.0 - e) * torch.cos(E / 2.0)
    return 2.0 * torch.atan2(sin_half, cos_half)


def rv_one_planet(t: Tensor,
                  P: Tensor, tp: Tensor, e: Tensor,
                  omega: Tensor, K: Tensor) -> Tensor:
    """Predicted RV contribution from a single planet at observation times t.
    Implements
        v_r(t) = K [cos(omega + nu(t)) + e cos(omega)]
    with M(t) = 2 pi (t - tp) / P. All inputs are broadcastable Tensors.
    """
    M = 2.0 * math.pi * (t - tp) / P
    nu = true_anomaly(M, e)
    return K * (torch.cos(omega + nu) + e * torch.cos(omega))


def rv_model_two_planet(t: Tensor, theta: Tensor) -> Tensor:
    """Predicted RV from a two-planet Keplerian model at observation times t.
    The parameter vector is laid out as
        theta = [P1, tp1, e1, omega1, K1, P2, tp2, e2, omega2, K2, v0]
    with units (days, days, unitless, radians, m/s, ..., m/s). 
    Note, the jitter parameter is handled separately in the likelihood, not here."""
    P1, tp1, e1, w1, K1 = theta[0], theta[1], theta[2], theta[3], theta[4]
    P2, tp2, e2, w2, K2 = theta[5], theta[6], theta[7], theta[8], theta[9]
    v0 = theta[10]
    rv1 = rv_one_planet(t, P1, tp1, e1, w1, K1)
    rv2 = rv_one_planet(t, P2, tp2, e2, w2, K2)
    return v0 + rv1 + rv2


def log_likelihood_two_planet(theta: Tensor,
                              t: Tensor,
                              rv_obs: Tensor,
                              rv_err: Tensor,
                              log_jitter: Tensor) -> Tensor:
    """Gaussian log-likelihood with an additive jitter term.
    The noise model is
        rv_obs[i] ~ Normal(rv_pred(t[i]; theta), rv_err[i]^2 + jitter^2),
    where jitter is a scalar added in quadrature to absorb stellar activity
    and instrumental systematics not captured by the reported error bars.
    Working in log-space for the jitter (log_jitter) keeps it strictly
    positive without needing a constrained optimizer."""
    rv_pred = rv_model_two_planet(t, theta)
    sigma2 = rv_err ** 2 + torch.exp(2.0 * log_jitter)
    resid = rv_obs - rv_pred
    return -0.5 * torch.sum(resid ** 2 / sigma2 + torch.log(2.0 * math.pi * sigma2))

