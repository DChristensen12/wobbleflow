
# Keplerian radial velocity model for exoplanet inference.

"""
This script implements the standard one- or multi-planet RV model used throughout the
exoplanet literature. Given orbital parameters, and computes the predicted
radial velocity of the host star at each observation time.
"""

from __future__ import annotations
import torch
from torch import Tensor


def solve_kepler_equation(
    M: Tensor,
    e: Tensor,
    n_iter: int = 50,
    tol: float = 1e-10,
) -> Tensor:
    """
    Solve Kepler's equation M = E - e * sin(E) for the eccentric anomaly E.

    Uses Newton's method with the standard starting guess E0 = M. This
    converges quadratically for e < 1 and is robust for the moderate
    eccentricities (e <~ 0.9) we expect in exoplanet RV fits.

    Parameters:
    M : Tensor
        Mean anomaly (radians). Any shape.
    e : Tensor
        Eccentricity, in [0, 1). Broadcastable to the shape of M.
    n_iter : int
        Maximum number of Newton iterations.
    tol : float
        Convergence tolerance on |f(E)| where f(E) = E - e sin E - M.

    Returns:
    Tensor
        Eccentric anomaly E with the same shape as M.
    """
    # Wrapping M into [-pi, pi] so Newton's method starts close to the root.
    M = torch.remainder(M + torch.pi, 2 * torch.pi) - torch.pi
    E = M.clone()
    for _ in range(n_iter):
        f = E - e * torch.sin(E) - M
        fp = 1.0 - e * torch.cos(E)
        delta = f / fp
        E = E - delta
        if torch.max(torch.abs(f)) < tol:
            break
    return E


def true_anomaly(M: Tensor, e: Tensor) -> Tensor:
    """
    Convert mean anomaly M to true anomaly nu given eccentricity e.
    Uses the half-angle formula:
        tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    and is expressed in numerically stable form via atan2.

    Parameters:
    M : Tensor
        Mean anomaly (radians).
    e : Tensor
        Eccentricity, in [0, 1).

    Returns:
    Tensor
        True anomaly nu (radians), same shape as M.
    """
    E = solve_kepler_equation(M, e)
    # The atan2 form is better than a direct arctan because it gives the correct quadrant of nu for any E.
    sin_half = torch.sqrt(1.0 + e) * torch.sin(E / 2.0)
    cos_half = torch.sqrt(1.0 - e) * torch.cos(E / 2.0)
    nu = 2.0 * torch.atan2(sin_half, cos_half)
    return nu


def rv_one_planet(
    t: Tensor,
    P: Tensor,
    tp: Tensor,
    e: Tensor,
    omega: Tensor,
    K: Tensor,
) -> Tensor:
    """
    Predicted radial velocity contribution from a single planet.

    Implements
        v_r(t) = K * (cos(omega + nu(t)) + e * cos(omega))
    where the mean anomaly is M(t) = (2 pi / P) * (t - tp).

    Parameters:
    t : Tensor of shape (n_obs,)
        Observation times (days).
    P : Tensor (scalar or batched)
        Orbital period (days).
    tp : Tensor
        Time of periastron passage (days).
    e : Tensor
        Eccentricity, in [0, 1).
    omega : Tensor
        Argument of periastron (radians).
    K : Tensor
        Radial velocity semi-amplitude (m/s).

    Returns:
    Tensor of shape (n_obs,)
        Predicted RV contribution from this planet.
    """
    M = 2.0 * torch.pi * (t - tp) / P
    nu = true_anomaly(M, e)
    return K * (torch.cos(omega + nu) + e * torch.cos(omega))


def rv_model_two_planet(
    t: Tensor,
    theta: Tensor,
) -> Tensor:
    """
    Predicted radial velocity for a two-planet Keplerian model.

    The parameter vector theta is laid out as
        [P1, tp1, e1, omega1, K1, P2, tp2, e2, omega2, K2, v0]
    with units (days, days, unitless, radians, m/s, ..., m/s). The jitter
    parameter is handled separately in the likelihood, not here.

    Parameters:
    t : Tensor of shape (n_obs,)
        Observation times.
    theta : Tensor of shape (11,)
        Orbital parameters in the order above.

    Returns:
    Tensor of shape (n_obs,)
        Predicted RV at each observation time.
    """
    P1, tp1, e1, w1, K1 = theta[0], theta[1], theta[2], theta[3], theta[4]
    P2, tp2, e2, w2, K2 = theta[5], theta[6], theta[7], theta[8], theta[9]
    v0 = theta[10]
    rv1 = rv_one_planet(t, P1, tp1, e1, w1, K1)
    rv2 = rv_one_planet(t, P2, tp2, e2, w2, K2)
    return v0 + rv1 + rv2


def log_likelihood_two_planet(
    theta: Tensor,
    t: Tensor,
    rv_obs: Tensor,
    rv_err: Tensor,
    log_jitter: Tensor,
) -> Tensor:
    """
    Gaussian log likelihood for the two-planet Keplerian model.

    The observation noise model is
        rv_obs[i] ~ Normal(rv_pred(t[i]; theta), sqrt(rv_err[i]^2 + jitter^2))
    where jitter is a scalar added-in-quadrature white-noise term to absorb
    unmodeled noise sources (stellar activity, instrument systematics).

    The Parameters are:
    theta : Tensor of shape (11,)
        Orbital parameters; see rv_model_two_planet.
    t : Tensor of shape (n_obs,)
        Observation times.
    rv_obs : Tensor of shape (n_obs,)
        Observed radial velocities (m/s).
    rv_err : Tensor of shape (n_obs,)
        Reported observation errors (m/s).
    log_jitter : Tensor (scalar)
        log of the jitter standard deviation (m/s). Working in log-space
        keeps jitter strictly positive without constrained optimization.

    Returns Tensor (scalar), Log likelihood, summed over all observations.
    """
    rv_pred = rv_model_two_planet(t, theta)
    sigma2 = rv_err**2 + torch.exp(2.0 * log_jitter)
    resid = rv_obs - rv_pred
    # Gaussian log density, summed over observations.
    return -0.5 * torch.sum(resid**2 / sigma2 + torch.log(2.0 * torch.pi * sigma2))