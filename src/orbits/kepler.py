from __future__ import annotations
import math
import torch
from torch import Tensor


@torch.compile
def solve_kepler_equation(M: Tensor, e: Tensor,
                          n_iter: int = 50, tol: float = 1e-10) -> Tensor:
    """Solves M = E - e sin(E) for the eccentric anomaly E, by Newton iteration.
    M is the mean anomaly (any shape), in radians. e is the eccentricity,
    broadcastable with M, clamped below 1 so fp = 1 - e cos(E) doesn't collapse
    to zero when E sits near zero. n_iter caps the Newton steps, but we bail out
    early once the residual is small. tol is the convergence threshold on
    |E - e sin(E) - M|.
    """
    # hard cap on e: at e == 1 the Newton update divides by zero whenever E is near zero,
    # and that NaN would otherwise propagate straight into the flow VI gradients
    e = e.clamp(max=1.0 - 1e-6)
    M = torch.remainder(M + math.pi, 2 * math.pi) - math.pi  # wrap into [-pi, pi] so iterates don't drift across cycles
    E = M.clone()
    for _ in range(n_iter):
        f = E - e * torch.sin(E) - M
        fp = 1.0 - e * torch.cos(E)
        E = E - f / fp
        if torch.max(torch.abs(f)) < tol:
            break
    return E


@torch.compile
def true_anomaly(M: Tensor, e: Tensor) -> Tensor:
    """Converts mean anomaly M to true anomaly nu, given eccentricity e.
    Uses the half-angle identity in atan2 form so the quadrant always comes out
    right (plain arctan would alias across the discontinuities). e gets clamped
    away from both bounds so sqrt(1 +/- e) stays finite under autograd.
    """
    e = e.clamp(min=1e-6, max=1.0 - 1e-6)
    E = solve_kepler_equation(M, e)
    sin_half = torch.sqrt(1.0 + e) * torch.sin(E / 2.0)
    cos_half = torch.sqrt(1.0 - e) * torch.cos(E / 2.0)
    return 2.0 * torch.atan2(sin_half, cos_half)


@torch.compile
def rv_one_planet(t: Tensor, P: Tensor, tp: Tensor, e: Tensor,
                  omega: Tensor, K: Tensor) -> Tensor:
    """RV contribution from one planet at observation times t.
    P, tp, e, omega, K are orbital period, time of periastron, eccentricity,
    argument of periastron, and semi-amplitude, in that order.
    """
    M = 2.0 * math.pi * (t - tp) / P
    nu = true_anomaly(M, e)
    return K * (torch.cos(omega + nu) + e * torch.cos(omega))


@torch.compile
def rv_model_two_planet(t: Tensor, theta: Tensor) -> Tensor:
    """Total RV from a two-planet system plus the systemic offset v0.
    theta layout: [P1, tp1, e1, omega1, K1, P2, tp2, e2, omega2, K2, v0].
    Days for periods and tp, radians for omega, m/s for K and v0.
    """
    P1, tp1, e1, w1, K1 = theta[0], theta[1], theta[2], theta[3], theta[4]
    P2, tp2, e2, w2, K2 = theta[5], theta[6], theta[7], theta[8], theta[9]
    v0 = theta[10]
    rv1 = rv_one_planet(t, P1, tp1, e1, w1, K1)
    rv2 = rv_one_planet(t, P2, tp2, e2, w2, K2)
    return v0 + rv1 + rv2


def log_likelihood_two_planet(theta: Tensor, t: Tensor,
                              rv_obs: Tensor, rv_err: Tensor,
                              log_jitter: Tensor) -> Tensor:
    """Gaussian log-likelihood, with a scalar jitter added in quadrature.
    We work in log-jitter instead of jitter so it stays positive without needing
    a constrained optimizer. The jitter itself is there to soak up stellar
    activity and other systematics that the reported error bars don't capture.
    """
    rv_pred = rv_model_two_planet(t, theta)
    sigma2 = rv_err ** 2 + torch.exp(2.0 * log_jitter)
    resid = rv_obs - rv_pred
    return -0.5 * torch.sum(resid ** 2 / sigma2 + torch.log(2.0 * math.pi * sigma2))
