"""
Normalized reparameterization: eta ~ N(0,1) in every dimension, centered
on the K2-24 transit values (P1 = 20.89 d, P2 = 42.36 d). This makes the
posterior roughly isotropic in eta-space, which is what HMC and MALA assume
when they pick a single step size for all coordinates.
"""

from __future__ import annotations
import math
import torch
from torch import Tensor
from .priors import log_posterior_two_planet

# per-coordinate centers and scales; eta = 0 puts every parameter at its K2 transit-derived value
P1_LOG_MEAN,     P1_LOG_SCALE    = math.log(20.89), 0.3
P2_LOG_MEAN,     P2_LOG_SCALE    = math.log(42.36), 0.3
TP_NORM_CENTER,  TP_NORM_SCALE   = 2400.0, 50.0
E_LOGIT_MEAN,    E_LOGIT_SCALE   = math.log(0.05 / 0.95), 1.5
K1_LOG_MEAN,     K2_LOG_MEAN     = math.log(8.0), math.log(6.0)
V0_NORM_SCALE    = 20.0
LOG_JIT_NORM_SCALE = 2.0


def eta_to_theta(eta: Tensor):
    """Map normalized eta in R^12 to (theta in R^11, log_jitter scalar)."""
    P1  = torch.exp(eta[0] * P1_LOG_SCALE + P1_LOG_MEAN)
    tp1 = eta[1] * TP_NORM_SCALE + TP_NORM_CENTER
    e1  = torch.sigmoid(eta[2] * E_LOGIT_SCALE + E_LOGIT_MEAN)
    w1  = eta[3]
    K1  = torch.exp(eta[4] + K1_LOG_MEAN)

    P2  = torch.exp(eta[5] * P2_LOG_SCALE + P2_LOG_MEAN)
    tp2 = eta[6] * TP_NORM_SCALE + TP_NORM_CENTER
    e2  = torch.sigmoid(eta[7] * E_LOGIT_SCALE + E_LOGIT_MEAN)
    w2  = eta[8]
    K2  = torch.exp(eta[9] + K2_LOG_MEAN)

    v0         = eta[10] * V0_NORM_SCALE
    log_jitter = eta[11] * LOG_JIT_NORM_SCALE

    e1 = torch.clamp(e1, 0.0, 0.999)
    e2 = torch.clamp(e2, 0.0, 0.999)
    K1 = torch.clamp(K1, 0.01, 100.0)
    K2 = torch.clamp(K2, 0.01, 100.0)
    P1 = torch.clamp(P1, 1.0, 1000.0)
    P2 = torch.clamp(P2, 1.0, 1000.0)

    theta = torch.stack([P1, tp1, e1, w1, K1, P2, tp2, e2, w2, K2, v0])
    return theta, log_jitter


def log_jacobian(eta: Tensor) -> Tensor:
    """Log |det d(prior-variable) / d eta| for the change of variables.

    Coordinate by coordinate:
      - Period: prior is on log P, so eta maps to log P. d(log P)/d eta = P_LOG_SCALE,
        giving log term log(P_LOG_SCALE).
      - tp: affine in eta with slope TP_NORM_SCALE, log term log(TP_NORM_SCALE).
      - Eccentricity: e = sigmoid(eta * E_LOGIT_SCALE + ...), so
        d e / d eta = E_LOGIT_SCALE * e * (1 - e), giving log term
        log(E_LOGIT_SCALE) + log(e) + log(1-e).
      - omega: identity map, contributes 0.
      - K: K = exp(eta + K_LOG_MEAN), so d K / d eta = K, log term eta + K_LOG_MEAN.
      - v0: affine with slope V0_NORM_SCALE, log term log(V0_NORM_SCALE).
      - log-jitter: affine with slope LOG_JIT_NORM_SCALE, log term log(LOG_JIT_NORM_SCALE).
    """
    # periods: prior is on log P, so the Jacobian is just log(P_LOG_SCALE)
    jac = math.log(P1_LOG_SCALE) + math.log(P2_LOG_SCALE)

    # times of periastron: affine
    jac = jac + math.log(TP_NORM_SCALE) + math.log(TP_NORM_SCALE)

    # eccentricities: scaled logit, sigmoid derivative is e * (1-e)
    e1_logit = eta[2] * E_LOGIT_SCALE + E_LOGIT_MEAN
    e2_logit = eta[7] * E_LOGIT_SCALE + E_LOGIT_MEAN
    e1 = torch.sigmoid(e1_logit)
    e2 = torch.sigmoid(e2_logit)
    jac = jac + math.log(E_LOGIT_SCALE) + math.log(E_LOGIT_SCALE)
    jac = jac + torch.log(e1) + torch.log1p(-e1)
    jac = jac + torch.log(e2) + torch.log1p(-e2)

    # omega: identity, contributes nothing

    # semi-amplitudes: prior is on K, so the Jacobian is log(K) = eta + K_LOG_MEAN
    jac = jac + (eta[4] + K1_LOG_MEAN) + (eta[9] + K2_LOG_MEAN)

    # systemic velocity and log-jitter: both affine
    jac = jac + math.log(V0_NORM_SCALE) + math.log(LOG_JIT_NORM_SCALE)

    return jac


def log_posterior_unconstrained(eta: Tensor, t: Tensor,
                                rv_obs: Tensor, rv_err: Tensor) -> Tensor:
    """Log posterior in normalized eta-space, including the Jacobian correction."""
    theta, log_jitter = eta_to_theta(eta)
    return (log_posterior_two_planet(theta, log_jitter, t, rv_obs, rv_err)
            + log_jacobian(eta))


def eta_samples_to_theta(eta_samples: Tensor) -> Tensor:
    """Convert a batch of eta samples to theta + log_jitter."""
    out_theta, out_lj = [], []
    for i in range(len(eta_samples)):
        theta, lj = eta_to_theta(eta_samples[i])
        out_theta.append(theta)
        out_lj.append(lj)
    theta_stack = torch.stack(out_theta)
    lj_stack    = torch.stack(out_lj)
    return torch.cat([theta_stack, lj_stack.unsqueeze(1)], dim=1)


def initial_eta() -> Tensor:
    """Starting point: eta = 0 maps exactly to the K2 transit configuration."""
    return torch.zeros(12)


def sort_planets_by_period(theta: Tensor) -> Tensor:
    """Enforce P1 < P2 in every row to resolve the planet-labeling symmetry."""
    out  = theta.clone()
    swap = out[:, 0] > out[:, 5]
    if swap.any():
        p1_block = out[swap, 0:5].clone()
        p2_block = out[swap, 5:10].clone()
        out[swap, 0:5] = p2_block
        out[swap, 5:10] = p1_block
    return out
