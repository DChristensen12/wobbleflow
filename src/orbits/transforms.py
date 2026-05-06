# Unconstrained reparameterization for HMC and VI.

# Libraries
from __future__ import annotations
import math
import torch
from torch import Tensor

# Functions
def eta_to_theta(eta: Tensor):
    """Map eta in R^12 to (theta in R^11, log_jitter scalar).

    Returns the constrained-space parameter vector and the scalar log-jitter,
    so downstream code (likelihoods, priors) can use them directly.
    """
    P1 = torch.exp(eta[0])
    tp1 = eta[1]
    e1 = torch.sigmoid(eta[2])
    w1 = eta[3]
    K1 = torch.exp(eta[4])
    P2 = torch.exp(eta[5])
    tp2 = eta[6]
    e2 = torch.sigmoid(eta[7])
    w2 = eta[8]
    K2 = torch.exp(eta[9])
    v0 = eta[10]
    log_jitter = eta[11]
    theta = torch.stack([P1, tp1, e1, w1, K1, P2, tp2, e2, w2, K2, v0])
    return theta, log_jitter


def log_jacobian(eta: Tensor) -> Tensor:
    """Sum of per-coordinate log-Jacobian contributions for eta -> theta."""
    # Two log-period contributions: log P = eta itself.
    j = eta[0] + eta[5]
    # Two log-K contributions, same pattern.
    j = j + eta[4] + eta[9]
    # Two logit-eccentricity contributions: log(e (1 - e)).
    e1 = torch.sigmoid(eta[2])
    e2 = torch.sigmoid(eta[7])
    j = j + torch.log(e1) + torch.log1p(-e1) + torch.log(e2) + torch.log1p(-e2)
    return j


def log_posterior_unconstrained(eta: Tensor,
                                t: Tensor,
                                rv_obs: Tensor,
                                rv_err: Tensor) -> Tensor:
    """Log posterior in eta-space, including the log-Jacobian correction."""
    from .priors import log_posterior_two_planet
    theta, log_jitter = eta_to_theta(eta)
    return (log_posterior_two_planet(theta, log_jitter, t, rv_obs, rv_err)
            + log_jacobian(eta))


def eta_samples_to_theta(eta_samples: Tensor) -> Tensor:
    """Vectorized eta -> theta mapping for batches of samples.
    Inputs a (n, 12) tensor of eta samples, returns a (n, 12) tensor whose
    last column is log_jitter (kept on the log scale, since that's what the
    likelihood uses)."""
    P1 = torch.exp(eta_samples[:, 0])
    tp1 = eta_samples[:, 1]
    e1 = torch.sigmoid(eta_samples[:, 2])
    w1 = eta_samples[:, 3]
    K1 = torch.exp(eta_samples[:, 4])
    P2 = torch.exp(eta_samples[:, 5])
    tp2 = eta_samples[:, 6]
    e2 = torch.sigmoid(eta_samples[:, 7])
    w2 = eta_samples[:, 8]
    K2 = torch.exp(eta_samples[:, 9])
    v0 = eta_samples[:, 10]
    lj = eta_samples[:, 11]
    return torch.stack([P1, tp1, e1, w1, K1, P2, tp2, e2, w2, K2, v0, lj], dim=1)


def initial_eta() -> Tensor:
    """A reasonable starting point for chains and VI initializations.
    Uses the K2-24 transit-derived periods (~20.89 and ~42.36 days) and weak
    guesses elsewhere. This puts the initial state in a region of high
    posterior density without injecting too much information.
    """
    return torch.tensor([
        math.log(20.89),         # log P1
        2380.0,                  # tp1, inside the data window
        math.log(0.05 / 0.95),   # logit e1, e ~ 0.05
        0.0,                     # omega1
        math.log(8.0),           # log K1, K ~ 8 m/s
        math.log(42.36),         # log P2
        2400.0,                  # tp2
        math.log(0.05 / 0.95),   # logit e2
        0.0,                     # omega2
        math.log(6.0),           # log K2
        0.0,                     # v0
        0.0,                     # log_jitter, jitter ~ 1 m/s
    ])

