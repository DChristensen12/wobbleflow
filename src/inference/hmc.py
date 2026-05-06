# Hamiltonian Monte Carlo with leapfrog integration.

from __future__ import annotations
from typing import Callable, Tuple
import torch
from torch import Tensor


def potential_and_grad(eta: Tensor,
                       log_post_fn: Callable[[Tensor], Tensor]
                       ) -> Tuple[Tensor, Tensor]:
    """Compute U(eta) = -log pi(eta) and grad U(eta).
    We use PyTorch autograd, which means `log_post_fn` must be a
    differentiable function of eta. Detaching at the end keeps the returned
    tensors out of the autograd graph.
    """
    eta = eta.detach().clone().requires_grad_(True)
    U = -log_post_fn(eta)
    grad = torch.autograd.grad(U, eta)[0]
    return U.detach(), grad.detach()


def hmc_step(eta_current: Tensor,
             log_post_fn: Callable[[Tensor], Tensor],
             epsilon: float,
             L: int) -> Tuple[Tensor, bool, float]:
    """One HMC transition: sample momentum, leapfrog, MH-correct.

    Parameters
    ----------
    eta_current : current state of the chain.
    log_post_fn : differentiable log-posterior in eta-space.
    epsilon : leapfrog step size.
    L : number of leapfrog steps per transition.

    Returns
    -------
    eta_new : the new state (either the proposal or the current state).
    accepted : whether the proposal was accepted.
    U_value : potential energy at the returned state, useful for diagnostics.
    """
    eta = eta_current.detach().clone()
    v = torch.randn_like(eta)

    U_curr, grad_curr = potential_and_grad(eta, log_post_fn)
    K_curr = 0.5 * torch.sum(v ** 2)

    # Leapfrog: half-step v, then alternating full-step eta and full-step v,
    # ending with a final half-step v. This is the symmetric form in lecture
    # 34, eq. 12; the symmetry is what makes the discretization reversible.
    eta_new = eta.clone()
    v_new = v - 0.5 * epsilon * grad_curr
    grad_new = grad_curr.clone()
    for j in range(L):
        eta_new = eta_new + epsilon * v_new
        U_new, grad_new = potential_and_grad(eta_new, log_post_fn)
        if j != L - 1:
            v_new = v_new - epsilon * grad_new
    v_new = v_new - 0.5 * epsilon * grad_new

    # Negate momentum for involutivity. The sign flip doesn't change the
    # acceptance probability (since v ~ N(0, I) is symmetric), but it makes
    # the proposal map an involution, which is the standard derivation
    # (Neal 2011; lecture 34 eq. 7).
    v_new = -v_new

    K_new = 0.5 * torch.sum(v_new ** 2)
    log_alpha = U_curr + K_curr - U_new - K_new

    if torch.log(torch.rand(())) < log_alpha:
        return eta_new.detach(), True, float(U_new)
    return eta_current.detach(), False, float(U_curr)


def run_hmc(eta_init: Tensor,
            log_post_fn: Callable[[Tensor], Tensor],
            n_samples: int,
            epsilon: float,
            L: int,
            burn_in: int = 0,
            verbose: bool = False) -> Tuple[Tensor, float]:
    """Run an HMC chain for `burn_in + n_samples` iterations.
    Returns the post-burn-in samples and the post-burn-in acceptance rate.
    Tuning epsilon and L is part of the project; for our 12-dimensional
    target, epsilon ~ 1e-3 and L ~ 30 give acceptance in the 70-90% range.
    """
    eta = eta_init.detach().clone()
    samples = torch.empty((n_samples, eta.numel()))
    n_accept = 0
    total = burn_in + n_samples

    for i in range(total):
        eta, accepted, U_val = hmc_step(eta, log_post_fn, epsilon, L)
        if i >= burn_in:
            samples[i - burn_in] = eta
            n_accept += int(accepted)
        if verbose and (i + 1) % 200 == 0:
            phase = "burn-in" if i < burn_in else "sampling"
            print(f"[{phase}] iter {i+1:5d}/{total}  U = {U_val:.2f}")

    return samples, n_accept / n_samples