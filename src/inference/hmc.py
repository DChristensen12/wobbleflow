# Hamiltonian Monte Carlo Baseline for the two-planet Keplerian model.

from __future__ import annotations
from typing import Callable, Tuple
import torch
from torch import Tensor


def _split_params(z: Tensor) -> Tuple[Tensor, Tensor]:
    return z[:-1], z[-1]


def _potential_and_grad(
    z: Tensor,
    log_post_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tuple[Tensor, Tensor]:
    z = z.detach().clone().requires_grad_(True)
    theta, log_jitter = _split_params(z)
    log_post = log_post_fn(theta, log_jitter)
    U = -log_post
    grad = torch.autograd.grad(U, z)[0]
    return U.detach(), grad.detach()


def hmc_step(
    z_current: Tensor,
    log_post_fn: Callable[[Tensor, Tensor], Tensor],
    epsilon: float,
    L: int,
) -> Tuple[Tensor, bool, float]:
    z = z_current.detach().clone()
    v = torch.randn_like(z)

    U_curr, grad_curr = _potential_and_grad(z, log_post_fn)
    K_curr = 0.5 * torch.sum(v**2)

    # Leapfrog integration
    z_new = z.clone()
    v_new = v.clone()
    grad_new = grad_curr.clone()

    v_new = v_new - 0.5 * epsilon * grad_curr  # initial half-step on v
    for j in range(L):
        z_new = z_new + epsilon * v_new
        U_new, grad_new = _potential_and_grad(z_new, log_post_fn)
        if j != L - 1:
            v_new = v_new - epsilon * grad_new  # full v-steps in interior
    v_new = v_new - 0.5 * epsilon * grad_new  # final half-step on v
    v_new = -v_new

    K_new = 0.5 * torch.sum(v_new**2)

    log_alpha = U_curr + K_curr - U_new - K_new

    if torch.log(torch.rand(())) < log_alpha:
        return z_new.detach(), True, float(U_new)
    else:
        return z_current.detach(), False, float(U_curr)


def run_hmc(
    z_init: Tensor,
    log_post_fn: Callable[[Tensor, Tensor], Tensor],
    n_samples: int,
    epsilon: float,
    L: int,
    burn_in: int = 0,
    verbose: bool = False,
) -> Tuple[Tensor, float]:
    z = z_init.detach().clone()
    samples = torch.empty((n_samples, z.numel()))
    n_accept = 0

    total = burn_in + n_samples
    for i in range(total):
        z, accepted, U_val = hmc_step(z, log_post_fn, epsilon, L)
        if i >= burn_in:
            samples[i - burn_in] = z
            n_accept += int(accepted)
        if verbose and (i + 1) % 100 == 0:
            phase = "burn-in" if i < burn_in else "sampling"
            recent_acc = n_accept / max(1, i - burn_in + 1) if i >= burn_in else 0.0
            print(
                f"[{phase}] iter {i+1:5d}/{total}, "
                f"U = {U_val:.2f}, accept rate = {recent_acc:.3f}"
            )

    accept_rate = n_accept / n_samples
    return samples, accept_rate