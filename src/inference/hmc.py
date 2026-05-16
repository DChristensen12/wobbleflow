# Hamiltonian Monte Carlo with leapfrog integration.

from __future__ import annotations
from typing import Callable, Optional, Tuple
import torch
from torch import Tensor


def potential_and_grad(eta: Tensor,
                       log_post_fn: Callable[[Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
    """Computes U(eta) = -log pi(eta) and its gradient via autograd.
    log_post_fn must be differentiable in eta. We detach at the end so the
    returned tensors don't carry the autograd graph forward.
    """
    eta = eta.detach().clone().requires_grad_(True)
    U = -log_post_fn(eta)
    grad = torch.autograd.grad(U, eta)[0]
    return U.detach(), grad.detach()


def hmc_step(eta_current: Tensor,
             log_post_fn: Callable[[Tensor], Tensor],
             epsilon: float, L: int) -> Tuple[Tensor, bool, float]:
    """One HMC transition: sample momentum, leapfrog L steps, Metropolis-correct.
    Returns (new_state, accepted, U_value). The momentum negation at the end
    of the integrator doesn't affect acceptance but makes the proposal map
    an involution, which is standard.
    """
    eta = eta_current.detach().clone()
    v = torch.randn_like(eta)
    U_curr, grad_curr = potential_and_grad(eta, log_post_fn)
    K_curr = 0.5 * torch.sum(v ** 2)

    # Leapfrog integration
    eta_new = eta.clone()
    v_new = v - 0.5 * epsilon * grad_curr

    grad_new = grad_curr.clone()
    for j in range(L):
        eta_new = eta_new + epsilon * v_new
        U_new, grad_new = potential_and_grad(eta_new, log_post_fn)
        if j != L - 1:
            v_new = v_new - epsilon * grad_new

    v_new = v_new - 0.5 * epsilon * grad_new
    v_new = -v_new  # Flip momentum for reversibility

    K_new = 0.5 * torch.sum(v_new ** 2)
    log_alpha = U_curr + K_curr - U_new - K_new

    if torch.log(torch.rand(())) < log_alpha:
        return eta_new.detach(), True, float(U_new)
    return eta_current.detach(), False, float(U_curr)


def run_hmc_chain(eta_init: Tensor,
                  log_post_fn: Callable[[Tensor], Tensor],
                  n_samples: int, epsilon: float, L: int,
                  n_burnin: int = 500,
                  chain_id: Optional[int] = None,
                  verbose: bool = False) -> Tuple[Tensor, float]:
    """Runs a single HMC chain and return (samples, accept_rate).
    Burn-in samples are not stored; they're just for the chain to forget
    its initial point.
    """
    eta = eta_init.detach().clone()

    # Burn-in phase
    for i in range(n_burnin):
        eta, accepted, U_val = hmc_step(eta, log_post_fn, epsilon, L)

    # Sampling phase
    samples = torch.empty((n_samples, eta.numel()))
    n_accept = 0

    for i in range(n_samples):
        eta, accepted, U_val = hmc_step(eta, log_post_fn, epsilon, L)
        samples[i] = eta
        n_accept += int(accepted)

    accept_rate = n_accept / n_samples

    if verbose:
        tag = f"Chain {chain_id}" if chain_id is not None else "Chain"
        print(f"{tag}: acceptance = {accept_rate:.3f}")

    return samples, accept_rate


def run_hmc_multichain(eta_init: Tensor,
                       log_post_fn: Callable[[Tensor], Tensor],
                       n_chains: int, n_samples: int,
                       epsilon: float, L: int, n_burnin: int = 500,
                       verbose: bool = True) -> Tuple[list, list]:
    """Runs n_chains independent HMC chains and return per-chain results.
    Each chain gets its own random seed and starts at the same eta_init.
    Caller is responsible for combining and label-sorting the chains.
    """
    all_samples = []
    all_accept_rates = []
    for chain_id in range(1, n_chains + 1):
        torch.manual_seed(chain_id)
        samples, acc = run_hmc_chain(
            eta_init, log_post_fn,
            n_samples=n_samples, epsilon=epsilon, L=L,
            n_burnin=n_burnin, chain_id=chain_id, verbose=verbose,
        )
        all_samples.append(samples)
        all_accept_rates.append(acc)
    return all_samples, all_accept_rates
