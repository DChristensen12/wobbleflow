# Adaptive Markov Chain Monte Carlo augmented with normalizing flows.

from __future__ import annotations
import math
from typing import Callable, Tuple
import torch
from torch import Tensor
from src.flows.realnvp import RealNVP


def log_post_and_grad(eta: Tensor,
                      log_post_fn: Callable[[Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
    """log pi(eta) and its gradient. MALA needs grad log pi directly, not -grad U."""
    eta = eta.detach().clone().requires_grad_(True)
    log_pi = log_post_fn(eta)
    grad = torch.autograd.grad(log_pi, eta)[0]
    return log_pi.detach(), grad.detach()


def mala_step(eta: Tensor,
              log_post_fn: Callable[[Tensor], Tensor],
              tau: float) -> Tuple[Tensor, bool]:
    """One Metropolis-adjusted Langevin step.
    Proposal: y = x + (tau/2) grad log pi(x) + sqrt(tau) * N(0,I).
    Acceptance: MH ratio that accounts for the asymmetric proposal density.
    """
    eta = eta.detach().clone()
    log_pi_curr, grad_curr = log_post_and_grad(eta, log_post_fn)
    mu_fwd = eta + 0.5 * tau * grad_curr
    eta_prop = mu_fwd + math.sqrt(tau) * torch.randn_like(eta)
    log_pi_prop, grad_prop = log_post_and_grad(eta_prop, log_post_fn)
    mu_back = eta_prop + 0.5 * tau * grad_prop
    log_q_fwd = -0.5 * torch.sum((eta_prop - mu_fwd) ** 2) / tau
    log_q_back = -0.5 * torch.sum((eta - mu_back) ** 2) / tau
    log_alpha = (log_pi_prop - log_pi_curr) + (log_q_back - log_q_fwd)
    if torch.log(torch.rand(())) < log_alpha:
        return eta_prop.detach(), True
    return eta.detach(), False


def flow_proposal_step(eta: Tensor,
                       log_post_fn: Callable[[Tensor], Tensor],
                       flow: RealNVP) -> Tuple[Tensor, bool]:
    """Independence MH step using the flow as the proposal distribution.
    Draws y from the flow's pushforward, evaluates the flow density at both
    the current state x and the proposal y, then accepts with the ratio
    pi(y) * rho_hat(x) / (pi(x) * rho_hat(y)). When the flow is well-tuned
    to the target, this proposal makes long, non-local jumps between modes
    that local MALA cannot reach.
    """
    with torch.no_grad():
        log_pi_curr = log_post_fn(eta)
        log_q_curr = flow.log_q(eta.unsqueeze(0))[0]
        z = torch.randn(1, flow.D)
        y, log_det = flow.forward(z)
        log_base = (-0.5 * z ** 2 - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
        log_q_prop = log_base - log_det
    log_pi_prop = log_post_fn(y[0])
    log_alpha = (log_pi_prop - log_pi_curr) + (log_q_curr - log_q_prop[0])
    if torch.log(torch.rand(())) < log_alpha:
        return y[0].detach(), True
    return eta.detach(), False


def run_flowmc(eta_inits: Tensor,
               log_post_fn: Callable[[Tensor], Tensor],
               flow: RealNVP,
               n_iter: int, mala_tau: float,
               n_local_per_global: int = 5,
               lr: float = 5e-3,
               flow_warmup: int = 20,
               buffer_size: int = 2000,
               verbose: bool = False):
    """Adaptive flow-MCMC. Each outer iteration:

    1. n_local_per_global MALA sweeps per chain (local within-mode exploration).
    2. After flow_warmup outer steps, one flow proposal per chain.
    3. One Adam step on the flow's negative-log-likelihood loss against the
       last buffer_size chain states. The buffer is a rolling FIFO that
       smooths the training signal compared to using only the current state.

    Returns the full per-step chain history, the flow loss history, and the
    cumulative MALA / flow acceptance rates.
    """
    n_chains = eta_inits.shape[0]
    chains = eta_inits.detach().clone()
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    history = []
    saved = []
    buffer = []
    n_loc_acc = n_loc_try = 0
    n_glob_acc = n_glob_try = 0

    for step in range(n_iter):
        # Local MALA sweeps
        for _ in range(n_local_per_global):
            for c in range(n_chains):
                chains[c], a = mala_step(chains[c], log_post_fn, mala_tau)
                n_loc_acc += int(a); n_loc_try += 1
        # Global flow proposals after the warmup
        if step >= flow_warmup:
            for c in range(n_chains):
                chains[c], a = flow_proposal_step(chains[c], log_post_fn, flow)
                n_glob_acc += int(a); n_glob_try += 1
        # Appends to the training buffer (rolling FIFO)
        buffer.append(chains.detach().clone())
        if len(buffer) * n_chains > buffer_size:
            buffer.pop(0)
        train_batch = torch.cat(buffer, dim=0)
        # One gradient step on the flow loss (negative log-likelihood of buffer)
        opt.zero_grad()
        loss = -flow.log_q(train_batch).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()

        history.append(float(loss.detach()))
        saved.append(chains.detach().clone())

        if verbose and (step + 1) % 50 == 0:
            la = n_loc_acc / max(1, n_loc_try)
            ga = n_glob_acc / max(1, n_glob_try)
            print(f"step {step+1:4d}  loss = {float(loss):.2f}  "
                  f"local acc = {la:.2f}  global acc = {ga:.2f}")

    return (torch.stack(saved), history,
            n_loc_acc / max(1, n_loc_try),
            n_glob_acc / max(1, n_glob_try))
