# Adaptive MCMC augmented with normalizing flows 

from __future__ import annotations
import math
from typing import Callable, List, Tuple
import torch
from torch import Tensor
from src.flows.realnvp import RealNVP


def log_post_and_grad(eta: Tensor,
                      log_post_fn: Callable[[Tensor], Tensor]
                      ) -> Tuple[Tensor, Tensor]:
    """log pi(eta) and its gradient (not the negative). MALA needs the grad
    of log pi, not of U; this is the cleanest place to compute it once."""
    eta = eta.detach().clone().requires_grad_(True)
    log_pi = log_post_fn(eta)
    grad = torch.autograd.grad(log_pi, eta)[0]
    return log_pi.detach(), grad.detach()


def mala_step(eta: Tensor,
              log_post_fn: Callable[[Tensor], Tensor],
              tau: float) -> Tuple[Tensor, bool]:
    """One Metropolis-adjusted Langevin step (lecture 32, eq. 2).

    Proposal:    y = x + (tau / 2) * grad log pi(x) + sqrt(tau) * eps
    Acceptance:  Metropolis-Hastings using the asymmetric proposal density.
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
    """One global flow-based MH step (Gabrié et al. eq. 6, 7).
    Generates an independent proposal y ~ rho_hat from the flow's
    pushforward, evaluates rho_hat at both x and y (no autograd needed,
    hence the no_grad context), and applies the MH ratio.
    """
    with torch.no_grad():
        log_pi_curr = log_post_fn(eta)
        log_q_curr = flow.log_q(eta.unsqueeze(0))[0]

        # Draw proposal from the flow's pushforward.
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
               n_iter: int,
               mala_tau: float,
               n_local_per_global: int,
               lr: float,
               verbose: bool = False
               ) -> Tuple[Tensor, List[float], float, float]:
    """Run Algorithm 1 of Gabrié et al. (2022) for `n_iter` outer iterations.
    `eta_inits` has shape (n_chains, D) and seeds the n_chains parallel
    walkers. Each iteration: n_local MALA steps per chain, one flow proposal
    per chain, one Adam step on the flow loss using the current chain states
    as a training batch.

    Returns the full history of chain states (shape (n_iter, n_chains, D)),
    the flow training-loss history, and the cumulative MALA / flow
    acceptance rates.
    """
    n_chains = eta_inits.shape[0]
    chains = eta_inits.detach().clone()
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    history: List[float] = []
    saved: List[Tensor] = []
    n_local_acc = n_local_tries = 0
    n_global_acc = n_global_tries = 0

    for step in range(n_iter):
        # Local sweeps: a few MALA steps per chain to refine within-mode mixing.
        for _ in range(n_local_per_global):
            for c in range(n_chains):
                chains[c], acc = mala_step(chains[c], log_post_fn, mala_tau)
                n_local_acc += int(acc)
                n_local_tries += 1

        # Global flow step: one independent proposal per chain.
        for c in range(n_chains):
            chains[c], acc = flow_proposal_step(chains[c], log_post_fn, flow)
            n_global_acc += int(acc)
            n_global_tries += 1

        # Train the flow on the current chain states (forward KL, eq. 11).
        # Using the live chain as the training batch is what makes this
        # method "adaptive". Gradient clipping prevents the early-training
        # explosions that Real-NVP can produce.
        optimizer.zero_grad()
        loss = -flow.log_q(chains.detach()).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        optimizer.step()

        history.append(float(loss))
        saved.append(chains.detach().clone())

        if verbose and (step + 1) % 50 == 0:
            la = n_local_acc / max(1, n_local_tries)
            ga = n_global_acc / max(1, n_global_tries)
            print(f"step {step+1:4d}  loss = {float(loss):.2f}  "
                  f"local acc = {la:.2f}  global acc = {ga:.2f}")

    samples = torch.stack(saved)
    return (samples,
            history,
            n_local_acc / max(1, n_local_tries),
            n_global_acc / max(1, n_global_tries))
