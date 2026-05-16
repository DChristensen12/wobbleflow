# Mean-field Gaussian variational inference.

from __future__ import annotations
import math
from typing import Callable, Tuple
import torch
from torch import Tensor


def gaussian_entropy(log_s: Tensor) -> Tensor:
    """Entropy of N(mu, diag(s^2)). Independent of mu."""
    return torch.sum(log_s) + 0.5 * log_s.numel() * (1.0 + math.log(2.0 * math.pi))


def elbo_meanfield(mu: Tensor, log_s: Tensor,
                   log_post_fn: Callable[[Tensor], Tensor],
                   n_mc: int = 32) -> Tensor:
    """Monte Carlo ELBO using the reparameterization trick.
    Draws n_mc samples eta = mu + s * eps with eps ~ N(0,I), averages
    log pi(eta), then adds the closed-form Gaussian entropy.
    """
    s = torch.exp(log_s)
    eps = torch.randn(n_mc, mu.numel())
    eta_samples = mu.unsqueeze(0) + s.unsqueeze(0) * eps
    log_pi = torch.stack([log_post_fn(eta_samples[i]) for i in range(n_mc)])
    return log_pi.mean() + gaussian_entropy(log_s)


def fit_meanfield(eta_init: Tensor,
                  log_post_fn: Callable[[Tensor], Tensor],
                  n_iter: int = 2000, lr: float = 1e-2,
                  n_mc: int = 64, log_s_init: float = -3.0,
                  verbose: bool = False) -> Tuple[Tensor, Tensor, list]:
    """Optimizes the ELBO with Adam plus cosine LR decay.
    A tight initial variance (log_s = -3, so s ~ 0.05) keeps the likelihood
    gradient dominant early and prevents the variance from blowing up before
    the mean has had a chance to settle. Cosine decay smooths out late
    training where MC noise would otherwise produce visible ELBO jitter.
    """
    mu = eta_init.detach().clone().requires_grad_(True)
    log_s = torch.full_like(eta_init, log_s_init).requires_grad_(True)

    opt = torch.optim.Adam([mu, log_s], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_iter, eta_min=lr * 1e-2,
    )

    history = []
    for it in range(n_iter):
        opt.zero_grad()
        elbo = elbo_meanfield(mu, log_s, log_post_fn, n_mc=n_mc)
        if torch.isnan(elbo):
            print(f"  NaN at iter {it}; aborting")
            break
        loss = -elbo
        loss.backward()
        # Clip rare large gradients that would otherwise destabilize training.
        torch.nn.utils.clip_grad_norm_([mu, log_s], 5.0)
        opt.step()
        scheduler.step()
        history.append(float(elbo.detach()))
        if verbose and (it + 1) % 200 == 0:
            print(f"iter {it+1:5d}  ELBO = {float(elbo.detach()):.2f}")

    return mu.detach(), log_s.detach(), history


def sample_meanfield(mu: Tensor, log_s: Tensor, n: int) -> Tensor:
    """Draws n samples from the fitted Gaussian variational distribution."""
    s = torch.exp(log_s)
    eps = torch.randn(n, mu.numel())
    return mu.unsqueeze(0) + s.unsqueeze(0) * eps
