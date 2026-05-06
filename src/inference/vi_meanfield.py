# Mean-field Gaussian variational inference.

from __future__ import annotations
import math
from typing import Callable, List, Tuple
import torch
from torch import Tensor


def gaussian_entropy(log_s: Tensor) -> Tensor:
    """Entropy of N(mu, diag(s^2)). Independent of mu."""
    return torch.sum(log_s) + 0.5 * log_s.numel() * (1.0 + math.log(2.0 * math.pi))


def elbo(mu: Tensor,
         log_s: Tensor,
         log_post_fn: Callable[[Tensor], Tensor],
         n_mc: int = 8) -> Tensor:
    """Monte Carlo estimate of the ELBO using the reparameterization trick.
    Drawing n_mc samples per gradient step keeps gradient noise low; 8 is
    plenty for a 12-dimensional target.
    """
    s = torch.exp(log_s)
    eps = torch.randn(n_mc, mu.numel())
    eta_samples = mu.unsqueeze(0) + s.unsqueeze(0) * eps
    log_pi = torch.stack([log_post_fn(eta_samples[i]) for i in range(n_mc)])
    return log_pi.mean() + gaussian_entropy(log_s)


def fit(eta_init: Tensor,
        log_post_fn: Callable[[Tensor], Tensor],
        n_iter: int = 1500,
        lr: float = 5e-3,
        n_mc: int = 8,
        verbose: bool = False) -> Tuple[Tensor, Tensor, List[float]]:
    """Optimize the ELBO with Adam.
    Returns the fitted (mu, log_s) and the ELBO history (one value per
    iteration). Initial log_s = -2 corresponds to s ~ 0.14, which is a
    reasonable default for our reparameterized eta coordinates.
    """
    mu = eta_init.detach().clone().requires_grad_(True)
    log_s = torch.full_like(eta_init, -2.0).requires_grad_(True)

    opt = torch.optim.Adam([mu, log_s], lr=lr)
    history: List[float] = []
    for it in range(n_iter):
        opt.zero_grad()
        loss = -elbo(mu, log_s, log_post_fn, n_mc=n_mc)
        loss.backward()
        opt.step()
        history.append(-float(loss))
        if verbose and (it + 1) % 200 == 0:
            print(f"iter {it+1:5d}  ELBO = {-float(loss):.2f}")

    return mu.detach(), log_s.detach(), history


def sample(mu: Tensor, log_s: Tensor, n: int) -> Tensor:
    """Draw n samples from the fitted variational distribution."""
    s = torch.exp(log_s)
    eps = torch.randn(n, mu.numel())
    return mu.unsqueeze(0) + s.unsqueeze(0) * eps