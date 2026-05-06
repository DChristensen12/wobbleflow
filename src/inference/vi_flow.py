# Variational inference with planar normalizing flows.

from __future__ import annotations
import math
from typing import Callable, List
import torch
import torch.nn as nn
from torch import Tensor
from src.flows.planar import PlanarLayer


class FlowVI(nn.Module):
    """Mean-field Gaussian base + K planar transformations."""

    def __init__(self, D: int, K: int, mu_init: Tensor | None = None):
        super().__init__()
        if mu_init is None:
            mu_init = torch.zeros(D)
        self.mu = nn.Parameter(mu_init.clone())
        # log_s = -2 -> s ~ 0.14, same default as mean-field VI for fair
        # initialization across methods.
        self.log_s = nn.Parameter(torch.full((D,), -2.0))
        self.layers = nn.ModuleList([PlanarLayer(D) for _ in range(K)])

    def sample_and_log_q(self, n: int):
        """Draw n samples from q_phi and return (samples, log q at samples).
        We compute log q on the fly to avoid an expensive separate evaluation.
        log q_K = log q_0 - sum_k log|det df_k/deta_{k-1}|.
        """
        eps = torch.randn(n, self.mu.numel())
        s = torch.exp(self.log_s)
        z = self.mu.unsqueeze(0) + s.unsqueeze(0) * eps

        # log q_0(eta_0) for a diagonal Gaussian, expressed via eps so we
        # don't have to invert the reparameterization map.
        log_q = (-0.5 * eps ** 2 - self.log_s - 0.5 * math.log(2.0 * math.pi)).sum(dim=1)

        for layer in self.layers:
            z, log_det = layer(z)
            log_q = log_q - log_det
        return z, log_q


def fit(model: FlowVI,
        log_post_fn: Callable[[Tensor], Tensor],
        n_iter: int = 1500,
        lr: float = 5e-3,
        n_mc: int = 8,
        verbose: bool = False) -> List[float]:
    """Train a FlowVI model by maximizing the flow-based ELBO.
    Planar layers can produce large gradients early in training (the u, w
    parameters interact through u_hat in a way that occasionally amplifies
    updates). Mild gradient clipping keeps things stable; the value 5.0 is
    standard for normalizing-flow VI in practice.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[float] = []
    for it in range(n_iter):
        opt.zero_grad()
        z, log_q = model.sample_and_log_q(n_mc)
        log_pi = torch.stack([log_post_fn(z[i]) for i in range(n_mc)])
        elbo = (log_pi - log_q).mean()
        loss = -elbo
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        history.append(float(elbo))
        if verbose and (it + 1) % 200 == 0:
            print(f"iter {it+1:5d}  ELBO = {float(elbo):.2f}")
    return history