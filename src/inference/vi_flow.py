# Variational inference with planar normalizing flows.

from __future__ import annotations
import math
from typing import Callable, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from src.flows.planar import PlanarLayer


class FlowVI(nn.Module):
    """Mean-field Gaussian base distribution + K planar transformations.
    Sampling pushes a reparameterized Gaussian draw through the K layers in
    sequence; the log-density of each sample is tracked alongside by
    subtracting the per-layer log|det Jacobian|.
    """

    def __init__(self, D: int, K: int,
                 mu_init: Optional[Tensor] = None,
                 log_s_init: float = -3.0):
        super().__init__()
        if mu_init is None:
            mu_init = torch.zeros(D)
        self.mu = nn.Parameter(mu_init.clone())
        self.log_s = nn.Parameter(torch.full((D,), log_s_init))
        self.layers = nn.ModuleList([PlanarLayer(D) for _ in range(K)])

    def sample_and_log_q(self, n: int) -> Tuple[Tensor, Tensor]:
        """Draw n samples from the flow and return (samples, log q at samples)."""
        eps = torch.randn(n, self.mu.numel())
        s = torch.exp(self.log_s)
        z = self.mu.unsqueeze(0) + s.unsqueeze(0) * eps
        # log q_0 of the diagonal Gaussian base, written via eps so we
        # don't have to invert the reparam map.
        log_q = (-0.5 * eps ** 2 - self.log_s - 0.5 * math.log(2.0 * math.pi)).sum(dim=1)
        for layer in self.layers:
            z, log_det = layer(z)
            log_q = log_q - log_det
        return z, log_q


def fit_flow_vi(model: FlowVI,
                log_post_fn: Callable[[Tensor], Tensor],
                n_iter: int = 2000, lr: float = 1e-3,
                n_mc: int = 32, verbose: bool = False) -> list:
    """Trains a FlowVI model by maximizing the flow-based ELBO.
    The learning rate is lower than mean-field VI (1e-3 vs 1e-2): planar
    layers compose gradients across K layers, so the safe step size shrinks.
    Cosine LR decay and a gentle norm clip at 10 keep training stable
    without strangling the flow.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_iter, eta_min=lr * 1e-2,
    )

    history = []
    for it in range(n_iter):
        opt.zero_grad()
        z, log_q = model.sample_and_log_q(n_mc)

        if torch.isnan(z).any() or torch.isnan(log_q).any():
            print(f"  NaN in flow output at iter {it}; aborting")
            break

        log_pi = torch.stack([log_post_fn(z[i]) for i in range(n_mc)])
        if torch.isnan(log_pi).any():
            bad = int(torch.isnan(log_pi).nonzero()[0, 0])
            print(f"  NaN in log_pi at iter {it}, sample {bad}: z = {z[bad].tolist()}")
            break

        elbo = (log_pi - log_q).mean()
        loss = -elbo
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        scheduler.step()
        history.append(float(elbo.detach()))

        if verbose and (it + 1) % 200 == 0:
            print(f"iter {it+1:5d}  ELBO = {float(elbo):.2f}")

    return history
