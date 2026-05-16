# Planar normalizing flow used by flow-based variational inference.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PlanarLayer(nn.Module):
    """One planar transformation f(z) = z + u * tanh(w^T z + b).
    A single planar layer is cheap (linear in the dimension D) and gives
    log-determinants in closed form. The reparameterization in u_hat() forces
    w^T u_hat >= -1 so the layer is guaranteed invertible no matter what
    the optimizer does with u and w.
    """

    def __init__(self, D: int):
        super().__init__()
        """Small init: each layer starts essentially as the identity map, which
         gives the optimizer a sane place to start."""
        self.u = nn.Parameter(torch.randn(D) * 0.01)
        self.w = nn.Parameter(torch.randn(D) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def u_hat(self) -> Tensor:
        """Reparameterized u that satisfies w^T u_hat >= -1."""
        wu = torch.dot(self.w, self.u)
        m_wu = -1.0 + F.softplus(wu)  # softplus(x) = log(1 + exp x)
        return self.u + (m_wu - wu) * self.w / (torch.dot(self.w, self.w) + 1e-12)

    def forward(self, z: Tensor):
        """Applys the planar map and return (f(z), log|det df/dz|).a
        z has shape (n, D); the returned tensors are (n, D) and (n,).
        """
        u_hat = self.u_hat()
        a = z @ self.w + self.b                                              # (n,)
        f_z = z + u_hat.unsqueeze(0) * torch.tanh(a).unsqueeze(1)             # (n, D)
        psi = (1.0 - torch.tanh(a) ** 2).unsqueeze(1) * self.w.unsqueeze(0)   # (n, D)
        log_det = torch.log(torch.abs(1.0 + psi @ u_hat) + 1e-12)             # (n,)
        return f_z, log_det
