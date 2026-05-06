# Planar normalizing flow (Rezende and Mohamed 2015)

# Libraries
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarLayer(nn.Module):
    """A single planar transformation f(z) = z + u h(w^T z + b)."""

    def __init__(self, D: int):
        super().__init__()
        # Small init: starts essentially as the identity map.
        self.u = nn.Parameter(torch.randn(D) * 0.01)
        self.w = nn.Parameter(torch.randn(D) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def u_hat(self) -> torch.Tensor:
        """Reparameterized u that enforces w^T u_hat >= -1."""
        wu = torch.dot(self.w, self.u)
        m_wu = -1.0 + F.softplus(wu)  # m(x) = -1 + log(1 + exp x)
        return self.u + (m_wu - wu) * self.w / (torch.dot(self.w, self.w) + 1e-12)

    def forward(self, z: torch.Tensor):
        """Apply the planar map and return (f(z), log|det df/dz|).
        z is shape (n, D); the returned tensors are (n, D) and (n,).
        """
        u_hat = self.u_hat()
        a = z @ self.w + self.b                          # (n,)
        f_z = z + u_hat.unsqueeze(0) * torch.tanh(a).unsqueeze(1)
        psi = (1.0 - torch.tanh(a) ** 2).unsqueeze(1) * self.w.unsqueeze(0)  # (n, D)
        log_det = torch.log(torch.abs(1.0 + psi @ u_hat) + 1e-12)            # (n,)
        return f_z, log_det

