from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PlanarLayer(nn.Module):
    """One planar transformation, f(z) = z + u * tanh(w^T z + b).
    Cheap (linear in D) and the log-determinant comes out in closed form.
    The reparameterization in u_hat() keeps w^T u_hat >= -1 always, so the
    layer stays invertible no matter where the optimizer pushes u and w.
    """

    def __init__(self, D: int):
        super().__init__()
        # tiny init so the layer starts out close to the identity, gives the optimizer a sane starting point
        self.u = nn.Parameter(torch.randn(D) * 0.01)
        self.w = nn.Parameter(torch.randn(D) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def u_hat(self) -> Tensor:
        """The reparameterized u, guaranteed to satisfy w^T u_hat >= -1."""
        wu = torch.dot(self.w, self.u)
        m_wu = -1.0 + F.softplus(wu)  # softplus(x) = log(1 + exp x)
        return self.u + (m_wu - wu) * self.w / (torch.dot(self.w, self.w) + 1e-12)

    def forward(self, z: Tensor):
        """Applies the planar map, returns (f(z), log|det df/dz|).
        z is (n, D) in, and you get back (n, D) and (n,).
        """
        u_hat = self.u_hat()
        a = z @ self.w + self.b                                              # (n,)
        f_z = z + u_hat.unsqueeze(0) * torch.tanh(a).unsqueeze(1)             # (n, D)
        psi = (1.0 - torch.tanh(a) ** 2).unsqueeze(1) * self.w.unsqueeze(0)   # (n, D)
        log_det = torch.log(torch.abs(1.0 + psi @ u_hat) + 1e-12)             # (n,)
        return f_z, log_det
