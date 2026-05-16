# Real-valued Non-Volume Preserving coupling-layer flow used by Normalizing Flow-Markov Chain Monte Carlo.

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor


class CouplingLayer(nn.Module):
    """One Real-NVP coupling layer with a fixed binary mask.
    The masked half of the input is passed through unchanged; the unmasked
    half gets an affine transformation whose scale and shift are computed
    from the masked half. The Jacobian is triangular so log|det| is just
    the sum of the (1 - mask)-weighted scale outputs.
    """

    def __init__(self, D: int, mask: Tensor, hidden: int = 64):
        super().__init__()
        self.register_buffer("mask", mask)
        """Two small MLPs. The final tanh on s_net bounds the scale output,
           which empirically helps stability early in training"""
        self.s_net = nn.Sequential(
            nn.Linear(D, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, D), nn.Tanh(),
        )
        self.t_net = nn.Sequential(
            nn.Linear(D, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, D),
        )

        """Zero out the final Linear of each net so each layer starts as the
           identity. s_net ends with Tanh (its last Linear is at -2); t_net
           ends with Linear at -1."""
        nn.init.zeros_(self.s_net[-2].weight)
        nn.init.zeros_(self.s_net[-2].bias)
        nn.init.zeros_(self.t_net[-1].weight)
        nn.init.zeros_(self.t_net[-1].bias)

    def forward(self, z: Tensor):
        """z -> y, return (y, log|det df/dz|)."""
        z_masked = z * self.mask
        s = self.s_net(z_masked) * (1 - self.mask)
        t = self.t_net(z_masked) * (1 - self.mask)
        y = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
        return y, s.sum(dim=-1)

    def inverse(self, y: Tensor):
        """y -> z, return (z, log|det dg/dy|) where g is the inverse map."""
        y_masked = y * self.mask
        s = self.s_net(y_masked) * (1 - self.mask)
        t = self.t_net(y_masked) * (1 - self.mask)
        z = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return z, -s.sum(dim=-1)


class RealNVP(nn.Module):
    """Stacks of coupling layers with alternating-stripe masks.
    Alternating the mask each layer ensures every coordinate gets transformed.
    The base distribution is a standard isotropic Gaussian.
    """

    def __init__(self, D: int, n_layers: int = 6, hidden: int = 64):
        super().__init__()
        self.D = D
        self.layers = nn.ModuleList()
        for k in range(n_layers):
            mask = torch.zeros(D)
            mask[k % 2 :: 2] = 1.0  # alternating stripes
            self.layers.append(CouplingLayer(D, mask, hidden=hidden))

    def forward(self, z: Tensor):
        """Pushes base samples z through the flow. Returns (y, sum log|det|)."""
        log_det = torch.zeros(z.shape[0])
        for layer in self.layers:
            z, ld = layer(z)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, y: Tensor):
        """Pulls y back to base samples z. Returns (z, sum log|det|)."""
        log_det = torch.zeros(y.shape[0])
        for layer in reversed(self.layers):
            y, ld = layer.inverse(y)
            log_det = log_det + ld
        return y, log_det

    def log_q(self, y: Tensor) -> Tensor:
        """Log of the pushforward density at y.
        Uses the change of variables: if z = T^{-1}(y) and rho_B = N(0, I),
        then log rho_hat(y) = log rho_B(z) + log|det dT^{-1}/dy|.
        """
        z, log_det_inv = self.inverse(y)
        log_base = (-0.5 * z ** 2 - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
        return log_base + log_det_inv

    def sample(self, n: int):
        """Draws n samples from the pushforward distribution.
        Returns (y, log_q(y)) so the caller doesn't need a separate log_q call.
        """
        z = torch.randn(n, self.D)
        y, log_det = self.forward(z)
        log_base = (-0.5 * z ** 2 - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
        return y, log_base - log_det
