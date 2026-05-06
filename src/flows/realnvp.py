#Real-NVP coupling-layer flow (Dinh, Sohl-Dickstein & Bengio 2017).

# Import Libraries
from __future__ import annotations
import math
import torch
import torch.nn as nn


class CouplingLayer(nn.Module):
    """One Real-NVP coupling layer with a fixed binary mask."""

    def __init__(self, D: int, mask: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.register_buffer("mask", mask)
        # Two MLPs sharing architecture but distinct weights.
        # The final tanh on s_net keeps |s| bounded, which improves stability
        # in early training (a trick used in flonaco and many follow-ups).
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

        # Init the last layer of each net to zero so the layer starts as the
        # identity map. This makes early training much more stable.
        for m in (self.s_net[-1], self.t_net[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor):
        """z -> y. Returns (y, log|det df/dz|)."""
        z_masked = z * self.mask
        s = self.s_net(z_masked) * (1 - self.mask)
        t = self.t_net(z_masked) * (1 - self.mask)
        y = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor):
        """y -> z. Returns (z, log|det dg/dy|), where g = f^{-1}."""
        y_masked = y * self.mask
        s = self.s_net(y_masked) * (1 - self.mask)
        t = self.t_net(y_masked) * (1 - self.mask)
        z = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        log_det = -s.sum(dim=-1)
        return z, log_det


class RealNVP(nn.Module):
    """Stack of Real-NVP coupling layers with alternating stripe masks."""

    def __init__(self, D: int, n_layers: int = 6, hidden: int = 64):
        super().__init__()
        self.D = D
        self.layers = nn.ModuleList()
        for k in range(n_layers):
            mask = torch.zeros(D)
            mask[k % 2 :: 2] = 1.0          # alternating stripes
            self.layers.append(CouplingLayer(D, mask, hidden=hidden))

    def forward(self, z: torch.Tensor):
        """Push base samples z through the flow. Returns (y, sum log|det|)."""
        log_det = torch.zeros(z.shape[0])
        for layer in self.layers:
            z, ld = layer(z)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, y: torch.Tensor):
        """Pull y back to base samples z. Returns (z, sum log|det|)."""
        log_det = torch.zeros(y.shape[0])
        for layer in reversed(self.layers):
            y, ld = layer.inverse(y)
            log_det = log_det + ld
        return y, log_det

    def log_q(self, y: torch.Tensor) -> torch.Tensor:
        """log of the pushforward density at y, i.e. log rho_hat(y).
        Uses the change of variables: if z = T^{-1}(y) and rho_B is the base
        density (standard normal here), then
            rho_hat(y) = rho_B(z) |det dT^{-1}/dy|.
        """
        z, log_det_inv = self.inverse(y)
        log_base = (-0.5 * z ** 2 - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
        return log_base + log_det_inv

    def sample(self, n: int):
        """Draw n samples from the pushforward distribution.
        Returns (y, log_q(y)) so callers don't need a separate log_q call.
        """
        z = torch.randn(n, self.D)
        y, log_det = self.forward(z)
        log_base = (-0.5 * z ** 2 - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
        log_q = log_base - log_det
        return y, log_q
