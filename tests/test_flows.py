"""Tests for src/flows: PlanarLayer and RealNVP."""
import pytest
import torch

from src.flows.planar import PlanarLayer
from src.flows.realnvp import CouplingLayer, RealNVP


class TestPlanarLayer:
    def test_invertibility_constraint_at_init(self):
        """w^T u_hat >= -1 has to hold for every random init, not just some of them.
        Same check as cell 55 in the notebook, run before training starts, since
        it's what guarantees the layer is invertible in the first place.
        """
        torch.manual_seed(42)
        for _ in range(20):
            layer = PlanarLayer(12)
            wu = float(torch.dot(layer.w, layer.u_hat()))
            assert wu >= -1.0 - 1e-6, f"Constraint violated: w^T u_hat = {wu:.6f}"

    def test_invertibility_constraint_after_gradient_steps(self):
        """The constraint must hold even after the optimizer updates parameters."""
        torch.manual_seed(0)
        layer = PlanarLayer(12)
        opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
        z = torch.randn(8, 12)
        for _ in range(10):
            opt.zero_grad()
            _, log_det = layer(z)
            (-log_det.mean()).backward()
            opt.step()
        wu = float(torch.dot(layer.w, layer.u_hat()))
        assert wu >= -1.0 - 1e-6

    def test_forward_output_shapes(self):
        layer = PlanarLayer(12)
        z = torch.randn(16, 12)
        f_z, log_det = layer(z)
        assert f_z.shape == (16, 12)
        assert log_det.shape == (16,)

    def test_forward_log_det_finite(self):
        torch.manual_seed(0)
        layer = PlanarLayer(12)
        _, log_det = layer(torch.randn(8, 12))
        assert log_det.isfinite().all()

    def test_forward_output_finite(self):
        torch.manual_seed(0)
        layer = PlanarLayer(12)
        f_z, _ = layer(torch.randn(8, 12))
        assert f_z.isfinite().all()

    def test_single_sample_shape(self):
        layer = PlanarLayer(6)
        z = torch.randn(1, 6)
        f_z, log_det = layer(z)
        assert f_z.shape == (1, 6)
        assert log_det.shape == (1,)


class TestRealNVP:
    def test_forward_inverse_reconstruction_error(self):
        """max |z - inv(fwd(z))| < 1e-6.
        Same invertibility check as cell 57 in the notebook, run right before
        the FlowMC training loop starts.
        """
        torch.manual_seed(0)
        flow = RealNVP(D=12, n_layers=6, hidden=64)
        z_test = torch.randn(5, 12)
        y_test, _ = flow(z_test)
        z_back, _ = flow.inverse(y_test)
        recon_err = float(torch.max(torch.abs(z_test - z_back)))
        assert recon_err < 1e-6, f"Reconstruction error {recon_err:.2e} exceeds 1e-6"

    def test_forward_output_shapes(self):
        flow = RealNVP(D=12, n_layers=6, hidden=64)
        z = torch.randn(10, 12)
        y, log_det = flow(z)
        assert y.shape == (10, 12)
        assert log_det.shape == (10,)

    def test_inverse_output_shapes(self):
        flow = RealNVP(D=12, n_layers=6, hidden=64)
        y = torch.randn(10, 12)
        z, log_det = flow.inverse(y)
        assert z.shape == (10, 12)
        assert log_det.shape == (10,)

    def test_log_det_forward_inverse_cancel(self):
        """Forward and inverse log-dets sum to zero (change-of-variables consistency)."""
        torch.manual_seed(0)
        flow = RealNVP(D=12, n_layers=6, hidden=64)
        z = torch.randn(5, 12)
        y, ld_fwd = flow(z)
        _, ld_inv = flow.inverse(y)
        assert torch.allclose(ld_fwd + ld_inv, torch.zeros(5), atol=1e-5)

    def test_log_q_shape_and_finite(self):
        flow = RealNVP(D=12, n_layers=4, hidden=32)
        y = torch.randn(8, 12)
        lq = flow.log_q(y)
        assert lq.shape == (8,)
        assert lq.isfinite().all()

    def test_sample_output_shapes(self):
        flow = RealNVP(D=12, n_layers=4, hidden=32)
        y, log_q = flow.sample(20)
        assert y.shape == (20, 12)
        assert log_q.shape == (20,)

    def test_sample_output_finite(self):
        torch.manual_seed(0)
        flow = RealNVP(D=12, n_layers=4, hidden=32)
        y, log_q = flow.sample(10)
        assert y.isfinite().all()
        assert log_q.isfinite().all()

    def test_different_n_layers(self):
        """Should work for any reasonable number of coupling layers."""
        for n_layers in [2, 4, 8]:
            flow = RealNVP(D=12, n_layers=n_layers, hidden=32)
            z = torch.randn(4, 12)
            y, _ = flow(z)
            assert y.shape == (4, 12)
