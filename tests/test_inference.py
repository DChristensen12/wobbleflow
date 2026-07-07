"""Tests for src/inference: HMC, mean-field VI, flow VI, FlowMC."""
import math

import pytest
import torch

from src.flows.realnvp import RealNVP
from src.inference.flowmc import flow_proposal_step, log_post_and_grad, mala_step, run_flowmc
from src.inference.hmc import hmc_step, potential_and_grad, run_hmc_chain, run_hmc_multichain
from src.inference.vi_flow import FlowVI, fit_flow_vi
from src.inference.vi_meanfield import (
    elbo_meanfield,
    fit_meanfield,
    gaussian_entropy,
    sample_meanfield,
)


class TestHMC:
    def test_potential_and_grad_shapes(self, lp_eta_fn, eta0):
        U, grad = potential_and_grad(eta0, lp_eta_fn)
        assert U.shape == ()
        assert grad.shape == (12,)

    def test_potential_and_grad_finite(self, lp_eta_fn, eta0):
        U, grad = potential_and_grad(eta0, lp_eta_fn)
        assert U.isfinite()
        assert grad.isfinite().all()

    def test_potential_equals_negative_log_posterior(self, lp_eta_fn, eta0):
        U, _ = potential_and_grad(eta0, lp_eta_fn)
        lp = lp_eta_fn(eta0)
        assert torch.allclose(U, -lp, atol=1e-6)

    def test_hmc_step_output_shapes(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        eta_new, accepted, U_val = hmc_step(eta0, lp_eta_fn, epsilon=0.01, L=3)
        assert eta_new.shape == (12,)
        assert isinstance(accepted, bool)
        assert isinstance(U_val, float)

    def test_hmc_step_acceptance_is_bool(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        _, accepted, _ = hmc_step(eta0, lp_eta_fn, epsilon=0.01, L=3)
        assert accepted in (True, False)

    def test_hmc_step_tiny_epsilon_high_acceptance(self, lp_eta_fn, eta0):
        """With very small step size, virtually every proposal should be accepted."""
        n_trials = 20
        n_accepted = 0
        eta = eta0.clone()
        for seed in range(n_trials):
            torch.manual_seed(seed)
            eta, accepted, _ = hmc_step(eta, lp_eta_fn, epsilon=1e-5, L=1)
            n_accepted += int(accepted)
        assert n_accepted >= 15

    def test_run_hmc_chain_output_shapes(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        samples, acc = run_hmc_chain(
            eta0, lp_eta_fn, n_samples=5, epsilon=0.01, L=3, n_burnin=2
        )
        assert samples.shape == (5, 12)
        assert 0.0 <= acc <= 1.0

    def test_run_hmc_chain_samples_finite(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        samples, _ = run_hmc_chain(
            eta0, lp_eta_fn, n_samples=5, epsilon=0.01, L=3, n_burnin=2
        )
        assert samples.isfinite().all()

    def test_run_hmc_multichain_returns_correct_count(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        all_samples, all_acc = run_hmc_multichain(
            eta0, lp_eta_fn,
            n_chains=3, n_samples=4, epsilon=0.01, L=3, n_burnin=2, verbose=False,
        )
        assert len(all_samples) == 3
        assert len(all_acc) == 3
        for s in all_samples:
            assert s.shape == (4, 12)

    def test_run_hmc_multichain_acceptance_in_range(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        _, all_acc = run_hmc_multichain(
            eta0, lp_eta_fn,
            n_chains=2, n_samples=5, epsilon=0.01, L=3, n_burnin=2, verbose=False,
        )
        assert all(0.0 <= a <= 1.0 for a in all_acc)


class TestMeanFieldVI:
    def test_gaussian_entropy_scalar_formula(self):
        """For D=1: H = 0.5*(1 + log(2π)) + log_s."""
        log_s = torch.tensor([0.5], dtype=torch.float64)
        H = gaussian_entropy(log_s)
        expected = 0.5 * (1.0 + math.log(2.0 * math.pi)) + 0.5
        assert abs(float(H) - expected) < 1e-8

    def test_gaussian_entropy_increases_with_scale(self):
        log_s_small = torch.full((4,), -2.0, dtype=torch.float64)
        log_s_large = torch.full((4,),  2.0, dtype=torch.float64)
        assert gaussian_entropy(log_s_large) > gaussian_entropy(log_s_small)

    def test_elbo_meanfield_is_finite(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        log_s = torch.full((12,), -2.0, dtype=torch.float64)
        elbo = elbo_meanfield(eta0, log_s, lp_eta_fn, n_mc=8)
        assert elbo.isfinite()

    def test_fit_meanfield_output_shapes(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        mu, log_s, history = fit_meanfield(eta0, lp_eta_fn, n_iter=5, n_mc=4)
        assert mu.shape == (12,)
        assert log_s.shape == (12,)
        assert len(history) == 5

    def test_fit_meanfield_history_all_finite(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        _, _, history = fit_meanfield(eta0, lp_eta_fn, n_iter=10, n_mc=4)
        assert all(math.isfinite(v) for v in history)

    def test_fit_meanfield_elbo_trend_upward(self, lp_eta_fn, eta0):
        """ELBO should be higher at the end than at the start after enough iterations."""
        torch.manual_seed(0)
        _, _, history = fit_meanfield(eta0, lp_eta_fn, n_iter=100, n_mc=8)
        assert history[-1] > history[0]

    def test_sample_meanfield_shape(self):
        mu    = torch.zeros(12, dtype=torch.float64)
        log_s = torch.full((12,), -1.0, dtype=torch.float64)
        samples = sample_meanfield(mu, log_s, n=50)
        assert samples.shape == (50, 12)

    def test_sample_meanfield_mean_near_mu(self):
        """With tiny variance and many samples, the sample mean should be near mu."""
        torch.manual_seed(0)
        mu    = torch.ones(12, dtype=torch.float64) * 2.0
        log_s = torch.full((12,), -4.0, dtype=torch.float64)
        samples = sample_meanfield(mu, log_s, n=500)
        assert torch.allclose(samples.mean(0), mu, atol=0.05)


class TestFlowVI:
    def test_planar_layers_satisfy_invertibility_at_init(self):
        """All layers satisfy w^T u_hat >= -1 at initialization (from cell 55)."""
        torch.manual_seed(0)
        model = FlowVI(D=12, K=8)
        for k, layer in enumerate(model.layers):
            wu = float(torch.dot(layer.w, layer.u_hat()))
            assert wu >= -1.0 - 1e-6, f"Layer {k}: w^T u_hat = {wu:.6f}"

    def test_sample_and_log_q_shapes(self):
        model = FlowVI(D=12, K=4)
        z, log_q = model.sample_and_log_q(n=16)
        assert z.shape == (16, 12)
        assert log_q.shape == (16,)

    def test_sample_and_log_q_finite(self):
        torch.manual_seed(0)
        model = FlowVI(D=12, K=4)
        z, log_q = model.sample_and_log_q(n=8)
        assert z.isfinite().all()
        assert log_q.isfinite().all()

    def test_fit_flow_vi_returns_correct_length_history(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        model = FlowVI(D=12, K=2, mu_init=eta0)
        history = fit_flow_vi(model, lp_eta_fn, n_iter=5, n_mc=4)
        assert len(history) == 5

    def test_fit_flow_vi_history_all_finite(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        model = FlowVI(D=12, K=2, mu_init=eta0)
        history = fit_flow_vi(model, lp_eta_fn, n_iter=10, n_mc=4)
        assert all(math.isfinite(v) for v in history)

    def test_more_layers_does_not_break_shapes(self):
        for K in [1, 4, 16]:
            model = FlowVI(D=12, K=K)
            z, log_q = model.sample_and_log_q(n=4)
            assert z.shape == (4, 12)
            assert log_q.shape == (4,)


class TestFlowMC:
    def test_log_post_and_grad_shapes(self, lp_eta_fn, eta0):
        lp, grad = log_post_and_grad(eta0, lp_eta_fn)
        assert lp.shape == ()
        assert grad.shape == (12,)

    def test_log_post_and_grad_finite(self, lp_eta_fn, eta0):
        lp, grad = log_post_and_grad(eta0, lp_eta_fn)
        assert lp.isfinite()
        assert grad.isfinite().all()

    def test_log_post_and_grad_matches_log_posterior(self, lp_eta_fn, eta0):
        lp, _ = log_post_and_grad(eta0, lp_eta_fn)
        assert torch.allclose(lp, lp_eta_fn(eta0), atol=1e-6)

    def test_mala_step_output_shapes(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        eta_new, accepted = mala_step(eta0, lp_eta_fn, tau=1e-4)
        assert eta_new.shape == (12,)
        assert isinstance(accepted, bool)

    def test_mala_step_tiny_tau_high_acceptance(self, lp_eta_fn, eta0):
        """Very small step size, so almost every proposal should get accepted."""
        n_accepted = 0
        eta = eta0.clone()
        for seed in range(20):
            torch.manual_seed(seed)
            eta, acc = mala_step(eta, lp_eta_fn, tau=1e-8)
            n_accepted += int(acc)
        assert n_accepted >= 15

    def test_flow_proposal_step_output_shapes(self, lp_eta_fn, eta0):
        torch.manual_seed(0)
        flow = RealNVP(D=12, n_layers=4, hidden=32)
        eta_new, accepted = flow_proposal_step(eta0, lp_eta_fn, flow)
        assert eta_new.shape == (12,)
        assert isinstance(accepted, bool)

    def test_run_flowmc_chain_history_shapes(self, lp_eta_fn):
        torch.manual_seed(0)
        flow = RealNVP(D=12, n_layers=4, hidden=32)
        eta_inits = torch.zeros(3, 12, dtype=torch.float64)
        chains, history, local_acc, global_acc = run_flowmc(
            eta_inits, lp_eta_fn, flow,
            n_iter=5, mala_tau=1e-4,
            n_local_per_global=2, lr=1e-3,
            flow_warmup=2, buffer_size=10,
        )
        assert chains.shape == (5, 3, 12)
        assert len(history) == 5
        assert 0.0 <= local_acc <= 1.0
        assert 0.0 <= global_acc <= 1.0

    def test_run_flowmc_chains_finite(self, lp_eta_fn):
        torch.manual_seed(0)
        flow = RealNVP(D=12, n_layers=4, hidden=32)
        eta_inits = torch.zeros(2, 12, dtype=torch.float64)
        chains, _, _, _ = run_flowmc(
            eta_inits, lp_eta_fn, flow,
            n_iter=3, mala_tau=1e-4,
            n_local_per_global=1, lr=1e-3,
            flow_warmup=1, buffer_size=5,
        )
        assert chains.isfinite().all()

    def test_run_flowmc_history_finite(self, lp_eta_fn):
        torch.manual_seed(0)
        flow = RealNVP(D=12, n_layers=4, hidden=32)
        eta_inits = torch.zeros(2, 12, dtype=torch.float64)
        _, history, _, _ = run_flowmc(
            eta_inits, lp_eta_fn, flow,
            n_iter=5, mala_tau=1e-4,
            n_local_per_global=1, lr=1e-3,
            flow_warmup=1, buffer_size=5,
        )
        assert all(math.isfinite(v) for v in history)
