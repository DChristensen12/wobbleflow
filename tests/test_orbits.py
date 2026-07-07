"""Tests for src/orbits: kepler, priors, transforms."""
import math

import numpy as np
import pytest
import torch

from src.orbits.kepler import (
    log_likelihood_two_planet,
    rv_model_two_planet,
    rv_one_planet,
    solve_kepler_equation,
    true_anomaly,
)
from src.orbits.priors import log_posterior_two_planet, log_prior_two_planet
from src.orbits.transforms import (
    P1_LOG_MEAN,
    P2_LOG_MEAN,
    eta_samples_to_theta,
    eta_to_theta,
    initial_eta,
    log_jacobian,
    log_posterior_unconstrained,
    sort_planets_by_period,
)


class TestSolveKeplerEquation:
    def test_circular_orbit_E_equals_M(self):
        """When e=0, E=M exactly (circular orbit, one Newton step converges)."""
        M = torch.tensor([0.0, 1.0, -1.5, 3.0], dtype=torch.float64)
        e = torch.zeros(4, dtype=torch.float64)
        E = solve_kepler_equation(M, e)
        assert torch.allclose(E, torch.remainder(M + math.pi, 2 * math.pi) - math.pi, atol=1e-8)

    def test_zero_mean_anomaly_gives_zero_E(self):
        """M=0 gives E=0, for any eccentricity."""
        M = torch.zeros(5, dtype=torch.float64)
        e = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.8], dtype=torch.float64)
        E = solve_kepler_equation(M, e)
        assert torch.allclose(E, torch.zeros_like(E), atol=1e-8)

    def test_residual_below_tolerance(self):
        """M = E - e sin(E) must hold to machine precision after convergence."""
        M = torch.linspace(-3.0, 3.0, 20, dtype=torch.float64)
        e = torch.full((20,), 0.5, dtype=torch.float64)
        E = solve_kepler_equation(M, e)
        residual = torch.abs(E - e * torch.sin(E) - M)
        assert residual.max() < 1e-9

    def test_high_eccentricity_converges(self):
        """Should not produce NaN even for e close to 1."""
        M = torch.tensor([1.0, 2.0], dtype=torch.float64)
        e = torch.tensor([0.9, 0.95], dtype=torch.float64)
        E = solve_kepler_equation(M, e)
        assert E.isfinite().all()

    def test_shape_preserved(self):
        M = torch.randn(3, 4, dtype=torch.float64)
        e = torch.full((3, 4), 0.3, dtype=torch.float64)
        assert solve_kepler_equation(M, e).shape == (3, 4)


class TestTrueAnomaly:
    def test_zero_mean_anomaly_gives_zero_nu(self):
        """At M=0, nu=0 for any eccentricity."""
        M = torch.zeros(5, dtype=torch.float64)
        e = torch.tensor([0.01, 0.1, 0.3, 0.5, 0.7], dtype=torch.float64)
        nu = true_anomaly(M, e)
        assert torch.allclose(nu, torch.zeros_like(nu), atol=1e-6)

    def test_output_finite_for_range_of_inputs(self):
        M = torch.linspace(-3.0, 3.0, 12, dtype=torch.float64)
        e = torch.full((12,), 0.4, dtype=torch.float64)
        assert true_anomaly(M, e).isfinite().all()

    def test_shape_preserved(self):
        M = torch.randn(6, dtype=torch.float64)
        e = torch.full((6,), 0.2, dtype=torch.float64)
        assert true_anomaly(M, e).shape == (6,)


class TestRVModel:
    def test_zero_amplitude_gives_zero_rv(self):
        t = torch.linspace(2350.0, 2450.0, 10, dtype=torch.float64)
        rv = rv_one_planet(t,
                           P=torch.tensor(20.0, dtype=torch.float64),
                           tp=torch.tensor(2380.0, dtype=torch.float64),
                           e=torch.tensor(0.1, dtype=torch.float64),
                           omega=torch.tensor(0.0, dtype=torch.float64),
                           K=torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(rv, torch.zeros_like(rv), atol=1e-10)

    def test_two_planet_zero_amplitude_equals_offset(self):
        """With K1=K2=0, rv_model returns v0 everywhere."""
        t = torch.linspace(2350.0, 2450.0, 10, dtype=torch.float64)
        v0 = 7.5
        theta = torch.tensor(
            [20.0, 2380.0, 0.1, 0.0, 0.0, 42.0, 2400.0, 0.1, 0.0, 0.0, v0],
            dtype=torch.float64,
        )
        rv = rv_model_two_planet(t, theta)
        assert torch.allclose(rv, torch.full_like(rv, v0), atol=1e-8)

    def test_two_planet_output_shape_matches_t(self):
        t = torch.linspace(2350.0, 2450.0, 32, dtype=torch.float64)
        theta = torch.tensor(
            [20.89, 2380.0, 0.1, 0.0, 8.0, 42.36, 2400.0, 0.1, 0.0, 6.0, 0.0],
            dtype=torch.float64,
        )
        assert rv_model_two_planet(t, theta).shape == t.shape


class TestLogLikelihood:
    def _default_inputs(self):
        t   = torch.linspace(2350.0, 2450.0, 8, dtype=torch.float64)
        rv  = torch.zeros(8, dtype=torch.float64)
        err = torch.ones(8, dtype=torch.float64) * 2.0
        theta = torch.tensor(
            [20.89, 2380.0, 0.05, 0.0, 8.0, 42.36, 2400.0, 0.05, 0.0, 6.0, 0.0],
            dtype=torch.float64,
        )
        return t, rv, err, theta

    def test_finite_at_reasonable_parameters(self):
        t, rv, err, theta = self._default_inputs()
        ll = log_likelihood_two_planet(theta, t, rv, err, torch.tensor(0.0, dtype=torch.float64))
        assert ll.isfinite()

    def test_very_large_jitter_lowers_likelihood(self):
        """Inflating jitter on a well-fitting model should decrease the log-likelihood."""
        t, rv, err, theta = self._default_inputs()
        ll_low  = log_likelihood_two_planet(theta, t, rv, err, torch.tensor(0.0,  dtype=torch.float64))
        ll_high = log_likelihood_two_planet(theta, t, rv, err, torch.tensor(10.0, dtype=torch.float64))
        assert ll_low > ll_high

    def test_better_fit_gives_higher_likelihood(self):
        """Model that matches data exactly should beat a wrong model."""
        t   = torch.linspace(2350.0, 2450.0, 8, dtype=torch.float64)
        err = torch.ones(8, dtype=torch.float64) * 2.0
        theta = torch.tensor(
            [20.89, 2380.0, 0.05, 0.0, 8.0, 42.36, 2400.0, 0.05, 0.0, 6.0, 0.0],
            dtype=torch.float64,
        )
        log_jit = torch.tensor(0.0, dtype=torch.float64)
        rv_pred = rv_model_two_planet(t, theta)
        ll_exact = log_likelihood_two_planet(theta, t, rv_pred, err, log_jit)
        rv_wrong = rv_pred + 50.0
        ll_wrong = log_likelihood_two_planet(theta, t, rv_wrong, err, log_jit)
        assert ll_exact > ll_wrong


class TestPriors:
    def _transit_theta(self):
        return torch.tensor(
            [20.89, 2380.0, 0.05, 0.0, 8.0, 42.36, 2400.0, 0.05, 0.0, 6.0, 0.0],
            dtype=torch.float64,
        )

    def test_log_prior_finite_at_transit_values(self):
        lp = log_prior_two_planet(self._transit_theta(), torch.tensor(0.0, dtype=torch.float64))
        assert lp.isfinite()

    def test_log_prior_decreases_for_absurd_period(self):
        good = self._transit_theta()
        bad  = good.clone(); bad[0] = 1e6
        lp_g = log_prior_two_planet(good, torch.tensor(0.0, dtype=torch.float64))
        lp_b = log_prior_two_planet(bad,  torch.tensor(0.0, dtype=torch.float64))
        assert lp_g > lp_b

    def test_log_prior_decreases_for_huge_amplitude(self):
        good = self._transit_theta()
        bad  = good.clone(); bad[4] = 1e5  # K1 unreasonably large
        lp_g = log_prior_two_planet(good, torch.tensor(0.0, dtype=torch.float64))
        lp_b = log_prior_two_planet(bad,  torch.tensor(0.0, dtype=torch.float64))
        assert lp_g > lp_b

    def test_log_posterior_finite(self, tiny_data):
        t, rv, err = tiny_data
        lp = log_posterior_two_planet(
            self._transit_theta(), torch.tensor(0.0, dtype=torch.float64), t, rv, err
        )
        assert lp.isfinite()


class TestTransforms:
    def test_eta_zero_maps_to_transit_periods(self):
        """eta=0 should produce P1≈20.89 d and P2≈42.36 d."""
        theta, _ = eta_to_theta(torch.zeros(12, dtype=torch.float64))
        assert abs(float(theta[0]) - math.exp(P1_LOG_MEAN)) < 0.01
        assert abs(float(theta[5]) - math.exp(P2_LOG_MEAN)) < 0.01

    def test_eta_to_theta_output_shapes(self):
        theta, lj = eta_to_theta(torch.zeros(12, dtype=torch.float64))
        assert theta.shape == (11,)
        assert lj.shape == ()

    def test_eccentricities_always_in_unit_interval(self):
        torch.manual_seed(0)
        for _ in range(20):
            eta = torch.randn(12, dtype=torch.float64) * 3
            theta, _ = eta_to_theta(eta)
            assert 0.0 <= float(theta[2]) <= 1.0
            assert 0.0 <= float(theta[7]) <= 1.0

    def test_periods_and_amplitudes_always_positive(self):
        torch.manual_seed(0)
        for _ in range(20):
            eta = torch.randn(12, dtype=torch.float64) * 3
            theta, _ = eta_to_theta(eta)
            assert float(theta[0]) > 0   # P1
            assert float(theta[5]) > 0   # P2
            assert float(theta[4]) > 0   # K1
            assert float(theta[9]) > 0   # K2

    def test_initial_eta_is_zeros(self):
        eta = initial_eta()
        assert eta.shape == (12,)
        assert torch.all(eta == 0)

    def test_log_jacobian_finite_at_zero(self):
        jac = log_jacobian(torch.zeros(12, dtype=torch.float64))
        assert jac.isfinite()

    def test_log_jacobian_finite_at_random_eta(self):
        torch.manual_seed(0)
        for _ in range(10):
            eta = torch.randn(12, dtype=torch.float64)
            assert log_jacobian(eta).isfinite()

    def test_log_posterior_unconstrained_finite_at_zero(self, tiny_data):
        t, rv, err = tiny_data
        lp = log_posterior_unconstrained(torch.zeros(12, dtype=torch.float64), t, rv, err)
        assert lp.isfinite()

    def test_log_posterior_unconstrained_decreases_far_from_zero(self, tiny_data):
        """Very large eta should have a lower log posterior than eta=0."""
        t, rv, err = tiny_data
        lp_center = log_posterior_unconstrained(torch.zeros(12, dtype=torch.float64), t, rv, err)
        lp_far    = log_posterior_unconstrained(torch.full((12,), 100.0, dtype=torch.float64), t, rv, err)
        assert lp_center > lp_far

    def test_eta_samples_to_theta_shape(self):
        eta_samples = torch.randn(50, 12, dtype=torch.float64)
        theta = eta_samples_to_theta(eta_samples)
        assert theta.shape == (50, 12)

    def test_sort_planets_enforces_p1_lt_p2(self):
        torch.manual_seed(0)
        theta = torch.rand(100, 12, dtype=torch.float64)
        theta[:, 0] = theta[:, 0] * 50 + 1    # P1 in [1, 51]
        theta[:, 5] = theta[:, 5] * 50 + 1    # P2 in [1, 51], may end up less than P1
        sorted_theta = sort_planets_by_period(theta)
        assert (sorted_theta[:, 0] <= sorted_theta[:, 5]).all()

    def test_sort_planets_already_sorted_is_unchanged(self):
        theta = torch.zeros(5, 12, dtype=torch.float64)
        theta[:, 0] = 10.0   # P1
        theta[:, 5] = 40.0   # P2 > P1
        original = theta.clone()
        assert torch.allclose(sort_planets_by_period(theta), original)

    def test_sort_planets_swapped_rows_are_corrected(self):
        theta = torch.zeros(2, 12, dtype=torch.float64)
        theta[0, 0] = 40.0; theta[0, 5] = 10.0   # P1 > P2, needs a swap
        theta[1, 0] = 10.0; theta[1, 5] = 40.0   # already sorted
        sorted_theta = sort_planets_by_period(theta)
        assert float(sorted_theta[0, 0]) < float(sorted_theta[0, 5])
        assert float(sorted_theta[1, 0]) < float(sorted_theta[1, 5])
