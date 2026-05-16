"""Tests for src/diagnostics: ESS, autocorrelation, posterior summary."""
import math

import numpy as np
import pytest

from src.diagnostics.ess import autocorr_1d, ess_1d, posterior_summary


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

class TestAutocorr1d:
    def test_lag_zero_is_one(self):
        """rho[0] must be exactly 1 by definition."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        rho = autocorr_1d(x, max_lag=50)
        assert abs(rho[0] - 1.0) < 1e-10

    def test_iid_acf_small_beyond_lag_zero(self):
        """For iid samples, |rho_k| for k>0 should be small (within sampling noise)."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(2000)
        rho = autocorr_1d(x, max_lag=20)
        assert np.all(np.abs(rho[1:]) < 0.15)

    def test_high_ar1_acf_decays_slowly(self):
        """AR(1) with phi=0.95 should have rho[1] close to 0.95."""
        rng = np.random.default_rng(7)
        x = np.zeros(2000)
        for i in range(1, 2000):
            x[i] = 0.95 * x[i - 1] + rng.standard_normal()
        rho = autocorr_1d(x, max_lag=5)
        assert rho[1] > 0.7

    def test_output_length_matches_max_lag(self):
        rho = autocorr_1d(np.random.standard_normal(100), max_lag=30)
        assert len(rho) == 30

    def test_default_max_lag_is_n_over_4(self):
        x = np.random.standard_normal(200)
        rho = autocorr_1d(x)
        assert len(rho) == 50  # 200 // 4

    def test_negative_acf_for_alternating_chain(self):
        """Alternating +1/-1 chain should have strongly negative rho[1]."""
        x = np.array([(-1) ** i for i in range(200)], dtype=float)
        rho = autocorr_1d(x, max_lag=3)
        assert rho[1] < -0.9


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------

class TestESS1d:
    def test_iid_ess_near_n(self):
        """ESS of an iid chain should be at least 50% of N."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000)
        assert ess_1d(x) > 500

    def test_highly_correlated_ess_much_less_than_n(self):
        """AR(1) with phi=0.99 should have ESS << N."""
        rng = np.random.default_rng(0)
        x = np.zeros(500)
        for i in range(1, 500):
            x[i] = 0.99 * x[i - 1] + rng.standard_normal()
        assert ess_1d(x) < 50

    def test_ess_positive(self):
        assert ess_1d(np.random.standard_normal(100)) > 0

    def test_ess_at_most_n(self):
        x = np.random.standard_normal(100)
        assert ess_1d(x) <= 100

    def test_ess_decreases_with_more_autocorrelation(self):
        """Higher AR coefficient → lower ESS."""
        rng = np.random.default_rng(1)
        n = 500

        x_low = np.zeros(n)
        for i in range(1, n):
            x_low[i] = 0.5 * x_low[i - 1] + rng.standard_normal()

        x_high = np.zeros(n)
        for i in range(1, n):
            x_high[i] = 0.95 * x_high[i - 1] + rng.standard_normal()

        assert ess_1d(x_low) > ess_1d(x_high)


# ---------------------------------------------------------------------------
# Posterior summary
# ---------------------------------------------------------------------------

class TestPosteriorSummary:
    def test_required_keys_present(self):
        s = posterior_summary(np.random.standard_normal(100))
        for key in ("mean", "std", "median", "hdi_lo", "hdi_hi"):
            assert key in s

    def test_mean_and_median_correct_for_known_data(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = posterior_summary(x)
        assert abs(s["mean"]   - 3.0) < 1e-8
        assert abs(s["median"] - 3.0) < 1e-8

    def test_hdi_bounds_ordered(self):
        x = np.random.standard_normal(500)
        s = posterior_summary(x)
        assert s["hdi_lo"] < s["median"] < s["hdi_hi"]

    def test_hdi_covers_95_percent(self):
        """For a standard Normal, the 95% HDI should be roughly [-1.96, 1.96]."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(10_000)
        s = posterior_summary(x)
        assert abs(s["hdi_lo"] - (-1.96)) < 0.1
        assert abs(s["hdi_hi"] -   1.96)  < 0.1

    def test_all_values_are_python_floats(self):
        s = posterior_summary(np.random.standard_normal(100))
        for v in s.values():
            assert isinstance(v, float)

    def test_std_correct_for_known_data(self):
        x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        s = posterior_summary(x)
        assert abs(s["std"] - float(np.std(x))) < 1e-8
