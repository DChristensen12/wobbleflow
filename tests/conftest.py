import numpy as np
import pytest
import torch

from src.orbits.transforms import log_posterior_unconstrained


@pytest.fixture(autouse=True)
def set_default_dtype():
    """Use float64 globally, matching the notebook's torch.set_default_dtype(torch.float64)."""
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


@pytest.fixture
def tiny_data():
    """Eight synthetic RV observations for fast tests (no network call needed)."""
    torch.manual_seed(42)
    t   = torch.linspace(2350.0, 2450.0, 8, dtype=torch.float64)
    rv  = torch.randn(8, dtype=torch.float64) * 5.0
    err = torch.ones(8, dtype=torch.float64) * 2.0
    return t, rv, err


@pytest.fixture
def lp_eta_fn(tiny_data):
    """Log-posterior closure over the tiny synthetic dataset."""
    t, rv, err = tiny_data
    def lp(eta):
        return log_posterior_unconstrained(eta, t, rv, err)
    return lp


@pytest.fixture
def eta0():
    """Prior-mode starting point: eta=0 maps to the K2-24 transit configuration."""
    return torch.zeros(12, dtype=torch.float64)
