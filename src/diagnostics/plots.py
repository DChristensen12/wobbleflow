# Plotting helpers shared across the four inference methods.

from __future__ import annotations
from typing import Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_rv_with_models(t_obs: np.ndarray,
                        rv_obs: np.ndarray,
                        rv_err: np.ndarray,
                        t_dense: np.ndarray,
                        models: dict,
                        title: str = "K2-24 RV data and posterior-mean models"
                        ) -> Tuple[plt.Figure, plt.Axes]:
    """RV data with overlaid posterior-mean RV curves from each method.
    `models` is a dict of {method_name: rv_curve_array} where each curve has
    the same length as t_dense.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.errorbar(t_obs, rv_obs, yerr=rv_err, fmt=".k", capsize=2, label="K2-24 RV data")
    for name, curve in models.items():
        ax.plot(t_dense, curve, lw=1.2, label=f"{name} posterior mean")
    ax.set_xlabel("time [days]")
    ax.set_ylabel("radial velocity [m/s]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_period_posteriors(samples_dict: dict,
                           transit_truth: Tuple[float, float] = (20.89, 42.36),
                           bins_P1: Sequence[float] = None,
                           bins_P2: Sequence[float] = None
                           ) -> Tuple[plt.Figure, np.ndarray]:
    """Stacked histograms of P1 and P2 from each method.
    `samples_dict` is {method_name: theta_samples_array}, where the period
    columns are at indices 0 (P1) and 5 (P2).
    """
    if bins_P1 is None:
        bins_P1 = np.linspace(15, 28, 60)
    if bins_P2 is None:
        bins_P2 = np.linspace(35, 50, 60)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for name, samples in samples_dict.items():
        axes[0].hist(samples[:, 0], bins=bins_P1, density=True,
                     histtype="step", lw=1.5, label=name)
        axes[1].hist(samples[:, 5], bins=bins_P2, density=True,
                     histtype="step", lw=1.5, label=name)

    axes[0].axvline(transit_truth[0], ls="--", color="k", alpha=0.5,
                    label="K2 transit value")
    axes[1].axvline(transit_truth[1], ls="--", color="k", alpha=0.5,
                    label="K2 transit value")
    axes[0].set_xlabel(r"$P_1$ [days]")
    axes[1].set_xlabel(r"$P_2$ [days]")
    for ax in axes:
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    axes[0].set_title("Posterior over orbital periods: four methods compared")
    fig.tight_layout()
    return fig, axes

