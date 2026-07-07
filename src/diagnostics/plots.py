from __future__ import annotations
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from .ess import posterior_summary


def plot_per_method_panels(history, history_label: str,
                           p1_samples: np.ndarray, p2_samples: np.ndarray,
                           method_name: str,
                           hist_color_p1: str = "C1", hist_color_p2: str = "C2"):
    """1x3 panel: training metric on the left, then P1 and P2 posteriors.
    Every method uses this same layout so the plots are easy to compare side by side.
    """
    p1 = posterior_summary(p1_samples)
    p2 = posterior_summary(p2_samples)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history, lw=0.8, color="C0")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(history_label)
    axes[0].set_title(f"{method_name}: {history_label}")
    axes[0].grid(alpha=0.3)

    for ax, samples, summ, color, transit, label in [
        (axes[1], p1_samples, p1, hist_color_p1, 20.89, "P_1"),
        (axes[2], p2_samples, p2, hist_color_p2, 42.36, "P_2"),
    ]:
        ax.hist(samples, bins=50, density=True, color=color,
                alpha=0.7, edgecolor="black")
        ax.axvline(transit, ls="--", color="r",
                   label=f"K2 transit ({transit:.2f} d)")
        ax.axvline(summ["median"], ls="-", color="k", alpha=0.5,
                   label=f"posterior median ({summ['median']:.2f} d)")
        ymax = ax.get_ylim()[1]
        ax.fill_between([summ["hdi_lo"], summ["hdi_hi"]], 0, ymax,
                        alpha=0.2, color="gray", label="95% HDI")
        ax.set_xlabel(rf"${label}$ [days]")
        ax.set_ylabel("density")
        ax.set_title(f"{method_name} posterior: {label.replace('_', '')}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, axes


def plot_period_posteriors(samples_dict: dict,
                           transit_truth: Tuple[float, float] = (20.89, 42.36)):
    """Overlaid histograms of P1 and P2 posteriors, one line per method.
    samples_dict maps method name to its theta samples array. P1 lives at index 0,
    P2 at index 5, per the usual theta layout.
    """
    bins_P1 = np.linspace(15, 28, 60)
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
    axes[0].set_xlabel(r"$P_1$ [days]"); axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel(r"$P_2$ [days]"); axes[1].set_ylabel("density"); axes[1].legend(fontsize=8)
    axes[0].set_title("Posterior over orbital periods: four methods compared")
    fig.tight_layout()
    return fig, axes


def plot_rv_with_models(t_obs: np.ndarray, rv_obs: np.ndarray, rv_err: np.ndarray,
                        t_dense: np.ndarray, models: dict,
                        title: str = "K2-24 RV data and posterior-mean models"):
    """RV data with each method's posterior-mean curve drawn on top.
    models maps method name to an rv curve array, each aligned to t_dense.
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


# color scheme per method, just for visual consistency across figures
THEMED_PALETTES = {
    "HMC": {
        "line":      "#3FA34D",
        "marker_fc": "#1A1A1A",
        "marker_ec": "#D7263D",
    },
    "mean-field VI": {
        "line":      "#6B4423",
        "marker_fc": "#0F1A40",
        "marker_ec": "#000000",
    },
    "flow VI": {
        "line":      "#1F4FB8",
        "marker_fc": "#F4C430",
        "marker_ec": "#000000",
    },
    "flow MCMC": {
        "line":      "#6B8E23",
        "marker_fc": "#7C5A2A",
        "marker_ec": "#F1E5C5",
    },
}


def plot_themed_rv_fits(t_obs: np.ndarray, rv_obs: np.ndarray, rv_err: np.ndarray,
                        t_dense: np.ndarray, models: dict):
    """Same RV plot as plot_rv_with_models, but split into a 2x2 grid with each method's own color theme.
    The data gets replotted in every panel so the markers can carry that method's
    fill and edge color, makes it easy to tell curve and marker set apart at a glance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (name, curve) in zip(axes.flat, models.items()):
        palette = THEMED_PALETTES.get(name, {"line": "C0", "marker_fc": "k", "marker_ec": "k"})
        ax.errorbar(
            t_obs, rv_obs, yerr=rv_err,
            fmt="o", capsize=2,
            markerfacecolor=palette["marker_fc"],
            markeredgecolor=palette["marker_ec"],
            ecolor=palette["marker_ec"],
            markersize=5,
            label="K2-24 RV data",
        )
        ax.plot(t_dense, curve, lw=1.6, color=palette["line"],
                label=f"{name} posterior mean")
        ax.set_xlabel("time [days]")
        ax.set_ylabel("radial velocity [m/s]")
        ax.set_title(f"{name}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Themed RV fits by method", y=1.00)
    fig.tight_layout()
    return fig, axes
