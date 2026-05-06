"""End-to-end pipeline for the wobbleflow K2-24 analysis."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.orbits.transforms import (
    eta_samples_to_theta,
    initial_eta,
    log_posterior_unconstrained,
)
from src.flows.realnvp import RealNVP
from src.inference import hmc, vi_meanfield, vi_flow, flowmc
from src.diagnostics import ess, plots


DATA_URL = (
    "https://raw.githubusercontent.com/California-Planet-Search/"
    "radvel/master/example_data/epic203771098.csv"
)


def load_data(url: str = DATA_URL):
    """Load the K2-24 RV time series and return (t, rv, sigma) tensors."""
    df = pd.read_csv(url)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    t_obs = torch.tensor(df.t.values)
    rv_obs = torch.tensor(df.vel.values)
    rv_err = torch.tensor(df.errvel.values)
    return t_obs, rv_obs, rv_err


def run_pipeline(
    n_hmc_samples: int = 2000,
    n_hmc_burn: int = 500,
    hmc_eps: float = 2e-3,
    hmc_leapfrog: int = 30,
    n_vi_iter: int = 1500,
    flow_layers: int = 8,
    n_flowmc_iter: int = 400,
    n_chains: int = 8,
    mala_tau: float = 5e-5,
    output_dir: str = "results",
    figures_dir: str = "figures",
    seed: int = 0,
):
    """Run all four inference methods on K2-24, save results, and then save figures."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = Path(figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    # --- Data --- # 
    print("Loading K2-24 data")
    t_obs, rv_obs, rv_err = load_data()
    span = float(t_obs.max() - t_obs.min())
    print(f"  {t_obs.numel()} observations, span {span:.1f} days")

    # Closure that all methods target.
    def lp_eta(eta):
        return log_posterior_unconstrained(eta, t_obs, rv_obs, rv_err)

    eta0 = initial_eta()

    # --- Hamiltonian Monte Carlo Baseline --- #
    print("\nRunning HMC")
    hmc_eta, hmc_acc = hmc.run_hmc(
        eta0, lp_eta,
        n_samples=n_hmc_samples,
        epsilon=hmc_eps,
        L=hmc_leapfrog,
        burn_in=n_hmc_burn,
        verbose=True,
    )
    print(f"HMC accept rate: {hmc_acc:.3f}")
    hmc_theta = eta_samples_to_theta(hmc_eta)
    torch.save({"samples": hmc_theta, "accept_rate": hmc_acc}, out / "hmc.pt")

    # --- Mean-field Variational Inference Baseline --- # 
    print("\nRunning mean-field VI")
    mu_mf, log_s_mf, mf_hist = vi_meanfield.fit(
        eta0, lp_eta, n_iter=n_vi_iter, verbose=True,
    )
    mf_eta = vi_meanfield.sample(mu_mf, log_s_mf, n=2000)
    mf_theta = eta_samples_to_theta(mf_eta)
    torch.save({
        "samples": mf_theta, "mu": mu_mf, "log_s": log_s_mf, "history": mf_hist,
    }, out / "vi_meanfield.pt")

    # --- Flow Variational Inference --- #
    print("\nRunning flow VI")
    fvi_model = vi_flow.FlowVI(D=12, K=flow_layers, mu_init=eta0)
    fvi_hist = vi_flow.fit(fvi_model, lp_eta, n_iter=n_vi_iter, verbose=True)
    with torch.no_grad():
        fvi_eta, _ = fvi_model.sample_and_log_q(2000)
    fvi_theta = eta_samples_to_theta(fvi_eta)
    torch.save({
        "samples": fvi_theta,
        "model_state": fvi_model.state_dict(),
        "history": fvi_hist,
    }, out / "vi_flow.pt")

    # --- Flow Markov Chain Monte Carlo --- #
    print("\nRunning flow MCMC")
    eta_inits = eta0.unsqueeze(0) + 0.05 * torch.randn(n_chains, 12)
    flow_for_mcmc = RealNVP(D=12, n_layers=6, hidden=64)
    fmc_chains, fmc_hist, fmc_local, fmc_global = flowmc.run_flowmc(
        eta_inits, lp_eta, flow_for_mcmc,
        n_iter=n_flowmc_iter,
        mala_tau=mala_tau,
        n_local_per_global=5,
        lr=5e-3,
        verbose=True,
    )
    burn = n_flowmc_iter // 4
    fmc_pool_eta = fmc_chains[burn:].reshape(-1, 12)
    fmc_theta = eta_samples_to_theta(fmc_pool_eta)
    torch.save({
        "samples": fmc_theta,
        "chains": fmc_chains,
        "history": fmc_hist,
        "local_acc": fmc_local,
        "global_acc": fmc_global,
        "flow_state": flow_for_mcmc.state_dict(),
    }, out / "flowmc.pt")

    # --- Diagnostics --- #
    print("\nDiagnostics:")
    ess_p1 = ess.ess_1d(hmc_theta[:, 0].numpy())
    ess_p2 = ess.ess_1d(hmc_theta[:, 5].numpy())
    print(f"  HMC ESS (P1): {ess_p1:.1f}")
    print(f"  HMC ESS (P2): {ess_p2:.1f}")

    # --- Figures --- #
    print("\nGenerating figures")
    samples_dict = {
        "HMC": hmc_theta.numpy(),
        "mean-field VI": mf_theta.numpy(),
        "flow VI": fvi_theta.numpy(),
        "flow MCMC": fmc_theta.numpy(),
    }
    fig, _ = plots.plot_period_posteriors(samples_dict)
    fig_path = figs / "period_posteriors.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {fig_path}")

    print("\nDone.")
    return {
        "hmc_theta": hmc_theta, "hmc_acc": hmc_acc,
        "mf_theta": mf_theta, "mf_history": mf_hist,
        "fvi_theta": fvi_theta, "fvi_history": fvi_hist,
        "fmc_theta": fmc_theta, "fmc_history": fmc_hist,
        "fmc_local_acc": fmc_local, "fmc_global_acc": fmc_global,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-hmc-samples", type=int, default=2000)
    p.add_argument("--n-hmc-burn", type=int, default=500)
    p.add_argument("--hmc-eps", type=float, default=2e-3)
    p.add_argument("--hmc-leapfrog", type=int, default=30)
    p.add_argument("--n-vi-iter", type=int, default=1500)
    p.add_argument("--flow-layers", type=int, default=8)
    p.add_argument("--n-flowmc-iter", type=int, default=400)
    p.add_argument("--n-chains", type=int, default=8)
    p.add_argument("--mala-tau", type=float, default=5e-5)
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--figures-dir", type=str, default="figures")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(**vars(args))