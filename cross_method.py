from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.orbits.kepler import rv_model_two_planet
from src.orbits.transforms import (
    log_posterior_unconstrained,
    initial_eta,
    eta_samples_to_theta,
    sort_planets_by_period,
)
from src.flows.realnvp import RealNVP
from src.inference.hmc import run_hmc_multichain
from src.inference.vi_meanfield import fit_meanfield, sample_meanfield
from src.inference.vi_flow import FlowVI, fit_flow_vi
from src.inference.flowmc import run_flowmc
from src.diagnostics.ess import ess_1d, posterior_summary
from src.diagnostics.plots import (
    plot_period_posteriors,
    plot_rv_with_models,
    plot_themed_rv_fits,
    THEMED_PALETTES,
)

df = pd.read_csv('https://raw.githubusercontent.com/California-Planet-Search/radvel/master/example_data/epic203771098.csv')
df = df.drop(columns=['Unnamed: 0'])

# Configuration shared across methods
N_HMC_CHAINS   = 12
N_HMC_SAMPLES  = 500
N_HMC_BURN     = 300
HMC_EPS        = 0.01
HMC_LEAPFROG   = 30
N_MFVI_ITER    = 2000
N_VI_ITER      = 5000
VI_LR          = 1e-2
FLOW_LAYERS    = 8
FVI_LR         = 1e-3
N_FLOWMC_ITER  = 600
N_FMC_CHAINS   = 24
MALA_TAU       = 5e-4
LR_FLOW        = 5e-3
FLOW_WARMUP    = 50
BUFFER_SIZE    = 2000
SEED           = 0


# Every torch.manual_seed(0) call below is there so this script gives identical
# numbers on every run. Comment those lines out if you want fresh draws each time
# (results should look qualitatively the same, just noisier).
torch.set_default_dtype(torch.float64)
np.random.seed(SEED)


os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Data
t_obs  = torch.tensor(df.t.values)
rv_obs = torch.tensor(df.vel.values)
rv_err = torch.tensor(df.errvel.values)
span = float(t_obs.max() - t_obs.min())
print(f"{t_obs.numel()} observations, span {span:.1f} days")


def lp_eta(eta):
    """Log posterior in eta-space, closed over the K2-24 data so every method can just call lp_eta(eta)."""
    return log_posterior_unconstrained(eta, t_obs, rv_obs, rv_err)


eta0 = initial_eta()

print("\nHMC")
torch.manual_seed(0)
all_hmc_samples, all_hmc_acc = run_hmc_multichain(
    eta0, lp_eta,
    n_chains=N_HMC_CHAINS,
    n_samples=N_HMC_SAMPLES,
    epsilon=HMC_EPS, L=HMC_LEAPFROG,
    n_burnin=N_HMC_BURN, verbose=True,
)
hmc_eta = torch.cat(all_hmc_samples, dim=0)
hmc_theta = sort_planets_by_period(eta_samples_to_theta(hmc_eta))
torch.save({"samples": hmc_theta, "accept_rates": all_hmc_acc}, "results/hmc.pt")

print("\nMean-field VI")
torch.manual_seed(0)
mu_mf, log_s_mf, mf_hist = fit_meanfield(
    eta0, lp_eta, n_iter=N_MFVI_ITER, lr=VI_LR, n_mc=64, verbose=True,
)
mf_eta = sample_meanfield(mu_mf, log_s_mf, n=2000)
mf_theta = sort_planets_by_period(eta_samples_to_theta(mf_eta))
torch.save({
    "samples": mf_theta, "mu": mu_mf, "log_s": log_s_mf, "history": mf_hist,
}, "results/vi_meanfield.pt")

print("\nFlow VI")
torch.manual_seed(0)
fvi_model = FlowVI(D=12, K=FLOW_LAYERS, mu_init=eta0, log_s_init=-3.0)
fvi_hist = fit_flow_vi(fvi_model, lp_eta, n_iter=N_VI_ITER, lr=FVI_LR, verbose=True)
with torch.no_grad():
    fvi_eta, _ = fvi_model.sample_and_log_q(2000)
fvi_theta = sort_planets_by_period(eta_samples_to_theta(fvi_eta))
torch.save({
    "samples": fvi_theta,
    "model_state": fvi_model.state_dict(),
    "history": fvi_hist,
}, "results/vi_flow.pt")

print("\nFlow MCMC")
torch.manual_seed(0)
eta_inits = torch.zeros(N_FMC_CHAINS, 12) + 0.01 * torch.randn(N_FMC_CHAINS, 12)
flow_for_mcmc = RealNVP(D=12, n_layers=6, hidden=64)
fmc_chains, fmc_hist, fmc_local, fmc_global = run_flowmc(
    eta_inits, lp_eta, flow_for_mcmc,
    n_iter=N_FLOWMC_ITER,
    mala_tau=MALA_TAU,
    n_local_per_global=5, lr=LR_FLOW,
    flow_warmup=FLOW_WARMUP, buffer_size=BUFFER_SIZE,
    verbose=True,
)
burn = N_FLOWMC_ITER // 4
fmc_pool_eta = fmc_chains[burn:].reshape(-1, 12)
fmc_theta = sort_planets_by_period(eta_samples_to_theta(fmc_pool_eta))
torch.save({
    "samples":    fmc_theta,
    "chains":     fmc_chains,
    "history":    fmc_hist,
    "local_acc":  fmc_local,
    "global_acc": fmc_global,
    "flow_state": flow_for_mcmc.state_dict(),
}, "results/flowmc.pt")

print("\nDiagnostics")
ess_hmc_p1 = ess_1d(hmc_theta[:, 0].numpy())
ess_hmc_p2 = ess_1d(hmc_theta[:, 5].numpy())
ess_fmc_p1 = ess_1d(fmc_theta[:, 0].numpy())
ess_fmc_p2 = ess_1d(fmc_theta[:, 5].numpy())
print(f"  HMC ESS    P1: {ess_hmc_p1:.1f} / {len(hmc_theta)}  "
      f"P2: {ess_hmc_p2:.1f} / {len(hmc_theta)}")
print(f"  Flow MCMC ESS P1: {ess_fmc_p1:.1f} / {len(fmc_theta)}  "
      f"P2: {ess_fmc_p2:.1f} / {len(fmc_theta)}")

# Per-method summary table
print("\nPer-method posterior summaries")
print(f"{'method':<16} {'P1 median':>11} {'P1 std':>10} {'P2 median':>11} {'P2 std':>10}")
for name, theta in [
    ("HMC", hmc_theta),
    ("mean-field VI", mf_theta),
    ("flow VI", fvi_theta),
    ("flow MCMC", fmc_theta),
]:
    s1 = posterior_summary(theta[:, 0].numpy())
    s2 = posterior_summary(theta[:, 5].numpy())
    print(f"{name:<16} {s1['median']:>11.3f} {s1['std']:>10.3f} "
          f"{s2['median']:>11.3f} {s2['std']:>10.3f}")

print("\nFigures")
samples_dict = {
    "HMC":           hmc_theta.numpy(),
    "mean-field VI": mf_theta.numpy(),
    "flow VI":       fvi_theta.numpy(),
    "flow MCMC":     fmc_theta.numpy(),
}

fig, _ = plot_period_posteriors(samples_dict)
fig.savefig("figures/period_posteriors.png", dpi=150, bbox_inches="tight")
print("  saved figures/period_posteriors.png")

t_dense = np.linspace(float(t_obs.min()), float(t_obs.max()), 500)
t_dense_t = torch.tensor(t_dense)


def posterior_mean_rv(theta_samples: torch.Tensor) -> np.ndarray:
    """Average the RV curve over posterior samples so we get one mean curve per method to plot."""
    rv_curves = torch.stack([
        rv_model_two_planet(t_dense_t, theta_samples[i, :11])
        for i in range(len(theta_samples))
    ])
    return rv_curves.mean(0).numpy()


rv_models = {name: posterior_mean_rv(torch.tensor(s)) for name, s in samples_dict.items()}
fig2, _ = plot_rv_with_models(
    t_obs.numpy(), rv_obs.numpy(), rv_err.numpy(), t_dense, rv_models,
)
fig2.savefig("figures/rv_fit.png", dpi=150, bbox_inches="tight")
print("  saved figures/rv_fit.png")

# same RV fits, but split into a 2x2 grid, one panel per method
fig3, _ = plot_themed_rv_fits(
    t_obs.numpy(), rv_obs.numpy(), rv_err.numpy(), t_dense, rv_models,
)
fig3.savefig("figures/rv_fit_themed.png", dpi=150, bbox_inches="tight")
print("  saved figures/rv_fit_themed.png")

# side by side comparison of how wide each method's posterior ends up
fig4, axes = plt.subplots(1, 2, figsize=(11, 4))
method_order = ["HMC", "mean-field VI", "flow VI", "flow MCMC"]
p1_stds = [posterior_summary(samples_dict[m][:, 0])["std"] for m in method_order]
p2_stds = [posterior_summary(samples_dict[m][:, 5])["std"] for m in method_order]
colors = [THEMED_PALETTES[m]["line"] for m in method_order]
axes[0].bar(method_order, p1_stds, color=colors, edgecolor="black")
axes[0].set_ylabel("posterior std of P1 [days]")
axes[0].set_title("P1 posterior width by method")
axes[0].tick_params(axis="x", rotation=20)
axes[1].bar(method_order, p2_stds, color=colors, edgecolor="black")
axes[1].set_ylabel("posterior std of P2 [days]")
axes[1].set_title("P2 posterior width by method")
axes[1].tick_params(axis="x", rotation=20)
fig4.tight_layout()
fig4.savefig("figures/posterior_widths.png", dpi=150, bbox_inches="tight")
print("  saved figures/posterior_widths.png")

print("\nAnalysis Finished :)")
