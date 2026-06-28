# wobbleflow

K2-24 (EPIC 203771098) is a star about 530 light-years away hosting two confirmed planets. The California Planet Search team measured its radial velocity 32 times with Keck/HIRES, and those measurements are the data this whole project is built around. The goal is straightforward: given the wobble of the star, figure out the orbital parameters of both planets, and then compare four different ways of doing that inference.

<div align="center">
  <img src="assets/K2-24_Artistic_Picture.png" width="600"/>
</div>
<p align="center"><b>Artistic Rendition of the  K2-24 System</b></p>

The four methods are Hamiltonian Monte Carlo, mean-field variational inference, normalizing flow-based variational inference (planar normalizing flows, following Rezende & Mohamed 2015), and flow-augmented markov chain monte carlo (Real-NVP coupled with MALA, following Gabrié et al. 2022). Each method targets the same 12-dimensional posterior. The comparison is about what you gain when you add a normalizing flow to each respective baseline.

---

## The Dataset

There are 32 radial velocity measurements from Petigura et al. (2016, 2018), available through the RadVel package. The K2 transit periods are 20.89 days for the inner planet and 42.36 days for the outer one, these are the ground truth the posteriors should be near, not exactly at, since RV data has less timing precision than transit photometry.

The data loads directly from the RadVel GitHub in both the notebook and `cross_method.py`. There's also a local copy in `data/`.

---

## Methods

**Hamiltonian Monte Carlo** -- leapfrog integrator with MH correction, run across 12 parallel chains. Each chain starts at eta=0 (which maps exactly to the K2 transit configuration). Burn-in is 300 steps per chain, then 500 samples each.

**Mean-field VI** -- diagonal Gaussian variational family, ELBO maximized with Adam and cosine LR decay. Converges quickly but can't represent parameter correlations by construction.

**Flow VI** -- same diagonal Gaussian base, but 8 planar layers push it into a richer family. Slower to train and the ELBO improvement is real (~4 nats over mean-field), though the marginal histograms look almost identical. The difference is in the joint distribution, not the marginals.

**FlowMC** -- 24 parallel MALA chains with a Real-NVP flow trained concurrently on the chain history. The flow proposes global jumps; MALA handles local exploration. After a 50-step warmup the flow starts making proposals, and even a global acceptance rate of ~0.02 is enough to make a meaningful difference in robustness.

<div align="center">
  <img src="assets/chains_comparison.gif" width="700"/>
  <br><em>HMC chains (left) vs FlowMC chains (right) exploring the P1–P2 posterior.</em>
</div>

---

## Results

All four methods agree on the posterior mode: P1 around 20.4–20.5 days, P2 around 41–41.5 days, both within about half a day of the transit values. The interesting differences are in how they characterize uncertainty.

**On the VI comparison:** the headline claim from Rezende & Mohamed is that flows give a strictly better approximation than mean-field, and the ELBOs back this up. What's subtle is that the marginal period histograms look nearly identical between the two methods. The flow's gain comes from modeling joint correlations between parameters (period-eccentricity, period-amplitude), not from changing the marginals. If you evaluated this comparison by eyeballing histograms you'd conclude the flow barely helped, the ELBO is the honest diagnostic.

Neither VI method captures multimodality. HMC's P1 histogram shows secondary peaks at aliased periods around 10 and 15 days; both VI posteriors are clean unimodals at the dominant mode. Planar flows are local deformations of a single Gaussian, not architectures that can span well-separated modes. That's a known limitation, not a bug.

**On the MCMC comparison:** HMC's pooled P2 posterior has a standard deviation of 88 days, which looks absurd. Looking at the individual chains, 11 of the 12 have acceptance rates between 0.47 and 0.93 and explore the posterior normally. Chain 6 drifted into a high-curvature region where the fixed leapfrog parameters broke down (acceptance 0.286), and at least one chain wandered into the prior tail at large P2 and got stuck. The 88-day standard deviation is from one or two pathological chains contaminating the pool. FlowMC drops P2 std to 3.15 days and produces a much cleaner posterior, not because each individual chain mixes dramatically better, but because 24 chains plus occasional global proposals gives enough redundancy that a few bad chains don't dominate.

<div align="center">
  <img src="assets/rv_fit_themed.png" width="700"/>
  <br><em>Posterior-mean RV fits from all four methods overlaid on the K2-24 data.</em>
</div>

<div align="center">
  <img src="assets/period_posteriors.png" width="600"/>
  <br><em>P1 and P2 marginal posteriors across all four methods.</em>
</div>

---

## Layout

```
wobbleflowfolder/
├── notebooks/wobbleflow.ipynb   # full analysis, interactive
├── cross_method.py              # script version: runs all four methods end-to-end
├── src/
│   ├── orbits/
│   │   ├── kepler.py            # Kepler equation, RV model, log-likelihood
│   │   ├── priors.py            # log-prior, log-posterior
│   │   └── transforms.py       # normalized reparameterization (eta-space)
│   ├── flows/
│   │   ├── planar.py            # PlanarLayer (used by flow VI)
│   │   └── realnvp.py           # CouplingLayer + RealNVP (used by FlowMC)
│   ├── inference/
│   │   ├── hmc.py               # HMC with leapfrog
│   │   ├── vi_meanfield.py      # mean-field VI
│   │   ├── vi_flow.py           # flow VI
│   │   └── flowmc.py            # FlowMC (MALA + flow proposals)
│   └── diagnostics/
│       ├── ess.py               # autocorrelation, ESS, posterior summary
│       └── plots.py             # shared plotting helpers
├── tests/                       # pytest suite
├── data/epic203771098.csv       # local copy of the K2-24 RV data
└── assets/                      # figures and animations
```

The notebook and the scripts are the same code. The notebook is for interactive exploration; `cross_method.py` and the `src/` library are for running it straight from an IDE or terminal.

---

## Running It

```bash
pip install -r requirements.txt
```

**Notebook** (interactive, it has in depth math explanations and you can use it for exploration):
```bash
jupyter notebook notebooks/wobbleflow.ipynb
```

**All four methods at once** (saves results to `results/` and figures to `figures/`):
```bash
python cross_method.py
```

**Tests:**
```bash
pytest tests/
```

Heads up: `cross_method.py` does a full run, HMC across 12 chains, 5000 VI iterations for flow VI, 600 FlowMC outer iterations across 24 chains. It takes a while. The notebook has the same runs broken into individual cells so you can run methods one at a time if you'd prefer. 

---

## References

Petigura, E. A.; Howard, A. W.; Lopez, E. D.; et al. Two Transiting Low-Density Sub-Saturns from K2. Astrophys. J. 2016, 818 (1), 36. 
- For the K2-24 radial velocity measurements

Rezende, D. J.; Mohamed, S. Variational Inference with Normalizing Flows. Proceedings of the 32nd International Conference on Machine Learning (ICML) 2015, 37, 1530-1538. 
- For Variational inference with normalizing flows (planar layers)

Gabrié, M.; Rotskoff, G. M.; Vanden-Eijnden, E. Adaptive Monte Carlo Augmented with Normalizing Flows. Proc. Natl. Acad. Sci. U.S.A. 2022, 119 (10), e2109420119.
- (For adaptive MCMC with normalizing flows (FlowMC))

Kipping, D. M. Parametrizing the Exoplanet Eccentricity Distribution with the Beta Distribution. Mon. Not. R. Astron. Soc.: Lett. 2013, 434 (1), L51-L55. 
- For the empirical eccentricity prior (Beta distribution)
