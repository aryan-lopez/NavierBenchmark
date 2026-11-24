# Navier–Stokes PINN benchmark

This repository implements the project proposal through **Stage 1: steady 2D Kovasznay flow**. The code solves cases, `Re=20` and `Re=40`, with a fully connected tanh network that predicts velocity `(u, v)` and kinematic pressure `p`.

## Stage 1 implementation

The Stage 1 runner is `navier_benchmark/stage1.py`; its Kovasznay model and training code are in `navier_benchmark/kovasznay_model.py` and `navier_benchmark/kovasznay_solver.py`. Run it with `python -m navier_benchmark.stage1`; the package shortcut `python -m navier_benchmark` also runs Stage 1.

- Domain: `x ∈ [-0.5, 1]`, `y ∈ [-0.5, 1.5]`.
- The network receives the normalized coordinates plus `sin(2πy)` and `cos(2πy)` features by default, matching the Kovasznay solution's fundamental periodic variation in `y`. Disable these features with `--no-periodic-y-features` for a plain-coordinate comparison.
- A Coons-patch lifting function extends prescribed exact velocity values from the edges into the interior. The network predicts a correction multiplied by a smooth envelope that vanishes on every edge, so velocity Dirichlet conditions hold exactly without a boundary penalty or sampled boundary points. Exact interior values are not used by the lift.
- Automatic differentiation evaluates both steady momentum residuals and the continuity residual at interior collocation points.
- The objective is the sum of momentum and continuity residual MSE; there is no boundary loss because velocity boundary values are enforced exactly.
- Adam trains with exponential learning-rate decay, followed by fixed-point L-BFGS refinement. RAR is disabled by default; when enabled, it scores 8,192 uniform candidates by momentum-residual magnitude, adds the 256 highest-residual points to the persistent adaptive set, and retains fresh uniform samples in every training batch.
- Evaluation uses a dense grid independent of the training samples. Relative L2 errors are reported for `u`, `v`, pressure, and the velocity pair. Pressure means are aligned before measuring pressure error because pressure is defined up to an additive constant.
- Each run saves `model.pt`, `metrics.json`, and `history.csv`.

## Install and run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m navier_benchmark
```

The default command runs both Reynolds numbers with a 5-layer, 64-wide network, 20,000 Adam steps, and up to 1,500 L-BFGS iterations. Periodic `y` features are enabled and RAR is disabled by default. Results go to `runs/stage1/default/Re_20` and `runs/stage1/default/Re_40`; all Stage 1 variants are grouped under `runs/stage1/`.

For a shorter smoke-sized run or a single case:

```bash
python -m navier_benchmark --reynolds 20 --steps 1000 --lbfgs-steps 50 --output runs/quick
```

Use a positive `--rar-every` value to enable RAR, and adjust `--rar-candidates` and `--rar-points` to change its schedule and size. Use `--device cpu` or `--device cuda` to select compute, `--dtype float32` for faster lower-precision training, and `--help` to see all options. Float64 is the default to support the proposal's `1e-4` accuracy target. Actual accuracy depends on training duration and hardware; the code records measured errors instead of treating the target as guaranteed.


## Stage 2: Taylor–Green vortex

The Taylor–Green case has an exact unsteady solution, space-time collocation, an initial-velocity condition, and periodic boundaries. The Reynolds number and final time are command-line settings because the proposal does not fix them. The single-variant runner supports the velocity–pressure and streamfunction–pressure formulations, soft or hard periodicity, and optional causal weighting over time bins. Together, these choices form eight ablation combinations. The defaults use 1,024 fresh space-time collocation points per Adam step and float64 precision. Keep shared training and evaluation settings aligned across all eight variants for a fair comparison.

Run the first requested ablation, velocity–pressure with soft periodicity and no causal weighting:

```bash
python -m navier_benchmark.stage2 --reynolds 20 --final-time 1
```

It uses a 5-layer, 64-wide network, 20,000 Adam steps with exponential learning-rate decay, then up to 1,500 L-BFGS iterations. The defaults use 1,024 collocation points per Adam step and float64 precision; RAR is disabled. Periodic sine/cosine features for both x and y are appended to the raw coordinates, and the soft periodic edge loss remains active. The default output is `runs/stage2/velocity_pressure_soft_no_causality/Re_20`. Add `--periodicity hard` for hard periodicity or `--causality` to enable causal time-bin weights. Each run saves its full `config.json`, relative L2 metrics in `metrics.json` (pressure gauge aligned per evaluation frame), `history.csv`, `error_vs_time.csv`, `model.pt`, and Matplotlib plots in PNG for pointwise absolute errors, relative error versus time, and training loss. Adjust `--reynolds`, `--final-time`, `--steps`, `--lbfgs-steps`, evaluation grid options, and `--output` as needed; keep the physical and network settings fixed across ablations for a fair comparison. Matplotlib is listed in `requirements.txt`.

## Stage 3: 3D Ethier--Steinman (Beltrami) flow

`navier_benchmark.ethier_steinman.EthierSteinmanFlow` provides the proposal's unsteady 3D exact solution on `[-1, 1]^3 x [0, 1]`, with `a=d=1`. Pass the kinematic viscosity when constructing the case; this is fixed for a forward run and can be made trainable by an inverse-training model. Coordinates are columns `(x, y, z, t)`, and `exact()` returns `(u, v, w, p)`. The case also samples space-time collocation points, initial velocity points at `t=0`, and velocity Dirichlet data on all six spatial faces for `t` in `[0, 1]`. Pressure is available for reference/evaluation and is not imposed on the boundary. `unsteady_navier_stokes_3d_residual()` evaluates the three momentum residuals and incompressibility residual and accepts a scalar tensor viscosity for inverse-training gradients.

The forward trainer reuses the Stage 1 Adam learning-rate decay and L-BFGS sequence. Defaults are 20,000 Adam steps, 1,500 L-BFGS steps, periodic sine/cosine features of `y`, and no RAR. Initial and six-face velocity boundary data enter as soft losses. Run it with:

```bash
python -m navier_benchmark.stage3 --viscosity 0.01
```

The default output is `runs/stage3/forward`. It saves model weights, training history, configuration, relative L2 errors for each velocity component and pressure on a dense space-time grid, pointwise absolute-error maps on the central `z=0` slice, and relative error versus time (`error_vs_time.csv` and `.png`). Adjust `--eval-resolution` and `--eval-times` to control evaluation-grid size; set `--plot-time` to choose the map time (defaults to final time). Override defaults with `--no-periodic-y-features` or set `--rar-every` to a positive number to enable RAR.
