# Navier–Stokes PINN benchmark

This repository implements the project proposal through **Stage 1: steady 2D Kovasznay flow**. The code solves cases, `Re=20` and `Re=40`, with a fully connected tanh network that predicts velocity `(u, v)` and kinematic pressure `p`.

## Stage 1 implementation

- Domain: `x ∈ [-0.5, 1]`, `y ∈ [-0.5, 1.5]`.
- The network receives the normalized coordinates plus `sin(2πy)` and `cos(2πy)` features by default, matching the Kovasznay solution's fundamental periodic variation in `y`. Disable these features with `--no-periodic-y-features` for a plain-coordinate comparison.
- A Coons-patch lifting function extends prescribed exact velocity values from the edges into the interior. The network predicts a correction multiplied by a smooth envelope that vanishes on every edge, so velocity Dirichlet conditions hold exactly without a boundary penalty or sampled boundary points. Exact interior values are not used by the lift.
- Automatic differentiation evaluates both steady momentum residuals and the continuity residual at interior collocation points.
- The objective is the sum of momentum and continuity residual MSE; there is no boundary loss because velocity boundary values are enforced exactly.
- Adam trains with exponential learning-rate decay, followed by fixed-point L-BFGS refinement. By default, RAR runs every 5,000 Adam steps: it scores 8,192 uniform candidates by momentum-residual magnitude, adds the 256 highest-residual points to the persistent adaptive set, and retains fresh uniform samples in every training batch.
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

The default command runs both Reynolds numbers with a 5-layer, 64-wide network, 20,000 Adam steps, and up to 500 L-BFGS iterations. Results go to `runs/stage1/Re_20` and `runs/stage1/Re_40`.

For a shorter smoke-sized run or a single case:

```bash
python -m navier_benchmark --reynolds 20 --steps 1000 --lbfgs-steps 50 --output runs/quick
```

Use `--rar-every 0` to disable RAR, or adjust `--rar-every`, `--rar-candidates`, and `--rar-points` to change its schedule and size. Use `--device cpu` or `--device cuda` to select compute, `--dtype float32` for faster lower-precision training, and `--help` to see all options. Float64 is the default to support the proposal's `1e-4` accuracy target. Actual accuracy depends on training duration and hardware; the code records measured errors instead of treating the target as guaranteed.


## Stage 2: Taylor–Green vortex

The Taylor–Green case has an exact unsteady solution, space-time collocation, an initial-velocity condition, and periodic boundaries. The Reynolds number and final time are command-line settings because the proposal does not fix them. The single-variant runner supports the velocity–pressure and streamfunction–pressure formulations, soft or hard periodicity, and optional causal weighting over time bins. Together, these choices form eight ablation combinations. The defaults use 1,024 fresh space-time collocation points per Adam step and float64 precision. Keep shared training and evaluation settings aligned across all eight variants for a fair comparison.

Run the first requested ablation, velocity–pressure with soft periodicity and no causal weighting:

```bash
python -m navier_benchmark.stage2 --reynolds 20 --final-time 1
```

It uses a 5-layer, 64-wide network, 20,000 Adam steps with exponential learning-rate decay, then up to 1,500 L-BFGS iterations. The defaults use 1,024 collocation points per Adam step and float64 precision; RAR is disabled. Periodic sine/cosine features for both x and y are appended to the raw coordinates, and the soft periodic edge loss remains active. The default output is `runs/stage2/velocity_pressure_soft_no_causality/Re_20`. Add `--periodicity hard` for hard periodicity or `--causality` to enable causal time-bin weights. Each run saves its full `config.json`, relative L2 metrics in `metrics.json` (pressure gauge aligned per evaluation frame), `history.csv`, `error_vs_time.csv`, `model.pt`, and Matplotlib plots in PNG for pointwise absolute errors, relative error versus time, and training loss. Adjust `--reynolds`, `--final-time`, `--steps`, `--lbfgs-steps`, evaluation grid options, and `--output` as needed; keep the physical and network settings fixed across ablations for a fair comparison. Matplotlib is listed in `requirements.txt`.
