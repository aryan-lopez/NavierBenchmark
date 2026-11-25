# Navier–Stokes PINN Benchmark

This repository contains a three-stage benchmark study of physics-informed neural networks (PINNs) for incompressible Navier–Stokes flows. It covers steady 2D Kovasznay flow at Reynolds numbers 20 and 40; the unsteady 2D Taylor–Green vortex with formulation, periodicity, and causal-training ablations; and the unsteady 3D Ethier–Steinman (Beltrami) forward problem. Stage 3 inverse viscosity estimation is not yet implemented.

## Repository layout

| Module | Purpose |
| --- | --- |
| `navier_benchmark/stage1/` | Stage 1 package |
| `navier_benchmark/stage1/cli.py` | Stage 1 command-line runner |
| `navier_benchmark/stage1/kovasznay.py` | Stage 1 exact solution, domain, and samplers |
| `navier_benchmark/stage1/kovasznay_model.py` | Stage 1 PINN architecture |
| `navier_benchmark/stage1/kovasznay_solver.py` | Stage 1 training and evaluation |
| `navier_benchmark/stage2/` | Stage 2 package |
| `navier_benchmark/stage2/cli.py` | Stage 2 command-line runner |
| `navier_benchmark/stage2/taylor_green.py` | Stage 2 exact solution and samplers |
| `navier_benchmark/stage2/taylor_green_model.py` | Stage 2 PINN formulations and periodic boundary losses |
| `navier_benchmark/stage2/taylor_green_solver.py` | Stage 2 training and evaluation |
| `navier_benchmark/stage2/compare.py` | Stage 2 ablation comparison utility |
| `navier_benchmark/stage3/` | Stage 3 package |
| `navier_benchmark/stage3/cli.py` | Stage 3 command-line runner |
| `navier_benchmark/stage3/ethier_steinman.py` | Stage 3 exact solution, domain, and samplers |
| `navier_benchmark/stage3/ethier_steinman_model.py` | Stage 3 PINN architecture |
| `navier_benchmark/stage3/ethier_steinman_solver.py` | Stage 3 training and evaluation |
| `navier_benchmark/stage2/taylor_green_residual.py` | Automatic-differentiation residuals for the Stage 2 2D problem |
| `navier_benchmark/stage3/ethier_steinman_residual.py` | Automatic-differentiation residuals for the Stage 3 3D problem |

## Requirements and installation

Python 3.10 or newer is recommended. Install the dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

All runners accept `--help` to list their options. Use `--device auto`, `--device cpu`, or `--device cuda` to select a device, and `--dtype float32` or `--dtype float64` to select precision. Float64 is the default. Training accuracy depends on the selected configuration and hardware; the reported metrics are measured results, not guarantees.

## Stage 1: steady 2D Kovasznay flow

The Stage 1 runner solves Kovasznay flow on the rectangle `x ∈ [-0.5, 1]`, `y ∈ [-0.5, 1.5]` for Reynolds numbers 20 and 40 by default. The network predicts velocity `(u, v)` and kinematic pressure `p`. Velocity boundary conditions are imposed exactly with a Coons-patch lift and a boundary-vanishing correction. The loss contains the steady momentum and incompressibility residuals.

Default training uses a 5-layer, 64-unit tanh MLP, 20,000 Adam steps with exponential learning-rate decay, followed by up to 1,500 L-BFGS iterations. Periodic `y` input features are enabled. Residual-based adaptive refinement (RAR) is disabled by default.

Run both default Reynolds-number cases:

```powershell
python -m navier_benchmark.stage1
```

The package shortcut `python -m navier_benchmark` runs the same command. Results are saved in `runs/stage1/default/Re_20` and `runs/stage1/default/Re_40`. Older Stage 1 variants are grouped under `runs/stage1/` in separate subfolders.

For one case or a shorter run:

```powershell
python -m navier_benchmark.stage1 --reynolds 20 --steps 1000 --lbfgs-steps 50 --output runs/stage1/quick
```

Use `--no-periodic-y-features` to disable the input features. Set `--rar-every` to a positive number to enable RAR; `--rar-candidates` and `--rar-points` control its candidate and retained point counts. Runs save model weights, metrics, and training history.

## Stage 2: unsteady 2D Taylor–Green vortex

Stage 2 solves the Taylor–Green vortex with an exact unsteady solution on `[0, 2π] × [0, 2π] × [0, T]`. It supports velocity–pressure and streamfunction–pressure formulations, soft or hard periodic boundary conditions, and optional causal weighting across time bins. These choices form an eight-variant ablation. Reynolds number and final time are configurable; the defaults are 20 and 1.

Run the velocity–pressure, soft-periodicity, no-causality variant:

```powershell
python -m navier_benchmark.stage2 --reynolds 20 --final-time 1
```

The default uses a 5-layer, 64-unit network, 20,000 Adam steps, up to 1,500 L-BFGS iterations, 1,024 fresh space-time collocation points per Adam step, and float64 precision. RAR is disabled. Outputs are saved in `runs/stage2/<formulation>_<periodicity>_<causality>/Re_<number>` and include the configuration, model weights, metrics, training history, pointwise error maps, and error-versus-time data and plots.

Choose variants with `--formulation velocity_pressure` or `--formulation streamfunction`, `--periodicity soft` or `--periodicity hard`, and optionally `--causality`. Keep the same physical and training settings when comparing variants.

## Stage 3: unsteady 3D Ethier–Steinman (Beltrami) flow

Stage 3 implements the Ethier–Steinman exact solution on `x, y, z ∈ [-1, 1]` and `t ∈ [0, 1]`, with `a = d = 1`.

The network predicts `(u, v, w, p)`. Training uses the 3D momentum and incompressibility residuals, the exact initial velocity, and exact velocity data on all six spatial faces as soft loss terms. The viscosity is fixed for a forward run and is configurable with `--viscosity` (default 0.01). Pressure is used for evaluation, not as a boundary condition.

Run the forward benchmark:

```powershell
python -m navier_benchmark.stage3 --viscosity 0.01
```

Default settings are a 5-layer, 64-unit tanh MLP, up to 20,000 total Adam steps, up to 1,500 L-BFGS iterations, float64 precision, periodic `y` input features, and no RAR. These input features do not impose periodic boundary conditions. Results are written to `runs/stage3/forward` and include model weights, configuration, training history, relative L2 errors, pointwise error maps on the central `z = 0` slice, and error-versus-time data and plots. Pressure errors are reported for completeness; the project’s stated Stage 3 target applies to the velocity components.

Causal time-bin weighting is optional and disabled by default. Enable it with `--causality`. The epsilon schedule defaults to `0.01 0.1 1 10 100`; stages advance when the minimum causal weight reaches the threshold (default 0.99) or the available step budget for that stage is used. `--steps` is the maximum total Adam-step budget across all stages, not a per-stage count. For example, compare causal weighting against the no-periodic-`y` run with a separate output directory:

```powershell
python -m navier_benchmark.stage3 --no-periodic-y-features --causality --steps 20000 --output runs/stage3/forward_no_periodic_y_causal_threshold
```

Use `--causal-time-bins`, `--causal-epsilon-schedule`, and `--causal-weight-threshold` to change the causal-training settings. Use `--eval-resolution` and `--eval-times` to control the dense evaluation grid, and `--plot-time` to select the time used for the error maps.

**Scope:** Stage 3 forward training is implemented. The inverse extension that estimates viscosity from noisy interior velocity observations is not yet implemented.

## Output and reproducibility

Each run stores its configuration and metrics alongside the model checkpoint and training history. Use a distinct `--output` directory for each experiment to preserve prior results. Fix the random seed and keep all other settings unchanged when testing one training change at a time.
