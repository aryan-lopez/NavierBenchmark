# Stage 3: Ethier–Steinman Beltrami flow

## Summary

Stage 3 solves the unsteady three-dimensional Ethier–Steinman (Beltrami) flow on `[-1, 1]^3` over `t in [0, 1]`, with `a = d = 1` and viscosity `0.01`. The velocity–pressure PINN predicts `(u, v, w, p)`. It minimizes the 3D momentum and incompressibility residuals, with exact initial velocity and six-face velocity data included as soft loss terms. The benchmark case and domain match the Beltrami-flow case used in NSFnets.

## Results and comparison

The main saved runs compare periodic-y input features and causal weighting. Errors below are combined relative L2 errors over the saved space-time evaluation grid, which uses 21 spatial points per direction and 11 times.

| Run | Periodic-y features | Causal weighting | `u` error | `v` error | `w` error | Combined velocity error | Time (s) |
| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `forward` | On | No | `1.464e-3` | `1.598e-3` | `1.448e-3` | `1.505e-3` | 2,566.8 |
| `forward_no_periodic_y` | Off | No | `1.205e-3` | `1.262e-3` | `1.179e-3` | `1.216e-3` | 2,606.8 |
| `forward_no_periodic_y_causal` | Off | Yes, threshold 0.99 | `1.203e-3` | `1.189e-3` | `1.216e-3` | `1.203e-3` | 2,514.2 |

Disabling the periodic-y input features improved the combined velocity error by about 19% compared with the default-feature run. Adding corrected causal weighting to the no-periodic-y setup reduced the combined error by about 1.1% relative to its noncausal counterpart. The causal run improved the `u` and `v` metrics slightly, while its `w` error was slightly higher. It did not reach the project target of `1e-3` for any component.

At the final evaluation time `t = 1`, the corrected causal run had component errors `1.484e-3`, `1.477e-3`, and `1.530e-3` for `u`, `v`, and `w`. These are also above the target. The saved output is in `runs/stage3/forward_no_periodic_y_causal/`.

The corrected causal run reached the 0.99 minimum-weight threshold for epsilon values `0.01` and `0.1`. It used its allocated step budget at epsilon values `1`, `10`, and `100` without reaching the threshold. Therefore, the run completed the 20,000-step budget, but the full causal-weight convergence criterion was not met at the larger epsilon values.

## Problems encountered

- The velocity errors remain slightly above the `1e-3` target, including at the final time.
- Periodic-y input features did not help this case; the no-periodic-y baseline performed better.
- Causal weighting provided only a small aggregate improvement over the no-periodic-y baseline, and the individual component changes were mixed.
- An earlier causal-weight implementation incorrectly included initial-condition loss in the causal weights. Its output runs were excluded from this comparison. The current implementation starts the first time-bin weight at 1 and weights later bins using preceding PDE residual losses.
- The 20,000-step run takes about 42 minutes in the recorded environment. Increasing model size or collocation count will raise this cost.

## Relevant comparison with NSFnets

The Stage 3 flow is the three-dimensional unsteady Beltrami benchmark in [NSFnets, Section 3.3](https://arxiv.org/html/2003.06496v1#S3.SS3). NSFnets used a 10-layer, 100-neuron network, 10,000 equation points, `31 x 31` boundary points on each face, and fixed boundary and initial loss weights of 100. This repository currently uses a 5-layer, 64-unit network, 1,024 collocation points, 512 points per face, and sums the PDE, boundary, and initial loss terms without additional weighting coefficients.

The NSFnets VP results at `t = 1` also exceed this project’s `1e-3` component target. Its reported velocity errors are `0.426%`, `0.366%`, and `0.587%`; its VV results are lower but still above `0.1%`. Thus the paper offers useful training settings to test, but does not show that copying its setup alone will meet this stricter target.

## Improvements to investigate

- Add configurable weights for the initial and boundary losses, then compare a paper-inspired weighted setup with the current unit-weight baseline. Align the loss reductions with the paper before directly using its weight value of 100.
- Increase network capacity and collocation density in separate experiments so their effects and compute costs remain distinguishable.
- Keep periodic-y features disabled in the next baseline, since the saved comparison shows lower error without them.
- Run multiple random seeds for the best configuration and report both combined and componentwise errors.
- Increase evaluation resolution for final reporting to confirm the result on a denser space-time grid.

## Saved results used

- `runs/stage3/forward/` — periodic-y features enabled, no causality.
- `runs/stage3/forward_no_periodic_y/` — periodic-y features disabled, no causality.
- `runs/stage3/forward_no_periodic_y_causal/` — periodic-y features disabled, corrected causal weighting.
