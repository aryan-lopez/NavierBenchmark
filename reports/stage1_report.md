# Stage 1: Kovasznay flow

## Summary

Stage 1 approximates steady two-dimensional Kovasznay flow at Reynolds numbers 20 and 40. The PINN predicts velocity `(u, v)` and kinematic pressure `p`. It uses a five-hidden-layer, 64-unit tanh network. Velocity boundary values are imposed exactly through a Coons-patch lift and a boundary-vanishing correction; the training objective minimizes the steady momentum and incompressibility residuals.

## Problems encountered

The first saved runs had relatively high errors, particularly for the vertical velocity `v`. With 20,000 Adam steps and 500 L-BFGS iterations, the combined velocity relative L2 errors were `1.650e-3` at Re=20 and `1.736e-3` at Re=40. The `v` error at Re=40 was `9.284e-3`.

Increasing L-BFGS from 500 to 1,500 iterations reduced the combined velocity errors to `1.153e-4` and `1.141e-4` for Re=20 and Re=40. This showed that the refinement stage was important for accuracy.

The strongest remaining componentwise error is `v`. In the best saved no-RAR run, its errors were `2.264e-4` at Re=20 and `2.107e-4` at Re=40. Thus, while the combined velocity error is below `1e-4` in that run, the `1e-4` threshold is not met by each velocity component individually.

## Effect of periodic-y input features

Periodic-y features append `sin(2πy)` and `cos(2πy)` to the model inputs. They do not impose a periodic boundary condition. Comparing the saved no-RAR runs with 1,500 L-BFGS iterations shows a clear improvement when these features are enabled:

| Re | Combined velocity error without features | With periodic-y features | Reduction |
| ---: | ---: | ---: | ---: |
| 20 | `1.153e-4` | `5.949e-5` | 48.4% |
| 40 | `1.141e-4` | `3.734e-5` | 67.3% |

The `v` component also improved, from `3.610e-4` to `2.264e-4` at Re=20, and from `5.399e-4` to `2.107e-4` at Re=40. Periodic-y features are therefore retained in the Stage 1 defaults.

## RAR results

RAR did not provide a consistent improvement over the periodic-y, no-RAR run. Retaining 64 high-residual points every 5,000 Adam steps changed the combined velocity error from `5.949e-5` to `9.072e-5` at Re=20, while changing it from `3.734e-5` to `3.707e-5` at Re=40. Because the result worsened for one case and barely changed for the other, RAR remains disabled by default.

## Improvements to investigate

- Use the saved `periodic_y` run as the comparison baseline: periodic-y features enabled, 20,000 Adam steps, 1,500 L-BFGS iterations, and RAR disabled.
- Focus the next controlled experiment on reducing the `v` error, for example by testing loss balancing or sampling strategies while keeping the architecture and optimizer fixed.
- Avoid enabling RAR by default unless a matched experiment shows consistent improvement for both Reynolds numbers.
- Repeat the best configuration with multiple random seeds to determine whether the low errors are reproducible.
- Report both combined velocity error and individual `u` and `v` errors, since the combined metric alone hides the componentwise `v` gap.

## Saved results used

The comparisons above use the following run folders:

- `runs/stage1/initial/` — 500 L-BFGS iterations, no periodic-y features.
- `runs/stage1/lbfgs1500/` — 1,500 L-BFGS iterations, no periodic-y features.
- `runs/stage1/periodic_y/` — 1,500 L-BFGS iterations, periodic-y features, RAR disabled.
- `runs/stage1/periodic_y_rar64/` — 1,500 L-BFGS iterations, periodic-y features, 64 RAR points every 5,000 Adam steps.

All values are measured from saved run metrics and should be treated as results for those particular runs, not guaranteed performance.
