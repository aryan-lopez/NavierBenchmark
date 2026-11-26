# Stage 2: Taylor–Green vortex

## Summary

Stage 2 solves the unsteady two-dimensional Taylor–Green vortex at Re=20 over the time interval `t = 0` to `t = 1`. The experiments compare two network formulations (velocity–pressure and streamfunction–pressure), two periodicity methods (soft and hard), and causal weighting on or off. This creates eight configurations. All saved runs used the same physical case, 5-layer, 64-unit network, 20,000 Adam steps, up to 1,500 L-BFGS iterations, 1,024 collocation points per Adam step, float64 precision, and a 101×101 spatial evaluation grid at 21 times.

## Comparison challenge and method

With eight runs, there was no single obvious “best” result: some configurations had lower velocity error but took much longer to train. Ranking them by accuracy alone would ignore computational cost, while ranking by time alone would ignore solution quality.

Therefore a Pareto frontier was used with two objectives: minimize combined relative L2 velocity error and training time. A run is Pareto-efficient when no other run is both faster and more accurate. Pressure error is reported for context, but is not part of this comparison because the Stage 2 objective is velocity accuracy. Pressure means are aligned independently at each evaluation frame because pressure is defined up to an additive constant.

## Results

| Variant | Formulation | Periodicity | Causality | Velocity relative L2 | Training time (s) | Pareto-efficient |
| ---: | --- | --- | --- | ---: | ---: | :---: |
| 1 | Velocity–pressure | Soft | No | `6.672e-4` | 2,416.6 | No |
| 2 | Velocity–pressure | Hard | No | `6.020e-4` | 2,138.1 | Yes |
| 3 | Velocity–pressure | Hard | Yes | `5.704e-4` | 2,264.3 | Yes |
| 4 | Streamfunction–pressure | Soft | No | `2.490e-4` | 5,653.0 | No |
| 5 | Streamfunction–pressure | Hard | Yes | `2.180e-4` | 5,152.8 | Yes |
| 6 | Velocity–pressure | Soft | Yes | `7.199e-4` | 2,426.0 | No |
| 7 | Streamfunction–pressure | Soft | Yes | `2.572e-4` | 5,688.2 | No |
| 8 | Streamfunction–pressure | Hard | No | `2.632e-4` | 5,263.8 | No |

The frontier exposes three useful choices. Variant 2 is the fastest Pareto-efficient option. Variant 3 spends about 126 seconds more and modestly lowers velocity error. Variant 5 has the lowest velocity error, `2.180e-4`, at substantially higher cost—about 2.4 times the training time of variant 2. The frontier makes this tradeoff visible without declaring one configuration best for every use.

Across these runs, the streamfunction formulation produced lower velocity errors than the velocity–pressure formulation, but required more than twice as much training time. For velocity–pressure, hard periodicity improved both error and runtime compared with soft periodicity in the no-causality pair. Causality slightly improved the hard-periodic velocity–pressure result; its effect was not consistent across all formulations and periodicity settings.

All eight combined velocity errors are below `1e-3`. The lowest saved pressure error is `6.331e-4` for Variant 5; velocity, rather than pressure, is the objective used to form the frontier.

## Problems and limitations

- The results show an accuracy–runtime tradeoff, so reporting one “winner” would hide the reason to select another configuration.
- Streamfunction runs took approximately 5,153–5,688 seconds, compared with approximately 2,138–2,426 seconds for velocity–pressure runs.
- Causal weighting did not improve every configuration, so it should be treated as an ablation choice rather than a universally beneficial setting.
- Each variant has one saved run. Without repeated seeds, small differences—especially training-time differences—may reflect initialization or runtime variation.

## Improvements to investigate

- Repeat the eight configurations with multiple seeds and compare mean and spread of both velocity error and training time.
- Select a Pareto-efficient configuration according to the project’s priority: speed (Variant 2), a modest accuracy improvement (Variant 3), or lowest observed error (Variant 5).
- Keep the evaluation grid, training budget, and physical parameters fixed when adding further ablations.
- Retain pressure metrics in the run artifacts for analysis, while keeping the velocity-based target and frontier definition explicit.

## Saved results used

The run metrics and ablation artifacts are under `runs/stage2/`. The generated summary is `runs/stage2/ablation_table.md`; the nondominated runs are listed in `runs/stage2/pareto_frontier.csv` and plotted in `runs/stage2/pareto_frontier.png`.
