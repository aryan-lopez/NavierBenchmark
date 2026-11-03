# Stage 2 Taylor–Green ablation results

All cases use Re=20, T=1, a 5×64 network, 20,000 Adam steps, up to 1,500 L-BFGS iterations, 1,024 space-time collocation points per Adam step, float64 precision, and a 101×101 grid at 21 times. Relative pressure error aligns pressure means independently at each evaluation frame. RAR is disabled.

| Variant | Formulation | Periodicity | Causality | Rel. L2 u | Rel. L2 v | Rel. L2 p | Rel. L2 velocity | Training (s) |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | Velocity–pressure | Soft | No | 0.000661257 | 0.000673069 | 0.00338916 | 0.000667189 | 2416.6 |
| 2 | Velocity–pressure | Hard | No | 0.000556388 | 0.000644401 | 0.00527527 | 0.000602005 | 2138.1 |
| 3 | Velocity–pressure | Hard | Yes | 0.000564023 | 0.000576622 | 0.00519322 | 0.000570357 | 2264.3 |
| 4 | Streamfunction–pressure | Soft | No | 0.000247796 | 0.000250287 | 0.000759849 | 0.000249044 | 5653.0 |
| 5 | Streamfunction–pressure | Hard | Yes | 0.000216982 | 0.000218932 | 0.000633138 | 0.000217959 | 5152.8 |

Per-run directories contain config, metrics, history, checkpoints, and individual plots.
