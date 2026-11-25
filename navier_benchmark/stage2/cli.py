"""Run a single configurable Stage 2 Taylor–Green PINN variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .taylor_green_solver import TaylorGreenTrainingConfig, train_taylor_green


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Stage 2 Taylor–Green PINN variant.")
    parser.add_argument("--reynolds", type=float, default=20.0)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--formulation", choices=("velocity_pressure", "streamfunction"), default="velocity_pressure")
    parser.add_argument("--periodicity", choices=("soft", "hard"), default="soft")
    parser.add_argument("--steps", type=int, default=20_000, help="Adam iterations")
    parser.add_argument("--collocation", type=int, default=1_024, help="Fresh space-time residual points per Adam iteration")
    parser.add_argument("--initial-points", type=int, default=256)
    parser.add_argument("--periodic-pairs", type=int, default=256)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--final-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--lbfgs-steps", type=int, default=1_500)
    parser.add_argument("--causality", action="store_true", help="Enable causal weighting across time bins")
    parser.add_argument("--causal-epsilon", type=float, default=1.0)
    parser.add_argument("--causal-time-bins", type=int, default=10)
    parser.add_argument("--periodic-features", action=argparse.BooleanOptionalAction, default=True,
                        help="Append sin/cos features for x and y while retaining raw coordinates")
    parser.add_argument("--eval-resolution", type=int, default=101, help="Spatial grid points per direction")
    parser.add_argument("--eval-times", type=int, default=21)
    parser.add_argument("--plot-time", type=float, default=None, help="Time for pointwise maps; defaults to final time")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    variant = f"{args.formulation}_{args.periodicity}_{'causal' if args.causality else 'no_causality'}"
    output = args.output or Path("runs/stage2") / variant / f"Re_{args.reynolds:g}"
    config = TaylorGreenTrainingConfig(
        reynolds=args.reynolds, final_time=args.final_time,
        formulation=args.formulation, periodicity=args.periodicity,
        steps=args.steps, collocation_points=args.collocation,
        initial_points=args.initial_points, periodic_pairs=args.periodic_pairs,
        hidden_layers=args.layers, width=args.width,
        learning_rate=args.learning_rate, final_learning_rate=args.final_learning_rate,
        lbfgs_steps=args.lbfgs_steps, periodic_features=args.periodic_features,
        causality=args.causality, causal_epsilon=args.causal_epsilon,
        causal_time_bins=args.causal_time_bins,
        evaluation_spatial_resolution=args.eval_resolution,
        evaluation_time_points=args.eval_times, plot_time=args.plot_time,
        seed=args.seed, device=args.device, dtype=args.dtype,
    )
    result = train_taylor_green(config, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
