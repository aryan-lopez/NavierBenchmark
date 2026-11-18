"""Train the Stage 3 3D Ethier--Steinman forward PINN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ethier_steinman_solver import EthierSteinmanTrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Stage 3 Ethier--Steinman forward PINN.")
    parser.add_argument("--viscosity", type=float, default=0.01)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--collocation", type=int, default=1_024)
    parser.add_argument("--initial-points", type=int, default=512)
    parser.add_argument("--boundary-points-per-face", type=int, default=512)
    parser.add_argument("--rar-every", type=int, default=0,
                        help="RAR interval in Adam steps; 0 disables RAR (default)")
    parser.add_argument("--rar-candidates", type=int, default=4_096)
    parser.add_argument("--rar-points", type=int, default=256)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--lr-decay", type=float, default=0.9999)
    parser.add_argument("--lbfgs-steps", type=int, default=1_500)
    parser.add_argument("--periodic-y-features", action=argparse.BooleanOptionalAction,
                        default=True, help="Append sin/cos features of y (default: enabled)")
    parser.add_argument("--eval-resolution", type=int, default=21)
    parser.add_argument("--eval-times", type=int, default=11)
    parser.add_argument("--plot-time", type=float, default=None,
                        help="Time for pointwise error maps; defaults to final time")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=Path("runs/stage3/forward"))
    args = parser.parse_args()
    config = EthierSteinmanTrainingConfig(
        viscosity=args.viscosity, final_time=args.final_time,
        steps=args.steps, collocation_points=args.collocation,
        initial_points=args.initial_points,
        boundary_points_per_face=args.boundary_points_per_face,
        rar_every=args.rar_every, rar_candidates=args.rar_candidates,
        rar_points=args.rar_points, hidden_layers=args.layers, width=args.width,
        learning_rate=args.learning_rate, lr_decay=args.lr_decay,
        lbfgs_steps=args.lbfgs_steps, periodic_y_features=args.periodic_y_features,
        evaluation_resolution=args.eval_resolution,
        evaluation_times=args.eval_times, seed=args.seed, device=args.device,
        plot_time=args.plot_time, dtype=args.dtype,
    )
    result = train(config, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
