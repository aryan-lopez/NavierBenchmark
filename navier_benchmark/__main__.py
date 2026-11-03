"""Command line entry point for Stage 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .solver import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Stage 1 Kovasznay PINN.")
    parser.add_argument("--reynolds", nargs="+", type=float, default=(20.0, 40.0), help="Reynolds number(s), default: 20 40")
    parser.add_argument("--output", type=Path, default=Path("runs/stage1"), help="Output directory")
    parser.add_argument("--steps", type=int, default=20_000, help="Adam iterations")
    parser.add_argument("--collocation", type=int, default=1024, help="Fresh uniform interior points per Adam step")
    parser.add_argument("--rar-every", type=int, default=5000, help="Add high-residual points every N Adam steps; 0 disables RAR")
    parser.add_argument("--rar-candidates", type=int, default=8192, help="Uniform candidates scored at each RAR step")
    parser.add_argument("--rar-points", type=int, default=256, help="Highest-residual candidates retained per RAR step")
    parser.add_argument("--layers", type=int, default=5, help="Number of hidden layers")
    parser.add_argument("--width", type=int, default=64, help="Hidden units per layer")
    parser.add_argument("--periodic-y-features", action=argparse.BooleanOptionalAction, default=True,
                        help="Include sin(2πy), cos(2πy) input features (default: enabled)")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lbfgs-steps", type=int, default=500, help="L-BFGS iterations after Adam; 0 disables")
    parser.add_argument("--test-resolution", type=int, default=201, help="Points per side of the independent evaluation grid")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()

    summaries = []
    for index, reynolds in enumerate(args.reynolds):
        config = TrainingConfig(reynolds=reynolds, steps=args.steps,
                                collocation_points=args.collocation,
                                rar_every=args.rar_every, rar_candidates=args.rar_candidates,
                                rar_points=args.rar_points,
                                hidden_layers=args.layers, width=args.width,
                                periodic_y_features=args.periodic_y_features,
                                learning_rate=args.learning_rate, lbfgs_steps=args.lbfgs_steps,
                                test_resolution=args.test_resolution, seed=args.seed + index,
                                device=args.device, dtype=args.dtype)
        result = train(config, args.output / f"Re_{reynolds:g}")
        summaries.append({"reynolds": reynolds, "errors": result["errors"],
                          "training_seconds": result["training_seconds"]})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
