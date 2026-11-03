"""Training and evaluation for individual Taylor–Green Stage 2 variants."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import Tensor

from .taylor_green import TaylorGreenVortex
from .taylor_green_model import TaylorGreenPINN, Formulation, Periodicity, periodic_boundary_loss
from .unsteady import unsteady_navier_stokes_residual


@dataclass(frozen=True)
class TaylorGreenTrainingConfig:
    reynolds: float = 20.0
    final_time: float = 1.0
    formulation: Formulation = "velocity_pressure"
    periodicity: Periodicity = "soft"
    steps: int = 20_000
    collocation_points: int = 1_024
    initial_points: int = 256
    periodic_pairs: int = 256
    hidden_layers: int = 5
    width: int = 64
    learning_rate: float = 1.0e-3
    final_learning_rate: float = 1.0e-4
    lbfgs_steps: int = 1_500
    periodic_features: bool = True
    causality: bool = False
    causal_epsilon: float = 1.0
    causal_time_bins: int = 10
    lambda_initial: float = 1.0
    lambda_periodic: float = 1.0
    evaluation_spatial_resolution: int = 101
    evaluation_time_points: int = 21
    plot_time: float | None = None
    seed: int = 1234
    device: str = "auto"
    dtype: str = "float64"
    log_every: int = 100

    def __post_init__(self) -> None:
        if self.reynolds <= 0 or self.final_time <= 0:
            raise ValueError("reynolds and final_time must be positive")
        if self.steps < 1 or self.lbfgs_steps < 0:
            raise ValueError("steps must be positive and lbfgs_steps nonnegative")
        if min(self.collocation_points, self.initial_points, self.periodic_pairs) < 1:
            raise ValueError("all training sample counts must be positive")
        if min(self.hidden_layers, self.width, self.evaluation_spatial_resolution,
               self.evaluation_time_points, self.log_every) < 1:
            raise ValueError("network, evaluation, and logging sizes must be positive")
        if self.learning_rate <= 0 or self.final_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.lambda_initial < 0 or self.lambda_periodic < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.causal_epsilon < 0 or self.causal_time_bins < 1:
            raise ValueError("causal_epsilon must be nonnegative and causal_time_bins positive")
        if self.formulation not in ("velocity_pressure", "streamfunction"):
            raise ValueError("unknown formulation")
        if self.periodicity not in ("soft", "hard"):
            raise ValueError("unknown periodicity")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(name)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return selected


def _relative_l2(pred: Tensor, exact: Tensor) -> float:
    return float(torch.linalg.vector_norm(pred - exact) /
                 torch.linalg.vector_norm(exact).clamp_min(torch.finfo(exact.dtype).eps))


def _write_error_maps(path: Path, x: Tensor, y: Tensor, errors: dict[str, Tensor], *, time_value: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    for axis, key, label in zip(axes, ("u", "v", "p"), ("|u error|", "|v error|", "|p error|")):
        matrix = errors[key].detach().cpu().numpy()
        image = axis.imshow(matrix, origin="lower", extent=extent, aspect="equal", cmap="magma")
        axis.set_title(f"{label} (max {matrix.max():.2e})")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"Pointwise absolute errors at t={time_value:.4g}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_line_plot(path: Path, x_values: list[float], series: list[tuple[str, list[float], str]],
                     *, title: str, x_label: str, y_label: str) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for name, values, color in series:
        axis.semilogy(x_values, [max(v, 1e-16) for v in values], label=name, color=color, linewidth=2)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _evaluate(model: TaylorGreenPINN, flow: TaylorGreenVortex, config: TaylorGreenTrainingConfig,
              device: torch.device, dtype: torch.dtype) -> tuple[dict[str, Any], dict[str, Any]]:
    n = config.evaluation_spatial_resolution
    x = torch.linspace(0.0, float(flow.period), n, device=device, dtype=dtype)
    y = torch.linspace(0.0, float(flow.period), n, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    spatial = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    times = torch.linspace(0.0, flow.final_time, config.evaluation_time_points, device=device, dtype=dtype)
    rows: dict[str, list[float]] = {"u": [], "v": [], "p": []}
    final_map: dict[str, Tensor] | None = None
    map_time = config.plot_time if config.plot_time is not None else flow.final_time
    map_time = min(max(map_time, 0.0), flow.final_time)
    pressure_num = torch.zeros((), device=device, dtype=dtype)
    pressure_den = torch.zeros((), device=device, dtype=dtype)
    aggregate_sq = {"u": 0.0, "v": 0.0}
    aggregate_den = {"u": 0.0, "v": 0.0}
    p_pred_values, p_exact_values = [], []
    with torch.enable_grad():
        for time_value in times:
            pts = torch.cat((spatial, torch.full((len(spatial), 1), float(time_value), device=device, dtype=dtype)), dim=1)
            predicted_parts = []
            for chunk in pts.split(8192):
                predicted_parts.append(model(chunk).detach())
            predicted = torch.cat(predicted_parts)
            exact = flow.exact(pts)
            pred_u, pred_v = predicted[:, 0], predicted[:, 1]
            exact_u, exact_v = exact[:, 0], exact[:, 1]
            rows["u"].append(_relative_l2(pred_u, exact_u))
            rows["v"].append(_relative_l2(pred_v, exact_v))
            for i, key in enumerate(("u", "v")):
                aggregate_sq[key] += float((predicted[:, i] - exact[:, i]).square().sum())
                aggregate_den[key] += float(exact[:, i].square().sum())
            p_pred_values.append(predicted[:, 2])
            p_exact_values.append(exact[:, 2])
            # Align the arbitrary pressure gauge independently at each frame.
            pred_p = predicted[:, 2] - predicted[:, 2].mean()
            exact_p = exact[:, 2] - exact[:, 2].mean()
            rows["p"].append(_relative_l2(pred_p, exact_p))
            pressure_num += (pred_p - exact_p).square().sum()
            pressure_den += exact_p.square().sum()
            if abs(float(time_value) - map_time) <= flow.final_time / max(1, config.evaluation_time_points - 1) / 2 + 1e-12:
                final_map = {
                    "u": (pred_u - exact_u).abs().reshape(n, n),
                    "v": (pred_v - exact_v).abs().reshape(n, n),
                    "p": (pred_p - exact_p).abs().reshape(n, n),
                }
                map_time = float(time_value)
    # Aggregate over the complete fixed evaluation grid and align pressure's arbitrary constant.
    p_pred = torch.cat(p_pred_values)
    p_exact = torch.cat(p_exact_values)
    p_pred -= p_pred.mean()
    p_exact -= p_exact.mean()
    aggregate_sq["p"] = float((p_pred - p_exact).square().sum())
    aggregate_den["p"] = float(p_exact.square().sum())
    errors = {key: math.sqrt(aggregate_sq[key] / max(aggregate_den[key], 1e-30)) for key in aggregate_sq}
    errors["velocity"] = math.sqrt(sum(aggregate_sq[k] for k in ("u", "v")) /
                                   max(sum(aggregate_den[k] for k in ("u", "v")), 1e-30))
    # Use the per-frame alignment convention in the saved pressure metric.
    errors["p"] = float(torch.sqrt(pressure_num / pressure_den.clamp_min(1e-30)))
    errors["grid_points"] = int(n * n * len(times))
    errors["evaluation_time_points"] = int(len(times))
    curves = {"time": [float(t) for t in times], **rows}
    return errors, {"curves": curves, "map": final_map, "x": x.cpu(), "y": y.cpu(), "map_time": map_time}


def train_taylor_green(config: TaylorGreenTrainingConfig, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = _device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(1)
    generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    flow = TaylorGreenVortex(config.reynolds, config.final_time)
    model = TaylorGreenPINN(flow, config.formulation, config.periodicity,
                            config.hidden_layers, config.width,
                            periodic_features=config.periodic_features).to(device=device, dtype=dtype)
    metadata = {**asdict(config), "device_resolved": str(device), "viscosity": flow.viscosity,
                "periodic_feature_description": (
                    "raw x/y/t plus sin/cos x/y; boundary periodicity remains a soft loss"
                    if config.periodicity == "soft" and config.periodic_features else
                    "sin/cos x/y feature map; periodicity is hard by construction"
                    if config.periodicity == "hard" else "disabled"),
                "rar": False,
                "loss": ("Navier-Stokes residual + initial velocity + soft periodicity penalty"
                        if config.periodicity == "soft" else "Navier-Stokes residual + initial velocity"),
                "causal_weighting": ("w_k=exp(-epsilon*sum(previous time-bin residual losses)); weights detached"
                                     if config.causality else "disabled"),
                "pressure_gauge": "spatial mean aligned per evaluation time slice"}
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    initial_points, initial_velocity = flow.sample_initial_conditions(
        config.initial_points, device=device, dtype=dtype, generator=generator)
    periodic_pairs = flow.sample_periodic_pairs(
        config.periodic_pairs, device=device, dtype=dtype, generator=generator)
    adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    gamma = (config.final_learning_rate / config.learning_rate) ** (1.0 / config.steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=gamma)
    history: list[dict[str, float]] = []
    training_start = time.perf_counter()

    def compute_loss(collocation: Tensor) -> tuple[Tensor, dict[str, float]]:
        residuals = unsteady_navier_stokes_residual(model, collocation, flow.viscosity)
        if config.causality:
            per_point = sum(term.square() for term in residuals).squeeze(1)
            bins = torch.floor(collocation[:, 2] / flow.final_time * config.causal_time_bins).long()
            bins = bins.clamp_(0, config.causal_time_bins - 1)
            bin_losses = []
            for bin_index in range(config.causal_time_bins):
                selected = per_point[bins == bin_index]
                bin_losses.append(selected.mean() if selected.numel() else per_point.sum() * 0.0)
            bin_losses_tensor = torch.stack(bin_losses)
            prior_cumulative = torch.cat((bin_losses_tensor.new_zeros(1),
                                          torch.cumsum(bin_losses_tensor.detach(), dim=0)[:-1]))
            causal_weights = torch.exp(-config.causal_epsilon * prior_cumulative)
            residual_loss = (causal_weights * bin_losses_tensor).sum() / config.causal_time_bins
            causal_first, causal_last = float(causal_weights[0]), float(causal_weights[-1])
        else:
            residual_loss = sum(term.square().mean() for term in residuals)
            causal_first = causal_last = 1.0
        initial_loss = (model(initial_points)[:, :2] - initial_velocity).square().mean()
        periodic_loss = periodic_boundary_loss(model, periodic_pairs)
        total = residual_loss + config.lambda_initial * initial_loss + config.lambda_periodic * periodic_loss
        return total, {"total": float(total.detach()), "residual": float(residual_loss.detach()),
                       "initial": float(initial_loss.detach()), "periodic": float(periodic_loss.detach()),
                       "causal_weight_first": causal_first, "causal_weight_last": causal_last,
                       "learning_rate": float(adam.param_groups[0]["lr"])}

    for step in range(1, config.steps + 1):
        collocation = flow.sample_collocation(config.collocation_points, device=device,
                                              dtype=dtype, generator=generator)
        adam.zero_grad(set_to_none=True)
        loss, terms = compute_loss(collocation)
        loss.backward()
        adam.step()
        scheduler.step()
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            terms["step"] = float(step)
            history.append(terms)
            print(f"step {step}/{config.steps}: loss={terms['total']:.4e} "
                  f"residual={terms['residual']:.3e} initial={terms['initial']:.3e} "
                  f"periodic={terms['periodic']:.3e}", flush=True)

    if config.lbfgs_steps:
        lbfgs_points = flow.sample_collocation(config.collocation_points, device=device,
                                               dtype=dtype, generator=generator)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=config.lbfgs_steps,
                                      history_size=50, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-10 if dtype == torch.float64 else 1e-7,
                                      tolerance_change=1e-12 if dtype == torch.float64 else 1e-9)
        closure_calls = 0

        def closure() -> Tensor:
            nonlocal closure_calls
            optimizer.zero_grad(set_to_none=True)
            current, terms = compute_loss(lbfgs_points)
            current.backward()
            closure_calls += 1
            if closure_calls % 25 == 0:
                print(f"L-BFGS closure {closure_calls}: loss={terms['total']:.4e}", flush=True)
            return current

        optimizer.step(closure)
        with torch.enable_grad():
            final_loss, final_terms = compute_loss(lbfgs_points)
        final_terms["step"] = float(config.steps + closure_calls)
        history.append(final_terms)
        print(f"L-BFGS finished after {closure_calls} closure calls: loss={final_terms['total']:.4e}", flush=True)

    training_seconds = time.perf_counter() - training_start
    errors, evaluation = _evaluate(model, flow, config, device, dtype)
    torch.save({"state_dict": model.state_dict(), "config": metadata}, output / "model.pt")
    metrics = {"variant": f"{config.formulation}_{config.periodicity}_{'causal' if config.causality else 'no_causality'}",
               "reynolds": config.reynolds, "final_time": config.final_time,
               "errors": errors, "training_seconds": training_seconds,
               "training_steps_adam": config.steps, "lbfgs_steps_requested": config.lbfgs_steps,
               "configuration": metadata}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "total", "residual", "initial", "periodic",
                                                    "causal_weight_first", "causal_weight_last", "learning_rate"))
        writer.writeheader()
        writer.writerows(history)

    curves = evaluation["curves"]
    _write_line_plot(output / "error_vs_time.png", curves["time"],
                   [(key, curves[key], color) for key, color in (("u", "#1677b8"), ("v", "#dc5945"), ("p", "#2c9955"))],
                   title="Taylor–Green relative L2 error versus time", x_label="time", y_label="relative L2 error")
    if evaluation["map"] is not None:
        _write_error_maps(output / "pointwise_error_maps.png", evaluation["x"], evaluation["y"],
                          evaluation["map"], time_value=evaluation["map_time"])
    _write_line_plot(output / "training_loss.png", [row["step"] for row in history],
                   [(key, [row[key] for row in history], color) for key, color in
                    (("total", "#222222"), ("residual", "#1677b8"), ("initial", "#dc5945"), ("periodic", "#2c9955"))],
                   title="Training loss history", x_label="optimizer step / closure calls", y_label="loss")
    with (output / "error_vs_time.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("time", "u_relative_l2", "v_relative_l2", "p_relative_l2"))
        writer.writeheader()
        writer.writerows({"time": curves["time"][i], "u_relative_l2": curves["u"][i],
                          "v_relative_l2": curves["v"][i], "p_relative_l2": curves["p"][i]}
                         for i in range(len(curves["time"])))
    return metrics
