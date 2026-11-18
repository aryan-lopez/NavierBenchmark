"""Stage 3 forward training using the Stage 1 Adam/L-BFGS workflow."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import Tensor

from .ethier_steinman import EthierSteinmanFlow
from .ethier_steinman_model import EthierSteinmanPINN
from .unsteady import unsteady_navier_stokes_3d_residual


@dataclass(frozen=True)
class EthierSteinmanTrainingConfig:
    viscosity: float = 0.01
    final_time: float = 1.0
    steps: int = 20_000
    collocation_points: int = 1_024
    initial_points: int = 512
    boundary_points_per_face: int = 512
    rar_every: int = 0
    rar_candidates: int = 4_096
    rar_points: int = 256
    rar_chunk_size: int = 256
    hidden_layers: int = 5
    width: int = 64
    learning_rate: float = 1.0e-3
    lr_decay: float = 0.9999
    lbfgs_steps: int = 1_500
    periodic_y_features: bool = True
    evaluation_resolution: int = 21
    evaluation_times: int = 11
    plot_time: float | None = None
    seed: int = 1234
    device: str = "auto"
    dtype: str = "float64"
    log_every: int = 500

    def __post_init__(self) -> None:
        if self.viscosity <= 0 or self.final_time <= 0:
            raise ValueError("viscosity and final_time must be positive")
        if min(self.steps, self.collocation_points, self.initial_points,
               self.boundary_points_per_face) < 1:
            raise ValueError("training steps and sample counts must be positive")
        if self.lbfgs_steps < 0 or self.rar_every < 0 or self.rar_points < 0:
            raise ValueError("L-BFGS and RAR settings must be nonnegative")
        if self.rar_candidates < 1 or self.rar_chunk_size < 1:
            raise ValueError("RAR candidate and chunk sizes must be positive")
        if self.evaluation_resolution < 2 or self.evaluation_times < 1:
            raise ValueError("evaluation grid sizes are invalid")
        if self.plot_time is not None and not 0.0 <= self.plot_time <= self.final_time:
            raise ValueError("plot_time must be within [0, final_time]")
        if self.hidden_layers < 1 or self.width < 1 or self.log_every < 1:
            raise ValueError("network sizes and log_every must be positive")
        if self.learning_rate <= 0 or not 0 < self.lr_decay <= 1:
            raise ValueError("learning_rate must be positive and lr_decay in (0, 1]")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _losses(model: EthierSteinmanPINN, flow: EthierSteinmanFlow,
            interior: Tensor, initial_points: Tensor, initial_velocity: Tensor,
            boundary_points: Tensor, boundary_velocity: Tensor
            ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    r_u, r_v, r_w, r_c = unsteady_navier_stokes_3d_residual(
        model, interior, flow.viscosity)
    momentum = (r_u.square() + r_v.square() + r_w.square()).mean()
    continuity = r_c.square().mean()
    initial = (model(initial_points)[:, :3] - initial_velocity).square().mean()
    boundary = (model(boundary_points)[:, :3] - boundary_velocity).square().mean()
    return momentum + continuity + initial + boundary, momentum, continuity, initial, boundary


def _residual_scores(model: EthierSteinmanPINN, flow: EthierSteinmanFlow,
                     candidates: Tensor, chunk_size: int) -> Tensor:
    scores: list[Tensor] = []
    for chunk in candidates.split(chunk_size):
        residuals = unsteady_navier_stokes_3d_residual(model, chunk, flow.viscosity)
        scores.append(torch.sqrt(sum(value.detach().square() for value in residuals[:3])).squeeze(1))
    return torch.cat(scores)


def _relative_l2(pred: np.ndarray, exact: np.ndarray) -> float:
    denominator = np.linalg.norm(exact.ravel())
    return float(np.linalg.norm((pred - exact).ravel()) / max(denominator, np.finfo(float).tiny))


@torch.no_grad()
def evaluate(model: EthierSteinmanPINN, flow: EthierSteinmanFlow,
             resolution: int, time_points: int, device: torch.device,
             dtype: torch.dtype, plot_time: float | None = None,
             chunk_size: int = 16_384
             ) -> tuple[dict[str, Any], list[dict[str, float]], dict[str, np.ndarray]]:
    """Evaluate dense errors, per-time errors, and a central-z error-map slice."""
    axis = torch.linspace(-1.0, 1.0, resolution, device=device, dtype=dtype)
    times = torch.linspace(0.0, flow.final_time, time_points, device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
    xyz = torch.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), dim=1)
    predicted_frames: list[np.ndarray] = []
    exact_frames: list[np.ndarray] = []
    for t in times:
        points = torch.cat((xyz, torch.full((len(xyz), 1), float(t), device=device, dtype=dtype)), dim=1)
        predicted_parts: list[Tensor] = []
        exact_parts: list[Tensor] = []
        for block in points.split(chunk_size):
            predicted_parts.append(model(block).cpu())
            exact_parts.append(flow.exact(block).cpu())
        predicted_frames.append(torch.cat(predicted_parts).numpy())
        exact_frames.append(torch.cat(exact_parts).numpy())
    pred = np.concatenate(predicted_frames)
    exact = np.concatenate(exact_frames)
    errors = {name: _relative_l2(pred[:, index], exact[:, index])
              for index, name in enumerate(("u", "v", "w"))}
    pred_p = pred[:, 3] - pred[:, 3].mean()
    exact_p = exact[:, 3] - exact[:, 3].mean()
    errors["p"] = _relative_l2(pred_p, exact_p)
    errors["velocity"] = _relative_l2(pred[:, :3], exact[:, :3])
    errors["grid_points"] = int(len(pred))
    errors["spatial_resolution"] = resolution
    errors["time_points"] = time_points

    time_errors: list[dict[str, float]] = []
    for t, frame_pred, frame_exact in zip(times.cpu().tolist(), predicted_frames, exact_frames):
        frame_row = {"time": float(t)}
        for index, name in enumerate(("u", "v", "w")):
            frame_row[name] = _relative_l2(frame_pred[:, index], frame_exact[:, index])
        frame_p = frame_pred[:, 3] - frame_pred[:, 3].mean()
        exact_p = frame_exact[:, 3] - frame_exact[:, 3].mean()
        frame_row["p"] = _relative_l2(frame_p, exact_p)
        frame_row["velocity"] = _relative_l2(frame_pred[:, :3], frame_exact[:, :3])
        time_errors.append(frame_row)

    selected_index = min(range(len(times)), key=lambda index:
                         abs(float(times[index]) - (flow.final_time if plot_time is None else plot_time)))
    slice_index = int(np.argmin(np.abs(axis.cpu().numpy())))
    selected_pred = predicted_frames[selected_index]
    selected_exact = exact_frames[selected_index]
    abs_errors = np.abs(selected_pred - selected_exact)
    # Align pressure means because pressure is defined only up to a constant.
    pressure_error = ((selected_pred[:, 3] - selected_pred[:, 3].mean())
                      - (selected_exact[:, 3] - selected_exact[:, 3].mean()))
    abs_errors[:, 3] = np.abs(pressure_error)
    error_maps = {name: abs_errors[:, index].reshape(resolution, resolution, resolution)[:, :, slice_index]
                  for index, name in enumerate(("u", "v", "w", "p"))}
    error_maps["time"] = np.asarray(float(times[selected_index]))
    error_maps["z"] = np.asarray(float(axis[slice_index]))
    return errors, time_errors, error_maps


def _write_error_maps(path: Path, maps: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    extent = (-1.0, 1.0, -1.0, 1.0)
    for axis, (name, label) in zip(axes.flat, (("u", "|u error|"), ("v", "|v error|"),
                                               ("w", "|w error|"), ("p", "|p error|"))):
        values = maps[name]
        image = axis.imshow(values.T, origin="lower", extent=extent, aspect="equal", cmap="magma")
        axis.set_title(f"{label} (max {values.max():.2e})")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"Stage 3 absolute errors at t={float(maps['time']):.4g}, z={float(maps['z']):.4g}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_error_vs_time(path: Path, time_errors: list[dict[str, float]]) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    times = [row["time"] for row in time_errors]
    for name in ("u", "v", "w", "p", "velocity"):
        axis.semilogy(times, [max(row[name], 1e-16) for row in time_errors],
                      label=name, linewidth=2)
    axis.set_title("Stage 3 relative error versus time")
    axis.set_xlabel("time")
    axis.set_ylabel("relative L2 error")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train(config: EthierSteinmanTrainingConfig, output_dir: str | Path) -> dict[str, Any]:
    """Train a forward 3D flow PINN with Adam, optional RAR, and L-BFGS."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _resolve_device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    flow = EthierSteinmanFlow(config.viscosity, config.final_time)
    model = EthierSteinmanPINN(config.hidden_layers, config.width,
                               config.final_time,
                               config.periodic_y_features).to(device=device, dtype=dtype)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)
    history: list[dict[str, float | int | str]] = []
    adaptive: Tensor | None = None
    start = time.perf_counter()

    def sample_fixed_batch() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        initial_points, initial_velocity = flow.sample_initial_conditions(
            config.initial_points, device=device, dtype=dtype, generator=generator)
        boundary_points, boundary_velocity = flow.sample_boundary(
            config.boundary_points_per_face, device=device, dtype=dtype, generator=generator)
        interior = flow.sample_collocation(config.collocation_points, device=device,
                                           dtype=dtype, generator=generator)
        if adaptive is not None:
            interior = torch.cat((interior, adaptive), dim=0)
        return interior, initial_points, initial_velocity, boundary_points, boundary_velocity

    for step in range(1, config.steps + 1):
        interior, initial_points, initial_velocity, boundary_points, boundary_velocity = sample_fixed_batch()
        optimizer.zero_grad(set_to_none=True)
        total, momentum, continuity, initial, boundary = _losses(
            model, flow, interior, initial_points, initial_velocity, boundary_points, boundary_velocity)
        total.backward()
        optimizer.step()
        scheduler.step()

        if config.rar_every and config.rar_points and step % config.rar_every == 0:
            candidates = flow.sample_collocation(config.rar_candidates, device=device,
                                                 dtype=dtype, generator=generator)
            scores = torch.cat([_residual_scores(model, flow, block, config.rar_chunk_size)
                                for block in candidates.split(config.rar_chunk_size)])
            count = min(config.rar_points, len(candidates))
            values, indices = torch.topk(scores, count)
            new_points = candidates[indices].detach()
            adaptive = new_points if adaptive is None else torch.cat((adaptive, new_points), dim=0)
            history.append({"step": step, "phase": "rar", "added_points": count,
                            "total_adaptive_points": int(len(adaptive)),
                            "max_selected_residual": float(values[0].cpu())})

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            row: dict[str, float | int | str] = {
                "step": step, "phase": "adam", "loss": float(total.detach()),
                "momentum_loss": float(momentum.detach()),
                "continuity_loss": float(continuity.detach()),
                "initial_loss": float(initial.detach()),
                "boundary_loss": float(boundary.detach()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
            history.append(row)
            print(f"Stage 3 step={step:>6} loss={row['loss']:.3e} "
                  f"momentum={row['momentum_loss']:.3e} continuity={row['continuity_loss']:.3e} "
                  f"initial={row['initial_loss']:.3e} boundary={row['boundary_loss']:.3e}", flush=True)

    if config.lbfgs_steps:
        interior, initial_points, initial_velocity, boundary_points, boundary_velocity = sample_fixed_batch()
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=config.lbfgs_steps,
                                  tolerance_grad=1e-12 if dtype == torch.float64 else 1e-8,
                                  tolerance_change=1e-14 if dtype == torch.float64 else 1e-10,
                                  line_search_fn="strong_wolfe")
        calls = 0

        def closure() -> Tensor:
            nonlocal calls
            lbfgs.zero_grad(set_to_none=True)
            loss, *_ = _losses(model, flow, interior, initial_points, initial_velocity,
                               boundary_points, boundary_velocity)
            loss.backward()
            calls += 1
            return loss

        lbfgs.step(closure)
        final_losses = _losses(model, flow, interior, initial_points, initial_velocity,
                               boundary_points, boundary_velocity)
        history.append({"step": config.steps, "phase": "lbfgs", "closure_calls": calls,
                        "loss": float(final_losses[0].detach()),
                        "momentum_loss": float(final_losses[1].detach()),
                        "continuity_loss": float(final_losses[2].detach()),
                        "initial_loss": float(final_losses[3].detach()),
                        "boundary_loss": float(final_losses[4].detach())})

    elapsed = time.perf_counter() - start
    errors, time_errors, error_maps = evaluate(
        model, flow, config.evaluation_resolution, config.evaluation_times,
        device, dtype, config.plot_time)
    metrics: dict[str, Any] = {
        "problem": "ethier_steinman_3d_unsteady",
        "domain": {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [-1.0, 1.0],
                   "t": [0.0, config.final_time]},
        "boundary_condition": "exact_velocity_on_initial_time_and_six_spatial_faces",
        "viscosity": config.viscosity,
        "training_seconds": elapsed,
        "config": asdict(config), "errors": errors, "history": history,
    }
    torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in history for k in row)))
        writer.writeheader()
        writer.writerows(history)
    with (output / "error_vs_time.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(time_errors[0]))
        writer.writeheader()
        writer.writerows(time_errors)
    _write_error_maps(output / "pointwise_error_maps.png", error_maps)
    _write_error_vs_time(output / "error_vs_time.png", time_errors)
    return metrics
