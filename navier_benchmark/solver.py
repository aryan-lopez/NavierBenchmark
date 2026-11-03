"""Training and evaluation for the Stage 1 Kovasznay PINN."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .kovasznay import KovasznayFlow
from .model import VelocityPressureMLP


@dataclass(frozen=True)
class TrainingConfig:
    reynolds: float
    steps: int = 20_000
    collocation_points: int = 1024
    rar_every: int = 5_000
    rar_candidates: int = 8_192
    rar_points: int = 256
    rar_chunk_size: int = 1_024
    hidden_layers: int = 5
    width: int = 64
    periodic_y_features: bool = True
    learning_rate: float = 1e-3
    lr_decay: float = 0.9999
    lbfgs_steps: int = 500
    test_resolution: int = 201
    seed: int = 1234
    device: str = "cpu"
    dtype: str = "float64"
    log_every: int = 500

    def __post_init__(self) -> None:
        if self.steps < 1 or self.collocation_points < 1:
            raise ValueError("Training steps and sample counts must be positive")
        if self.test_resolution < 2 or self.lbfgs_steps < 0:
            raise ValueError("test_resolution must be >= 2 and lbfgs_steps must be >= 0")
        if self.rar_every < 0 or self.rar_points < 0 or self.rar_candidates < 1 or self.rar_chunk_size < 1:
            raise ValueError("RAR settings must be non-negative, with positive candidate and chunk sizes")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")


def _derivative(value: Tensor, points: Tensor) -> Tensor:
    return torch.autograd.grad(value, points, grad_outputs=torch.ones_like(value),
                               create_graph=True, retain_graph=True)[0]


def navier_stokes_residual(model: nn.Module, points: Tensor, viscosity: float) -> tuple[Tensor, Tensor, Tensor]:
    """Steady 2D momentum and continuity residuals at interior points."""
    points = points.detach().requires_grad_(True)
    prediction = model(points)
    u, v, pressure = prediction.split(1, dim=1)
    grad_u = _derivative(u, points)
    grad_v = _derivative(v, points)
    grad_p = _derivative(pressure, points)
    u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
    v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
    lap_u = _derivative(u_x, points)[:, 0:1] + _derivative(u_y, points)[:, 1:2]
    lap_v = _derivative(v_x, points)[:, 0:1] + _derivative(v_y, points)[:, 1:2]
    r_u = u * u_x + v * u_y + grad_p[:, 0:1] - viscosity * lap_u
    r_v = u * v_x + v * v_y + grad_p[:, 1:2] - viscosity * lap_v
    divergence = u_x + v_y
    return r_u, r_v, divergence


def select_high_residual_points(model: nn.Module, flow: KovasznayFlow,
                                candidates: Tensor, count: int,
                                chunk_size: int = 1024) -> tuple[Tensor, Tensor]:
    """Return the candidate points with largest momentum-residual magnitude."""
    if count < 1 or chunk_size < 1:
        raise ValueError("count and chunk_size must be positive")
    score_chunks = []
    with torch.enable_grad():
        for chunk in candidates.split(chunk_size):
            r_u, r_v, _ = navier_stokes_residual(model, chunk, flow.viscosity)
            score_chunks.append(torch.sqrt(r_u.detach().square() + r_v.detach().square()).squeeze(1))
    scores = torch.cat(score_chunks)
    selected_scores, indices = torch.topk(scores, k=min(count, len(candidates)), largest=True, sorted=True)
    return candidates[indices].detach(), selected_scores.detach()


def _losses(model: nn.Module, flow: KovasznayFlow,
            interior: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r_u, r_v, divergence = navier_stokes_residual(model, interior, flow.viscosity)
    momentum_loss = (r_u.square() + r_v.square()).mean()
    continuity_loss = divergence.square().mean()
    return momentum_loss + continuity_loss, momentum_loss, continuity_loss


def _relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference.ravel())
    return float(np.linalg.norm((prediction - reference).ravel()) / max(denominator, np.finfo(float).tiny))


def evaluate(model: nn.Module, flow: KovasznayFlow, resolution: int,
             device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    """Evaluate on a dense grid independent of training samples."""
    x = torch.linspace(*flow.x_bounds, resolution, device=device, dtype=dtype)
    y = torch.linspace(*flow.y_bounds, resolution, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    with torch.no_grad():
        pred = model(points).cpu().numpy()
        exact = flow.exact(points).cpu().numpy()

    errors: dict[str, float] = {}
    for index, name in enumerate(("u", "v")):
        errors[name] = _relative_l2(pred[:, index], exact[:, index])
    # Pressure is defined up to a constant; align the means before measuring error.
    pred_pressure = pred[:, 2] - pred[:, 2].mean()
    exact_pressure = exact[:, 2] - exact[:, 2].mean()
    errors["p"] = _relative_l2(pred_pressure, exact_pressure)
    errors["velocity"] = _relative_l2(pred[:, :2], exact[:, :2])
    errors["grid_points"] = int(points.shape[0])
    return errors


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def train(config: TrainingConfig, output_dir: str | Path) -> dict[str, Any]:
    """Fit one Reynolds case and write its checkpoint, history, and metrics."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _resolve_device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    flow = KovasznayFlow(config.reynolds)
    model = VelocityPressureMLP(flow, hidden_layers=config.hidden_layers,
                                width=config.width,
                                periodic_y_features=config.periodic_y_features).to(device=device, dtype=dtype)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=config.lr_decay)
    history: list[dict[str, float | int | str]] = []
    adaptive_points: Tensor | None = None
    start = time.perf_counter()
    for step in range(1, config.steps + 1):
        uniform_points = flow.sample_interior(config.collocation_points, device=device,
                                              dtype=dtype, generator=generator)
        interior = (uniform_points if adaptive_points is None
                    else torch.cat((uniform_points, adaptive_points), dim=0))
        adam.zero_grad(set_to_none=True)
        total, momentum_loss, continuity_loss = _losses(model, flow, interior)
        total.backward()
        adam.step()
        scheduler.step()
        if (config.rar_every > 0 and config.rar_points > 0
                and step % config.rar_every == 0):
            candidates = flow.sample_interior(config.rar_candidates, device=device,
                                              dtype=dtype, generator=generator)
            new_points, selected_scores = select_high_residual_points(
                model, flow, candidates, config.rar_points, config.rar_chunk_size)
            adaptive_points = (new_points if adaptive_points is None
                               else torch.cat((adaptive_points, new_points), dim=0))
            history.append({"step": step, "phase": "rar",
                            "added_points": int(new_points.shape[0]),
                            "total_adaptive_points": int(adaptive_points.shape[0]),
                            "max_selected_residual": float(selected_scores[0].cpu())})
            print(f"Re={config.reynolds:g} RAR at step {step}: added {len(new_points)} points; "
                  f"adaptive total={len(adaptive_points)}", flush=True)
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            record = {"step": step, "loss": float(total.detach()),
                      "momentum_loss": float(momentum_loss.detach()),
                      "continuity_loss": float(continuity_loss.detach()),
                      "learning_rate": float(scheduler.get_last_lr()[0])}
            history.append(record)
            print(f"Re={config.reynolds:g} step={step:>6} loss={record['loss']:.3e} "
                  f"momentum={record['momentum_loss']:.3e} continuity={record['continuity_loss']:.3e}", flush=True)

    if config.lbfgs_steps:
        # L-BFGS requires a fixed objective throughout its line searches.
        uniform_points = flow.sample_interior(config.collocation_points, device=device,
                                              dtype=dtype, generator=generator)
        interior = (uniform_points if adaptive_points is None
                    else torch.cat((uniform_points, adaptive_points), dim=0))
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=config.lbfgs_steps,
                                  tolerance_grad=1e-12 if dtype == torch.float64 else 1e-8,
                                  tolerance_change=1e-14 if dtype == torch.float64 else 1e-10,
                                  line_search_fn="strong_wolfe")
        calls = 0

        def closure() -> Tensor:
            nonlocal calls
            lbfgs.zero_grad(set_to_none=True)
            total, _, _ = _losses(model, flow, interior)
            total.backward()
            calls += 1
            return total

        lbfgs.step(closure)
        final_loss, final_momentum, final_continuity = _losses(model, flow, interior)
        history.append({"step": config.steps, "phase": "lbfgs", "closure_calls": calls,
                        "loss": float(final_loss.detach()), "momentum_loss": float(final_momentum.detach()),
                        "continuity_loss": float(final_continuity.detach())})

    elapsed = time.perf_counter() - start
    errors = evaluate(model, flow, config.test_resolution, device, dtype)
    metrics: dict[str, Any] = {
        "problem": "kovasznay_2d_steady",
        "boundary_condition": "exact_velocity_via_coons_lifting",
        "reynolds": config.reynolds,
        "viscosity": flow.viscosity,
        "decay_rate": flow.decay_rate,
        "training_seconds": elapsed,
        "config": asdict(config),
        "errors": errors,
        "history": history,
    }
    torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output / "history.csv").open("w", encoding="utf-8") as stream:
        if history:
            import csv
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in history for k in row)))
            writer.writeheader()
            writer.writerows(history)
    return metrics
