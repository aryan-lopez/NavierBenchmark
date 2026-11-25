"""Automatic-differentiation residuals for the Stage 2 unsteady 2D flow."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _derivative(value: Tensor, points: Tensor) -> Tensor:
    return torch.autograd.grad(
        value,
        points,
        grad_outputs=torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
    )[0]


def unsteady_navier_stokes_residual(
    model: nn.Module,
    points: Tensor,
    viscosity: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return 2D momentum and continuity residuals at ``(x, y, t)`` points.

    The model maps ``(N, 3)`` coordinates to ``(N, 3)`` predictions
    ``(u, v, p)``. Momentum uses ``u_t + (u dot grad)u + grad(p) - nu*lap(u)``;
    continuity is ``u_x + v_y = 0``.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3) with columns (x, y, t)")
    if viscosity <= 0:
        raise ValueError("viscosity must be positive")

    points = points.detach().requires_grad_(True)
    prediction = model(points)
    if prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("model must return columns (u, v, p) with shape (N, 3)")
    u, v, pressure = prediction.split(1, dim=1)
    grad_u = _derivative(u, points)
    grad_v = _derivative(v, points)
    grad_p = _derivative(pressure, points)

    u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    v_x, v_y, v_t = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
    p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
    lap_u = _derivative(u_x, points)[:, 0:1] + _derivative(u_y, points)[:, 1:2]
    lap_v = _derivative(v_x, points)[:, 0:1] + _derivative(v_y, points)[:, 1:2]

    residual_u = u_t + u * u_x + v * u_y + p_x - viscosity * lap_u
    residual_v = v_t + u * v_x + v * v_y + p_y - viscosity * lap_v
    continuity = u_x + v_y
    return residual_u, residual_v, continuity
