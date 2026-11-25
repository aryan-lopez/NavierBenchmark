"""Automatic-differentiation residuals for the Stage 3 unsteady 3D flow."""

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


def unsteady_navier_stokes_3d_residual(
    model: nn.Module,
    points: Tensor,
    viscosity: float | Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return 3D momentum and continuity residuals at ``(x, y, z, t)`` points.

    The model maps coordinates of shape ``(N, 4)`` to ``(u, v, w, p)``.
    Viscosity may be a scalar tensor so inverse training can differentiate the
    residual with respect to it.
    """
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points must have shape (N, 4) with columns (x, y, z, t)")
    if isinstance(viscosity, (int, float)) and viscosity <= 0:
        raise ValueError("viscosity must be positive")

    points = points.detach().requires_grad_(True)
    prediction = model(points)
    if prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ValueError("model must return columns (u, v, w, p) with shape (N, 4)")
    u, v, w, pressure = prediction.split(1, dim=1)
    gradients = [_derivative(value, points) for value in (u, v, w, pressure)]
    grad_u, grad_v, grad_w, grad_p = gradients
    velocity = (u, v, w)
    velocity_grads = (grad_u, grad_v, grad_w)
    pressure_grad = grad_p[:, :3]
    time_grads = (grad_u[:, 3:4], grad_v[:, 3:4], grad_w[:, 3:4])

    residuals: list[Tensor] = []
    for component, grad_component, time_component in zip(velocity, velocity_grads, time_grads):
        laplacian = sum(
            _derivative(grad_component[:, axis:axis + 1], points)[:, axis:axis + 1]
            for axis in range(3)
        )
        convection = sum(velocity[axis] * grad_component[:, axis:axis + 1]
                         for axis in range(3))
        index = len(residuals)
        residuals.append(time_component + convection + pressure_grad[:, index:index + 1]
                         - viscosity * laplacian)

    continuity = grad_u[:, 0:1] + grad_v[:, 1:2] + grad_w[:, 2:3]
    return residuals[0], residuals[1], residuals[2], continuity
