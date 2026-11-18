"""Automatic-differentiation residuals for unsteady 2D Navier–Stokes PINNs."""

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
    """Return ``(R_u, R_v, R_c)`` at ``(x, y, t)`` points.

    The model must map an ``(N, 3)`` tensor of ``(x, y, t)`` coordinates to
    an ``(N, 3)`` tensor of ``(u, v, p)`` predictions. Momentum residuals use
    ``u_t + u·∇u + ∇p - νΔu = 0``; continuity is ``u_x + v_y = 0``.
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


def unsteady_navier_stokes_3d_residual(
    model: nn.Module,
    points: Tensor,
    viscosity: float | Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``(R_u, R_v, R_w, R_c)`` for 3D unsteady incompressible flow.

    ``points`` has columns ``(x, y, z, t)`` and the model returns
    ``(u, v, w, p)``. The residual is
    ``u_t + (u dot grad)u + grad(p) - nu*laplacian(u) = 0``
    componentwise, with continuity ``u_x + v_y + w_z = 0``. ``viscosity`` may
    be a scalar tensor parameter, which keeps the residual differentiable for
    inverse training.
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
        laplacian = sum(_derivative(grad_component[:, axis:axis + 1], points)[:, axis:axis + 1]
                        for axis in range(3))
        convection = sum(velocity[axis] * grad_component[:, axis:axis + 1]
                         for axis in range(3))
        index = len(residuals)
        residuals.append(time_component + convection + pressure_grad[:, index:index + 1]
                         - viscosity * laplacian)

    continuity = grad_u[:, 0:1] + grad_v[:, 1:2] + grad_w[:, 2:3]
    return residuals[0], residuals[1], residuals[2], continuity
