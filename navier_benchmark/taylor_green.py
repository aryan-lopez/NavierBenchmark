"""Taylor–Green vortex exact solution and samplers for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TaylorGreenVortex:
    """Unsteady 2D Taylor–Green vortex on ``[0, 2π]² × [0, T]``.

    ``points`` passed to these methods have columns ``(x, y, t)``. The project
    proposal leaves the Reynolds number and final time to be selected, so both
    are explicit configuration values (the final time defaults to 1).
    """

    reynolds: float
    final_time: float = 1.0

    def __post_init__(self) -> None:
        if self.reynolds <= 0:
            raise ValueError("Reynolds number must be positive")
        if self.final_time <= 0:
            raise ValueError("final_time must be positive")

    @property
    def viscosity(self) -> float:
        return 1.0 / self.reynolds

    @property
    def period(self) -> float:
        return 2.0 * torch.pi

    def exact(self, points: Tensor) -> Tensor:
        """Return exact ``(u, v, p)`` at spatiotemporal ``(x, y, t)`` points."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3) with columns (x, y, t)")
        x, y, t = points[:, 0:1], points[:, 1:2], points[:, 2:3]
        nu = self.viscosity
        velocity_decay = torch.exp(-2.0 * nu * t)
        pressure_decay = torch.exp(-4.0 * nu * t)
        u = -torch.cos(x) * torch.sin(y) * velocity_decay
        v = torch.sin(x) * torch.cos(y) * velocity_decay
        pressure = -0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * pressure_decay
        return torch.cat((u, v, pressure), dim=1)

    def exact_streamfunction(self, points: Tensor) -> Tensor:
        """Return ``ψ`` where ``u=∂ψ/∂y`` and ``v=-∂ψ/∂x``."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3) with columns (x, y, t)")
        x, y, t = points[:, 0:1], points[:, 1:2], points[:, 2:3]
        return torch.cos(x) * torch.cos(y) * torch.exp(-2.0 * self.viscosity * t)

    def sample_collocation(
        self,
        count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> Tensor:
        """Sample uniform interior collocation points over space and time."""
        if count < 1:
            raise ValueError("At least one collocation point is required")
        unit = torch.rand(count, 3, device=device, dtype=dtype, generator=generator)
        scale = unit.new_tensor((self.period, self.period, self.final_time))
        return unit * scale

    def sample_initial_conditions(
        self,
        count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        """Sample ``t=0`` points and their exact initial velocity ``(u, v)``."""
        if count < 1:
            raise ValueError("At least one initial-condition point is required")
        xy = torch.rand(count, 2, device=device, dtype=dtype, generator=generator) * self.period
        time = torch.zeros(count, 1, device=device, dtype=dtype)
        points = torch.cat((xy, time), dim=1)
        return points, self.exact(points)[:, :2]

    def sample_periodic_pairs(
        self,
        count_per_direction: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample matching points on opposing edges for soft periodic penalties.

        Returns ``(x_left, x_right, y_bottom, y_top)``. Each tensor has
        ``count_per_direction`` rows of ``(x, y, t)``; paired rows have matching
        coordinates except for the periodic coordinate at 0 versus ``2π``.
        """
        if count_per_direction < 1:
            raise ValueError("At least one periodic pair per direction is required")
        n = count_per_direction
        yt = torch.rand(n, 2, device=device, dtype=dtype, generator=generator)
        yt[:, 0] *= self.period
        yt[:, 1] *= self.final_time
        x_left = torch.cat((torch.zeros(n, 1, device=device, dtype=dtype), yt), dim=1)
        x_right = torch.cat((torch.full((n, 1), self.period, device=device, dtype=dtype), yt), dim=1)

        xt = torch.rand(n, 2, device=device, dtype=dtype, generator=generator)
        xt[:, 0] *= self.period
        xt[:, 1] *= self.final_time
        y_bottom = torch.stack((xt[:, 0], torch.zeros_like(xt[:, 0]), xt[:, 1]), dim=1)
        y_top = torch.stack((xt[:, 0], torch.full_like(xt[:, 0], self.period), xt[:, 1]), dim=1)
        return x_left, x_right, y_bottom, y_top
