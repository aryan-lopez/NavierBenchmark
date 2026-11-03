"""Exact solution and domain definition for the Stage 1 Kovasznay flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class KovasznayFlow:
    """Steady incompressible Kovasznay solution on the proposal's rectangle."""

    reynolds: float
    x_bounds: tuple[float, float] = (-0.5, 1.0)
    y_bounds: tuple[float, float] = (-0.5, 1.5)

    def __post_init__(self) -> None:
        if self.reynolds <= 0:
            raise ValueError("Reynolds number must be positive")

    @property
    def viscosity(self) -> float:
        return 1.0 / self.reynolds

    @property
    def decay_rate(self) -> float:
        re = self.reynolds
        return re / 2.0 - (re**2 / 4.0 + 4.0 * torch.pi**2) ** 0.5

    def exact(self, points: Tensor) -> Tensor:
        """Return columns ``(u, v, p)`` for points with columns ``(x, y)``."""
        x, y = points[:, 0:1], points[:, 1:2]
        lam = self.decay_rate
        exponential = torch.exp(lam * x)
        u = 1.0 - exponential * torch.cos(2.0 * torch.pi * y)
        v = (lam / (2.0 * torch.pi)) * exponential * torch.sin(2.0 * torch.pi * y)
        p = 0.5 * (1.0 - torch.exp(2.0 * lam * x))
        return torch.cat((u, v, p), dim=1)

    def boundary_lift(self, points: Tensor) -> Tensor:
        """Coons-patch extension of prescribed edge velocity values.

        Exact values are evaluated only on the rectangle edges; this is a
        smooth extension into the interior, not the exact interior solution.
        """
        x, y = points[:, 0:1], points[:, 1:2]
        x0, x1 = self.x_bounds
        y0, y1 = self.y_bounds
        s = (x - x0) / (x1 - x0)
        t = (y - y0) / (y1 - y0)

        left = torch.cat((torch.full_like(x, x0), y), dim=1)
        right = torch.cat((torch.full_like(x, x1), y), dim=1)
        bottom = torch.cat((x, torch.full_like(y, y0)), dim=1)
        top = torch.cat((x, torch.full_like(y, y1)), dim=1)
        left_right = (1.0 - s) * self.exact(left)[:, :2] + s * self.exact(right)[:, :2]
        bottom_top = (1.0 - t) * self.exact(bottom)[:, :2] + t * self.exact(top)[:, :2]

        corners = points.new_tensor(((x0, y0), (x1, y0), (x0, y1), (x1, y1)))
        c00, c10, c01, c11 = self.exact(corners)[:, :2].unbind(dim=0)
        corner_blend = ((1.0 - s) * (1.0 - t) * c00
                        + s * (1.0 - t) * c10
                        + (1.0 - s) * t * c01
                        + s * t * c11)
        return left_right + bottom_top - corner_blend

    def boundary_envelope(self, points: Tensor) -> Tensor:
        """Smooth factor equal to zero on all edges and positive inside."""
        x0, x1 = self.x_bounds
        y0, y1 = self.y_bounds
        s = (points[:, 0:1] - x0) / (x1 - x0)
        t = (points[:, 1:2] - y0) / (y1 - y0)
        return 16.0 * s * (1.0 - s) * t * (1.0 - t)

    def sample_boundary(self, count_per_edge: int, *, device: torch.device, dtype: torch.dtype,
                        generator: torch.Generator) -> tuple[Tensor, Tensor]:
        """Sample Dirichlet points on all four edges, including corners."""
        if count_per_edge < 2:
            raise ValueError("At least two points per boundary edge are required")
        x0, x1 = self.x_bounds
        y0, y1 = self.y_bounds
        tx = torch.rand(count_per_edge, 1, device=device, dtype=dtype, generator=generator)
        ty = torch.rand(count_per_edge, 1, device=device, dtype=dtype, generator=generator)
        horizontal_x = x0 + (x1 - x0) * tx
        vertical_y = y0 + (y1 - y0) * ty
        bottom = torch.cat((horizontal_x, torch.full_like(horizontal_x, y0)), dim=1)
        top = torch.cat((horizontal_x, torch.full_like(horizontal_x, y1)), dim=1)
        left = torch.cat((torch.full_like(vertical_y, x0), vertical_y), dim=1)
        right = torch.cat((torch.full_like(vertical_y, x1), vertical_y), dim=1)
        points = torch.cat((bottom, top, left, right), dim=0)
        return points, self.exact(points)[:, :2]

    def sample_interior(self, count: int, *, device: torch.device, dtype: torch.dtype,
                        generator: torch.Generator) -> Tensor:
        """Draw uniform collocation points, excluding the boundary."""
        if count < 1:
            raise ValueError("At least one interior collocation point is required")
        unit = torch.rand(count, 2, device=device, dtype=dtype, generator=generator)
        lower = torch.tensor((self.x_bounds[0], self.y_bounds[0]), device=device, dtype=dtype)
        upper = torch.tensor((self.x_bounds[1], self.y_bounds[1]), device=device, dtype=dtype)
        return lower + unit * (upper - lower)
