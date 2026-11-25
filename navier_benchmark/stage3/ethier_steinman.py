"""Exact 3D unsteady Ethier--Steinman (Beltrami) flow for Stage 3.

Coordinates use columns ``(x, y, z, t)`` on ``[-1, 1]^3 x [0, 1]``. The
proposal fixes the solution parameters ``a = d = 1``; viscosity is supplied
by the caller so the same case can be used for forward and inverse runs.
Velocity is prescribed from the exact solution on all six spatial faces and
at the initial time. Pressure is returned for evaluation, not imposed as a
boundary condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EthierSteinmanFlow:
    """The Stage 3 Ethier--Steinman flow on the proposal's unit cube."""

    viscosity: float
    final_time: float = 1.0
    a: float = 1.0
    d: float = 1.0

    def __post_init__(self) -> None:
        if self.viscosity <= 0:
            raise ValueError("viscosity must be positive")
        if self.final_time <= 0:
            raise ValueError("final_time must be positive")
        if self.a == 0 or self.d == 0:
            raise ValueError("a and d must be nonzero")

    @property
    def lower(self) -> tuple[float, float, float]:
        return (-1.0, -1.0, -1.0)

    @property
    def upper(self) -> tuple[float, float, float]:
        return (1.0, 1.0, 1.0)

    def exact(self, points: Tensor) -> Tensor:
        """Return exact ``(u, v, w, p)`` at ``(x, y, z, t)`` points.

        This is the analytic Ethier--Steinman solution shown in the proposal.
        The pressure gauge matches that formula.
        """
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("points must have shape (N, 4) with columns (x, y, z, t)")
        x, y, z, t = (points[:, i:i + 1] for i in range(4))
        a, d, nu = self.a, self.d, self.viscosity
        decay = torch.exp(-nu * d * d * t)
        pressure_decay = decay.square()

        u = -a * (
            torch.exp(a * x) * torch.sin(a * y + d * z)
            + torch.exp(a * z) * torch.cos(a * x + d * y)
        ) * decay
        v = -d * (
            torch.exp(a * y) * torch.sin(a * z + d * x)
            + torch.exp(a * x) * torch.cos(a * y + d * z)
        ) * decay
        w = -d * (
            torch.exp(a * z) * torch.sin(a * x + d * y)
            + torch.exp(a * y) * torch.cos(a * z + d * x)
        ) * decay

        p = -0.5 * (
            torch.exp(2.0 * a * x)
            + torch.exp(2.0 * a * y)
            + torch.exp(2.0 * a * z)
            + 2.0 * torch.sin(a * x + d * y) * torch.cos(a * z + d * x)
              * torch.exp(a * (y + z))
            + 2.0 * torch.sin(a * y + d * z) * torch.cos(a * x + d * y)
              * torch.exp(a * (z + x))
            + 2.0 * torch.sin(a * z + d * x) * torch.cos(a * y + d * z)
              * torch.exp(a * (x + y))
        ) * pressure_decay
        return torch.cat((u, v, w, p), dim=1)

    def sample_collocation(
        self,
        count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> Tensor:
        """Sample uniform interior space-time points on ``[-1,1]^3 x [0,T]``."""
        if count < 1:
            raise ValueError("At least one collocation point is required")
        unit = torch.rand(count, 4, device=device, dtype=dtype, generator=generator)
        lower = unit.new_tensor((-1.0, -1.0, -1.0, 0.0))
        span = unit.new_tensor((2.0, 2.0, 2.0, self.final_time))
        return lower + unit * span

    def sample_initial_conditions(
        self,
        count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        """Sample ``t=0`` coordinates and exact initial velocity ``(u,v,w)``."""
        if count < 1:
            raise ValueError("At least one initial-condition point is required")
        xyz = 2.0 * torch.rand(count, 3, device=device, dtype=dtype, generator=generator) - 1.0
        points = torch.cat((xyz, torch.zeros(count, 1, device=device, dtype=dtype)), dim=1)
        return points, self.exact(points)[:, :3]

    def sample_boundary(
        self,
        count_per_face: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        """Sample all six spatial faces and return exact velocity targets.

        Spatial points lie on the six faces of ``[-1,1]^3`` and include a
        uniform time coordinate in ``[0, final_time]``.
        Returns ``(points, velocity)`` with shapes ``(6*n, 4)`` and ``(6*n, 3)``.
        """
        if count_per_face < 1:
            raise ValueError("At least one boundary point per face is required")
        n = count_per_face
        faces: list[Tensor] = []
        for axis in range(3):
            free_axes = [i for i in range(3) if i != axis]
            free = 2.0 * torch.rand(n, 2, device=device, dtype=dtype, generator=generator) - 1.0
            free = torch.cat((free, torch.rand(n, 1, device=device, dtype=dtype,
                                               generator=generator) * self.final_time), dim=1)
            for side in (-1.0, 1.0):
                face = torch.empty(n, 4, device=device, dtype=dtype)
                face[:, axis] = side
                face[:, free_axes[0]] = free[:, 0]
                face[:, free_axes[1]] = free[:, 1]
                face[:, 3] = free[:, 2]
                faces.append(face)
        points = torch.cat(faces, dim=0)
        return points, self.exact(points)[:, :3]
