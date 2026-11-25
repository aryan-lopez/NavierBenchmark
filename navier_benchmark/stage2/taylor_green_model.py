"""Taylor–Green PINN formulations and periodic boundary handling."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from .taylor_green import TaylorGreenVortex

Formulation = Literal["velocity_pressure", "streamfunction"]
Periodicity = Literal["soft", "hard"]


class TaylorGreenPINN(nn.Module):
    """MLP for velocity–pressure or streamfunction–pressure formulations.

    ``forward`` always returns physical ``(u, v, p)`` fields so it can be used
    directly with ``taylor_green_residual.unsteady_navier_stokes_residual``. For the
    streamfunction formulation, ``raw_outputs`` returns ``(ψ, p)`` and
    ``forward`` derives ``u=ψ_y`` and ``v=-ψ_x`` with automatic differentiation.

    Hard periodicity uses ``(sin x, cos x, sin y, cos y, scaled t)`` features.
    Soft periodicity keeps raw coordinates (optionally augmented with
    ``sin/cos`` x/y features) and is imposed with :func:`periodic_boundary_loss`.
    """

    def __init__(
        self,
        flow: TaylorGreenVortex,
        formulation: Formulation = "velocity_pressure",
        periodicity: Periodicity = "soft",
        hidden_layers: int = 5,
        width: int = 64,
        periodic_features: bool = False,
    ) -> None:
        super().__init__()
        if formulation not in ("velocity_pressure", "streamfunction"):
            raise ValueError("formulation must be 'velocity_pressure' or 'streamfunction'")
        if periodicity not in ("soft", "hard"):
            raise ValueError("periodicity must be 'soft' or 'hard'")
        if hidden_layers < 1 or width < 1:
            raise ValueError("hidden_layers and width must be positive")

        self.flow = flow
        self.formulation = formulation
        self.periodicity = periodicity
        self.periodic_features = periodic_features
        lower = torch.zeros(3, dtype=torch.get_default_dtype())
        span = torch.tensor((flow.period, flow.period, flow.final_time),
                            dtype=torch.get_default_dtype())
        self.register_buffer("lower", lower)
        self.register_buffer("span", span)

        input_size = 5 if periodicity == "hard" else (7 if periodic_features else 3)
        output_size = 3 if formulation == "velocity_pressure" else 2
        dimensions = [input_size] + [width] * hidden_layers + [output_size]
        layers: list[nn.Module] = []
        for index, (n_in, n_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            linear = nn.Linear(n_in, n_out)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index < len(dimensions) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def _features(self, points: Tensor) -> Tensor:
        scaled = 2.0 * (points - self.lower) / self.span - 1.0
        if self.periodicity == "soft":
            if self.periodic_features:
                x, y = points[:, 0:1], points[:, 1:2]
                return torch.cat((scaled, torch.sin(x), torch.cos(x),
                                  torch.sin(y), torch.cos(y)), dim=1)
            return scaled
        x, y, scaled_t = points[:, 0:1], points[:, 1:2], scaled[:, 2:3]
        return torch.cat((torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y), scaled_t), dim=1)

    def raw_outputs(self, points: Tensor) -> Tensor:
        """Return network outputs: ``(u,v,p)`` or ``(ψ,p)``."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3) with columns (x, y, t)")
        return self.network(self._features(points))

    def forward(self, points: Tensor) -> Tensor:
        """Return physical ``(u, v, p)`` fields for spatiotemporal points."""
        if self.formulation == "velocity_pressure":
            return self.raw_outputs(points)

        # Streamfunction derivatives are required even when a caller wraps
        # inference in no_grad; enable autograd locally for this transformation.
        with torch.enable_grad():
            coordinates = points if points.requires_grad else points.detach().requires_grad_(True)
            psi, pressure = self.raw_outputs(coordinates).split(1, dim=1)
            grad_psi = torch.autograd.grad(
                psi,
                coordinates,
                grad_outputs=torch.ones_like(psi),
                create_graph=True,
                retain_graph=True,
            )[0]
            u = grad_psi[:, 1:2]
            v = -grad_psi[:, 0:1]
            return torch.cat((u, v, pressure), dim=1)


def periodic_boundary_loss(
    model: TaylorGreenPINN,
    pairs: tuple[Tensor, Tensor, Tensor, Tensor],
) -> Tensor:
    """Mean squared physical-field mismatch on opposite periodic edges.

    Velocity, pressure, and (for the streamfunction formulation) derived
    velocity are compared. Hard-periodic models need no penalty and return a
    scalar zero on the model's device and dtype.
    """
    if model.periodicity == "hard":
        return next(model.parameters()).new_zeros(())
    x_left, x_right, y_bottom, y_top = pairs
    x_mismatch = model(x_left) - model(x_right)
    y_mismatch = model(y_bottom) - model(y_top)
    return 0.5 * (x_mismatch.square().sum(dim=1).mean()
                  + y_mismatch.square().sum(dim=1).mean())
