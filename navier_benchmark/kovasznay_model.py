"""Fully connected velocity–pressure network used by the Stage 1 PINN."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .kovasznay import KovasznayFlow


class VelocityPressureMLP(nn.Module):
    """Tanh MLP with exactly imposed velocity boundary conditions."""

    def __init__(self, flow: KovasznayFlow, hidden_layers: int = 5, width: int = 64,
                 periodic_y_features: bool = True) -> None:
        super().__init__()
        if hidden_layers < 1 or width < 1:
            raise ValueError("hidden_layers and width must be positive")
        self.flow = flow
        self.periodic_y_features = periodic_y_features
        lower = torch.tensor((flow.x_bounds[0], flow.y_bounds[0]), dtype=torch.get_default_dtype())
        span = torch.tensor((flow.x_bounds[1] - flow.x_bounds[0],
                             flow.y_bounds[1] - flow.y_bounds[0]), dtype=torch.get_default_dtype())
        self.register_buffer("lower", lower)
        self.register_buffer("span", span)
        input_size = 4 if periodic_y_features else 2
        dimensions = [input_size] + [width] * hidden_layers + [3]
        layers: list[nn.Module] = []
        for index, (n_in, n_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            linear = nn.Linear(n_in, n_out)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index < len(dimensions) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, points: Tensor) -> Tensor:
        scaled = 2.0 * (points - self.lower) / self.span - 1.0
        if self.periodic_y_features:
            y = points[:, 1:2]
            periodic = torch.cat((torch.sin(2.0 * torch.pi * y),
                                  torch.cos(2.0 * torch.pi * y)), dim=1)
            features = torch.cat((scaled, periodic), dim=1)
        else:
            features = scaled
        raw = self.network(features)
        envelope = self.flow.boundary_envelope(points)
        velocity = self.flow.boundary_lift(points) + envelope * raw[:, :2]
        return torch.cat((velocity, raw[:, 2:3]), dim=1)
