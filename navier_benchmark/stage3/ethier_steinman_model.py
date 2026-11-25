"""Fully connected PINN for the Stage 3 Ethier--Steinman flow."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EthierSteinmanPINN(nn.Module):
    """Tanh MLP mapping normalized ``(x,y,z,t)`` to ``(u,v,w,p)``."""

    def __init__(self, hidden_layers: int = 5, width: int = 64,
                 final_time: float = 1.0, periodic_y_features: bool = True) -> None:
        super().__init__()
        if hidden_layers < 1 or width < 1:
            raise ValueError("hidden_layers and width must be positive")
        if final_time <= 0:
            raise ValueError("final_time must be positive")
        self.final_time = final_time
        self.periodic_y_features = periodic_y_features
        dimensions = [6 if periodic_y_features else 4] + [width] * hidden_layers + [4]
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
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("points must have shape (N, 4) with columns (x, y, z, t)")
        lower = points.new_tensor((-1.0, -1.0, -1.0, 0.0))
        span = points.new_tensor((2.0, 2.0, 2.0, self.final_time))
        normalized = 2.0 * (points - lower) / span - 1.0
        if self.periodic_y_features:
            y = points[:, 1:2]
            periodic_y = torch.cat((torch.sin(2.0 * torch.pi * y),
                                    torch.cos(2.0 * torch.pi * y)), dim=1)
            normalized = torch.cat((normalized, periodic_y), dim=1)
        return self.network(normalized)
