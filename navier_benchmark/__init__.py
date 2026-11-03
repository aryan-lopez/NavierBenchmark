"""PINN benchmark implementations for incompressible Navier–Stokes flows."""

from .kovasznay import KovasznayFlow
from .model import VelocityPressureMLP
from .solver import TrainingConfig, train
from .taylor_green import TaylorGreenVortex
from .taylor_green_model import TaylorGreenPINN, periodic_boundary_loss
from .unsteady import unsteady_navier_stokes_residual

__all__ = [
    "KovasznayFlow",
    "TaylorGreenVortex",
    "TaylorGreenPINN",
    "TrainingConfig",
    "VelocityPressureMLP",
    "train",
    "unsteady_navier_stokes_residual",
    "periodic_boundary_loss",
]
