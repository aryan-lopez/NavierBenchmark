"""Stage 2: unsteady 2D Taylor–Green vortex."""

from .cli import main
from .taylor_green import TaylorGreenVortex
from .taylor_green_model import TaylorGreenPINN, periodic_boundary_loss
from .taylor_green_residual import unsteady_navier_stokes_residual
from .taylor_green_solver import TaylorGreenTrainingConfig, train_taylor_green

__all__ = [
    "TaylorGreenPINN",
    "TaylorGreenTrainingConfig",
    "TaylorGreenVortex",
    "main",
    "periodic_boundary_loss",
    "train_taylor_green",
    "unsteady_navier_stokes_residual",
]
