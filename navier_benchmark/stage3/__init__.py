"""Stage 3: unsteady 3D Ethier–Steinman flow."""

from .cli import main
from .ethier_steinman import EthierSteinmanFlow
from .ethier_steinman_model import EthierSteinmanPINN
from .ethier_steinman_residual import unsteady_navier_stokes_3d_residual
from .ethier_steinman_solver import EthierSteinmanTrainingConfig, train

__all__ = [
    "EthierSteinmanFlow",
    "EthierSteinmanPINN",
    "EthierSteinmanTrainingConfig",
    "main",
    "train",
    "unsteady_navier_stokes_3d_residual",
]
