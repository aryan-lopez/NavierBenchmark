"""Stage 1: steady 2D Kovasznay flow."""

from .cli import main
from .kovasznay import KovasznayFlow
from .kovasznay_model import VelocityPressureMLP
from .kovasznay_solver import TrainingConfig, train

__all__ = ["KovasznayFlow", "TrainingConfig", "VelocityPressureMLP", "main", "train"]
