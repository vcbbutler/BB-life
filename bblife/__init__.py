"""
Conway's Game of Life - 3D GPU-accelerated implementation
"""

from .model import GameOfLife, SparseGameOfLife
from .view import animate_game
from .settings import SettingsDialog
from .constants import *

__version__ = "0.1.0" 