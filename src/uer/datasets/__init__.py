"""Dataset factories that produce jraph.GraphsTuple batches."""

from uer.datasets.registry import DATASETS, build_dataset, register_dataset
from uer.datasets.npz_points import build_npz_points
from uer.datasets.tetris import TetrisConfig, build_tetris

__all__ = [
    "DATASETS",
    "TetrisConfig",
    "build_dataset",
    "build_npz_points",
    "build_tetris",
    "register_dataset",
]
