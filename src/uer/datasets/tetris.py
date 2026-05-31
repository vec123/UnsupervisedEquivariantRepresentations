from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from uer.datasets.base import GraphDataset, GraphDatasetInfo, batch_point_graphs
from uer.datasets.registry import register_dataset


@dataclass(frozen=True)
class TetrisConfig:
    radius: float = 1.1
    include_chiral_pair: bool = False


def tetris_point_clouds(include_chiral_pair: bool = False):
    chiral = [
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],
        [[1, 1, 1], [1, 1, 2], [2, 1, 1], [2, 0, 1]],
    ]
    base = [
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],
    ]
    points = (chiral if include_chiral_pair else []) + base
    return jnp.asarray(points, dtype=jnp.float32)


def build_tetris(radius: float = 1.1, include_chiral_pair: bool = False) -> GraphDataset:
    points = tetris_point_clouds(include_chiral_pair=include_chiral_pair)
    labels = range(points.shape[0])
    graphs = batch_point_graphs(points, labels=labels, radius=radius)
    info = GraphDatasetInfo(
        name="tetris",
        num_graphs=int(points.shape[0]),
        num_classes=int(points.shape[0]),
        nodes_per_graph=int(points.shape[1]),
        radius=radius,
    )
    return GraphDataset(graphs=graphs, info=info)


register_dataset("tetris", build_tetris)
