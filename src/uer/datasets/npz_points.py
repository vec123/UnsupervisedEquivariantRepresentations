from __future__ import annotations

from pathlib import Path

import numpy as np

from uer.datasets.base import GraphDataset, GraphDatasetInfo, batch_point_graphs, infer_nodes_per_graph
from uer.datasets.registry import register_dataset


def build_npz_points(path: str, radius: float = 1.1) -> GraphDataset:
    """Build graphs from an NPZ file with `points` and optional `labels` arrays.

    `points` should have shape `[num_graphs, num_nodes, 3]` for fixed-size graphs, or
    be an object array/list of `[num_nodes_i, 3]` arrays for variable-size graphs.
    """

    npz_path = Path(path)
    data = np.load(npz_path, allow_pickle=True)
    points = data["points"]
    labels = data["labels"] if "labels" in data else np.arange(len(points), dtype=np.int32)
    graphs = batch_point_graphs(points, labels=labels, radius=radius)
    unique_labels = np.unique(labels)
    info = GraphDatasetInfo(
        name=npz_path.stem,
        num_graphs=int(len(points)),
        num_classes=int(len(unique_labels)),
        nodes_per_graph=infer_nodes_per_graph(graphs),
        radius=radius,
    )
    return GraphDataset(graphs=graphs, info=info)


register_dataset("npz_points", build_npz_points)
