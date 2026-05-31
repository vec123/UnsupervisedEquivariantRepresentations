from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph


@dataclass(frozen=True)
class GraphDatasetInfo:
    """Small amount of metadata needed by trainers and model heads."""

    name: str
    num_graphs: int
    num_classes: int | None
    nodes_per_graph: int | None
    radius: float


@dataclass(frozen=True)
class GraphDataset:
    """A batched graph plus metadata.

    The training code only depends on this object, so new graph sources can be added by
    returning a GraphsTuple with node coordinates in `nodes` and labels in `globals`.
    """

    graphs: jraph.GraphsTuple
    info: GraphDatasetInfo


class DatasetBuilder(Protocol):
    def __call__(self, **kwargs) -> GraphDataset:
        """Build a dataset from keyword configuration."""


def graph_from_points(points, label: int | None = None, radius: float = 1.1) -> jraph.GraphsTuple:
    points = jnp.asarray(points, dtype=jnp.float32)
    senders, receivers = e3nn.radius_graph(points, radius)
    globals_ = jnp.asarray([0 if label is None else label], dtype=jnp.int32)
    return jraph.GraphsTuple(
        nodes=points,
        edges=None,
        globals=globals_,
        senders=senders,
        receivers=receivers,
        n_node=jnp.asarray([points.shape[0]], dtype=jnp.int32),
        n_edge=jnp.asarray([senders.shape[0]], dtype=jnp.int32),
    )


def batch_point_graphs(
    point_clouds: Iterable,
    labels: Iterable[int] | None = None,
    radius: float = 1.1,
) -> jraph.GraphsTuple:
    point_clouds = list(point_clouds)
    if labels is None:
        labels = range(len(point_clouds))

    graphs = [
        graph_from_points(points=points, label=int(label), radius=radius)
        for points, label in zip(point_clouds, labels)
    ]
    return jraph.batch(graphs)


def infer_nodes_per_graph(graphs: jraph.GraphsTuple) -> int | None:
    n_node = [int(n) for n in list(graphs.n_node)]
    if not n_node:
        return None
    first = n_node[0]
    return first if all(n == first for n in n_node) else None
