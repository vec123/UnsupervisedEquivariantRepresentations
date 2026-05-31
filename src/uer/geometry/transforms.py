from __future__ import annotations

import jax
import jax.numpy as jnp
import jraph

from uer.datasets.base import graph_from_points


def random_y_rotations(key, batch_size: int):
    theta = jax.random.uniform(key, (batch_size,), minval=0.0, maxval=2.0 * jnp.pi)

    def get_y_rot(t):
        return jnp.asarray(
            [
                [jnp.cos(t), 0.0, jnp.sin(t)],
                [0.0, 1.0, 0.0],
                [-jnp.sin(t), 0.0, jnp.cos(t)],
            ],
            dtype=jnp.float32,
        )

    return jax.vmap(get_y_rot)(theta)


def random_se3(key, batch_size: int, translation_range: float = 4.0):
    rot_key, trans_key = jax.random.split(key)
    rotations = random_y_rotations(rot_key, batch_size)
    translations = jax.random.uniform(
        trans_key,
        (batch_size, 3),
        minval=-translation_range,
        maxval=translation_range,
    )
    return rotations, translations


def apply_batched_transform(
    key,
    graphs: jraph.GraphsTuple,
    rotations,
    translations,
    radius: float,
    permute: bool = False,
) -> jraph.GraphsTuple:
    n_graphs = int(graphs.n_node.shape[0])
    keys = jax.random.split(key, n_graphs)
    node_offsets = jnp.cumsum(jnp.concatenate([jnp.asarray([0]), graphs.n_node]))
    transformed = []

    for i in range(n_graphs):
        start = int(node_offsets[i])
        stop = int(node_offsets[i + 1])
        points = graphs.nodes[start:stop]
        points = jnp.dot(points, rotations[i].T) + translations[i]
        if permute:
            points = jax.random.permutation(keys[i], points, axis=0)
        transformed.append(graph_from_points(points, label=int(graphs.globals[i]), radius=radius))

    return jraph.batch(transformed)
