from __future__ import annotations

import e3nn_jax as e3nn
import jax.numpy as jnp


def scalar_node_features(num_nodes: int, break_symmetry: bool = True) -> e3nn.IrrepsArray:
    if break_symmetry:
        values = jnp.arange(num_nodes).reshape(-1, 1)
    else:
        values = jnp.ones((num_nodes, 1))
    return e3nn.IrrepsArray("1x0e", values)
