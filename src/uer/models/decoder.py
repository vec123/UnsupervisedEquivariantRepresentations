from __future__ import annotations

import flax.linen as nn


class FixedNodeDecoder(nn.Module):
    nodes_per_graph: int
    hidden_dims: tuple[int, ...] = (32, 64, 128, 64)

    @nn.compact
    def __call__(self, invariant):
        x = invariant.array if hasattr(invariant, "array") else invariant
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.nodes_per_graph * 3)(x)
        return x.reshape((-1, self.nodes_per_graph, 3))
