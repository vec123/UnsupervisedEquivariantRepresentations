from __future__ import annotations

import e3nn_jax as e3nn
import flax.linen as nn

from uer.geometry.frames import rotation_matrix_from_two_vectors
from uer.models.features import scalar_node_features
from uer.models.layers import EquivariantGraphLayer


class EquivariantEncoder(nn.Module):
    hidden_irreps: tuple[str, ...] = 6 * ("32x0e + 32x0o + 16x1e + 16x1o",)
    output_irreps: str = "1x0e + 2x1o"
    break_symmetry: bool = True

    @nn.compact
    def __call__(self, graphs):
        positions = e3nn.IrrepsArray("1o", graphs.nodes)
        graphs = graphs._replace(
            nodes=scalar_node_features(len(positions), break_symmetry=self.break_symmetry)
        )

        for irreps in self.hidden_irreps + (self.output_irreps,):
            graphs = EquivariantGraphLayer(irreps)(graphs, positions)

        pooled = e3nn.scatter_sum(graphs.nodes, nel=graphs.n_node)
        invariant = pooled.filtered(keep="0e")
        vectors = pooled.filtered(keep="1o").array.reshape((-1, 2, 3))
        rotation = rotation_matrix_from_two_vectors(vectors[:, 0, :], vectors[:, 1, :])
        translation = e3nn.scatter_mean(positions.array, nel=graphs.n_node)
        return invariant, rotation, vectors, translation
