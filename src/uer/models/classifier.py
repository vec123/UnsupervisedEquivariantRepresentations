from __future__ import annotations

import e3nn_jax as e3nn
import flax.linen as nn

from uer.models.features import scalar_node_features
from uer.models.layers import EquivariantGraphLayer


class ClassificationModel(nn.Module):
    num_classes: int
    hidden_irreps: tuple[str, ...] = (
        "32x0e + 32x0o + 8x1e + 8x1o + 8x2e + 8x2o",
        "32x0e + 32x0o + 8x1e + 8x1o + 8x2e + 8x2o",
    )
    break_symmetry: bool = True

    @nn.compact
    def __call__(self, graphs):
        positions = e3nn.IrrepsArray("1o", graphs.nodes)
        graphs = graphs._replace(
            nodes=scalar_node_features(len(positions), break_symmetry=self.break_symmetry)
        )

        for irreps in self.hidden_irreps + (f"{self.num_classes}x0e",):
            graphs = EquivariantGraphLayer(irreps)(graphs, positions)

        logits = e3nn.scatter_sum(graphs.nodes.array, nel=graphs.n_node)
        return logits
