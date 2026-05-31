from __future__ import annotations

import e3nn_jax as e3nn
import flax.linen as nn
import jraph


class EquivariantGraphLayer(nn.Module):
    target_irreps: e3nn.Irreps
    denominator: float = 1.5
    sh_lmax: int = 3

    @nn.compact
    def __call__(self, graphs, positions):
        target_irreps = e3nn.Irreps(self.target_irreps)

        def update_edge_fn(edge_features, sender_features, receiver_features, globals_):
            del edge_features, receiver_features, globals_
            sh = e3nn.spherical_harmonics(
                list(range(1, self.sh_lmax + 1)),
                positions[graphs.receivers] - positions[graphs.senders],
                normalize=True,
            )
            return e3nn.concatenate(
                [sender_features, e3nn.tensor_product(sender_features, sh)]
            ).regroup()

        def update_node_fn(node_features, sender_features, receiver_features, globals_):
            del sender_features, globals_
            node_feats = receiver_features / self.denominator
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_pre")(node_feats)
            node_feats = e3nn.scalar_activation(node_feats)
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_post")(node_feats)
            shortcut = e3nn.flax.Linear(
                node_feats.irreps,
                name="shortcut",
                force_irreps_out=True,
            )(node_features)
            return shortcut + node_feats

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graphs)
