from __future__ import annotations

import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import numpy as np


def plot_graph_batch(batched_graph: jraph.GraphsTuple, indices: list[int] | None = None):
    graphs = jraph.unbatch(batched_graph)
    if indices is not None:
        graphs = [graphs[i] for i in indices]

    cols = 3
    rows = (len(graphs) + cols - 1) // cols
    fig = plt.figure(figsize=(15, 5 * rows))

    for i, graph in enumerate(graphs):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        points = np.asarray(graph.nodes)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=120,
            c="royalblue",
            edgecolors="black",
            alpha=0.85,
        )
        for sender, receiver in zip(graph.senders, graph.receivers):
            p0 = points[int(sender)]
            p1 = points[int(receiver)]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="black", alpha=0.4)

        center = jnp.mean(graph.nodes, axis=0)
        radius = max(float(jnp.max(jnp.abs(graph.nodes - center))), 1.0)
        ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
        ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
        ax.set_zlim(float(center[2] - radius), float(center[2] + radius))
        label = indices[i] if indices is not None else i
        ax.set_title(f"Graph {label} | label {int(graph.globals[0])}")

    plt.tight_layout()
    return fig
