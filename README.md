Implementation inspired by "Unsupervised Learning of Group Invariant and Equivariant Representations" presented at NeurIPS 2022.
Use E(3) equivariant features to simultaneously learn group transformations and invariant representation spaces.

The original notebook, `tetris_test.ipynb`, is kept intact as an exploratory reference. The reusable code now lives under `src/uer` so datasets, models, geometric transforms, and trainers can be mixed and reused.

## Project Layout

- `src/uer/datasets`: graph dataset builders. Tetris is one builder, not a hard-coded training dependency.
- `src/uer/geometry`: SE(3)-style augmentations and graph rebuilding.
- `src/uer/models`: equivariant graph layers, classifier, encoder, and fixed-node decoder.
- `src/uer/trainers`: classifier and reconstruction training loops.
- `src/uer/cli`: command-line entry points for training.
- `src/uer/config.py`: typed run configuration dataclasses.
- `src/uer/factories.py`: construction layer that wires datasets, models, decoders, and trainers.
- `configs`: small example configuration values.

## Architecture Patterns

- Dataset registry: new graph sources register a builder with `register_dataset`.
- Factory layer: scripts and notebooks can call `build_classification_components` or `build_reconstruction_components` instead of hand-wiring classes.
- Typed configs: `DatasetConfig`, `ModelConfig`, `ClassificationRunConfig`, and `ReconstructionRunConfig` keep runtime choices explicit.
- Shared geometry utilities: frame construction, random transforms, and graph rebuilding are isolated from trainers and models.
- Shared model utilities: node feature initialization is defined once and reused by both classifier and encoder.
- Logger protocol: trainers accept a logger object, so notebooks, CLIs, tests, or experiment trackers can choose how messages are recorded.

## Install

```bash
pip install -e .
```

## Train On Tetris

```bash
uer-train-classifier --dataset tetris --steps 200
uer-train-reconstruction --dataset tetris --steps 1000
```

The equivalent module form also works:

```bash
python -m uer.cli.train_reconstruction --dataset tetris --steps 1000
```

## Plug In Other Graphs

The fastest path is an `.npz` file:

- `points`: shape `[num_graphs, num_nodes, 3]`, or an object array of variable-size point clouds.
- `labels`: optional integer labels of shape `[num_graphs]`.

```bash
uer-train-classifier --dataset npz_points --npz-path path/to/graphs.npz
uer-train-reconstruction --dataset npz_points --npz-path path/to/graphs.npz
```

Reconstruction currently expects a fixed number of nodes per graph because the decoder emits a fixed-size point cloud. Classification can use variable node counts.

For richer sources, add a dataset builder that returns `uer.datasets.base.GraphDataset`:

```python
from uer.datasets.base import GraphDataset, GraphDatasetInfo, batch_point_graphs
from uer.datasets.registry import register_dataset

def build_my_graphs(radius=1.1):
    points = ...  # iterable of [num_nodes, 3] arrays
    labels = ...
    graphs = batch_point_graphs(points, labels=labels, radius=radius)
    return GraphDataset(
        graphs=graphs,
        info=GraphDatasetInfo(
            name="my_graphs",
            num_graphs=len(points),
            num_classes=len(set(labels)),
            nodes_per_graph=None,
            radius=radius,
        ),
    )

register_dataset("my_graphs", build_my_graphs)
```

Then train with:

```bash
uer-train-reconstruction --dataset my_graphs --dataset-module my_package.my_dataset
```

`--dataset-module` imports your module before lookup, which lets external projects register datasets without editing this package.
