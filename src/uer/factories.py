from __future__ import annotations

import importlib

from uer.config import ClassificationRunConfig, DatasetConfig, ReconstructionRunConfig
from uer.datasets import build_dataset
from uer.models import ClassificationModel, EquivariantEncoder, FixedNodeDecoder
from uer.trainers import ClassificationTrainer, ReconstructionTrainer
from uer.trainers.reconstruction import AugmentationConfig


def import_dataset_module(module: str | None) -> None:
    if module:
        importlib.import_module(module)


def dataset_kwargs(config: DatasetConfig) -> dict:
    kwargs = {"radius": config.radius}
    if config.name == "npz_points":
        if not config.path:
            raise ValueError("Dataset 'npz_points' requires DatasetConfig.path.")
        kwargs["path"] = config.path
    if config.name == "tetris":
        kwargs["include_chiral_pair"] = config.include_chiral_pair
    return kwargs


def build_dataset_from_config(config: DatasetConfig):
    import_dataset_module(config.module)
    return build_dataset(config.name, **dataset_kwargs(config))


def build_classification_components(config: ClassificationRunConfig):
    dataset = build_dataset_from_config(config.dataset)
    if dataset.info.num_classes is None:
        raise ValueError("Classification requires dataset.info.num_classes.")

    model = ClassificationModel(
        num_classes=dataset.info.num_classes,
        hidden_irreps=config.model.classifier_hidden_irreps,
        break_symmetry=config.model.break_symmetry,
    )
    trainer = ClassificationTrainer(model, learning_rate=config.learning_rate, seed=config.seed)
    return dataset, model, trainer


def build_reconstruction_components(config: ReconstructionRunConfig):
    dataset = build_dataset_from_config(config.dataset)
    if dataset.info.nodes_per_graph is None:
        raise ValueError("Reconstruction requires a fixed node count per graph.")

    encoder = EquivariantEncoder(
        hidden_irreps=config.model.encoder_hidden_irreps,
        output_irreps=config.model.encoder_output_irreps,
        break_symmetry=config.model.break_symmetry,
    )
    decoder = FixedNodeDecoder(
        nodes_per_graph=dataset.info.nodes_per_graph,
        hidden_dims=config.model.decoder_hidden_dims,
    )
    trainer = ReconstructionTrainer(
        encoder,
        decoder,
        radius=dataset.info.radius,
        learning_rate=config.learning_rate,
        augmentation=AugmentationConfig(
            translation_range=config.translation_range,
            permute_nodes=config.permute_nodes,
        ),
        seed=config.seed,
    )
    return dataset, encoder, decoder, trainer
