from __future__ import annotations

import argparse

import uer.datasets.npz_points  # noqa: F401
import uer.datasets.tetris  # noqa: F401
from uer.config import ClassificationRunConfig, DatasetConfig, ModelConfig
from uer.factories import build_classification_components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tetris")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--radius", type=float, default=1.1)
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--dataset-module", default=None)
    parser.add_argument("--no-break-symmetry", action="store_true")
    args = parser.parse_args()

    config = ClassificationRunConfig(
        dataset=DatasetConfig(
            name=args.dataset,
            radius=args.radius,
            path=args.npz_path,
            module=args.dataset_module,
        ),
        model=ModelConfig(break_symmetry=not args.no_break_symmetry),
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    dataset, _, trainer = build_classification_components(config)
    trainer.fit(dataset.graphs, steps=config.steps)


if __name__ == "__main__":
    main()
