from __future__ import annotations

import argparse

import uer.datasets.npz_points  # noqa: F401
import uer.datasets.tetris  # noqa: F401
from uer.config import DatasetConfig, ModelConfig, ReconstructionRunConfig
from uer.factories import build_reconstruction_components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tetris")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--radius", type=float, default=1.1)
    parser.add_argument("--translation-range", type=float, default=4.0)
    parser.add_argument("--permute-nodes", action="store_true")
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--dataset-module", default=None)
    parser.add_argument("--no-break-symmetry", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    config = ReconstructionRunConfig(
        dataset=DatasetConfig(
            name=args.dataset,
            radius=args.radius,
            path=args.npz_path,
            module=args.dataset_module,
        ),
        model=ModelConfig(break_symmetry=not args.no_break_symmetry),
        steps=args.steps,
        learning_rate=args.learning_rate,
        translation_range=args.translation_range,
        permute_nodes=args.permute_nodes,
        log_every=args.log_every,
    )
    dataset, _, _, trainer = build_reconstruction_components(config)
    trainer.fit(dataset.graphs, num_steps=config.steps, log_every=config.log_every)


if __name__ == "__main__":
    main()
