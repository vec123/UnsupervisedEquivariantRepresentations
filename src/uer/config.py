from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "tetris"
    radius: float = 1.1
    path: str | None = None
    module: str | None = None
    include_chiral_pair: bool = False


@dataclass(frozen=True)
class ModelConfig:
    break_symmetry: bool = True
    classifier_hidden_irreps: tuple[str, ...] = (
        "32x0e + 32x0o + 8x1e + 8x1o + 8x2e + 8x2o",
        "32x0e + 32x0o + 8x1e + 8x1o + 8x2e + 8x2o",
    )
    encoder_hidden_irreps: tuple[str, ...] = 6 * ("32x0e + 32x0o + 16x1e + 16x1o",)
    encoder_output_irreps: str = "1x0e + 2x1o"
    decoder_hidden_dims: tuple[int, ...] = (32, 64, 128, 64)


@dataclass(frozen=True)
class ClassificationRunConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    steps: int = 200
    learning_rate: float = 1e-2
    seed: int = 3


@dataclass(frozen=True)
class ReconstructionRunConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    steps: int = 1000
    learning_rate: float = 1e-3
    translation_range: float = 4.0
    permute_nodes: bool = False
    log_every: int = 100
    seed: int = 0
