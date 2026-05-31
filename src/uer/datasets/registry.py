from __future__ import annotations

from uer.datasets.base import DatasetBuilder, GraphDataset

DATASETS: dict[str, DatasetBuilder] = {}


def register_dataset(name: str, builder: DatasetBuilder) -> None:
    normalized = name.lower().replace("-", "_")
    DATASETS[normalized] = builder


def build_dataset(name: str, **kwargs) -> GraphDataset:
    normalized = name.lower().replace("-", "_")
    try:
        builder = DATASETS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(DATASETS)) or "<none>"
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}") from exc
    return builder(**kwargs)
