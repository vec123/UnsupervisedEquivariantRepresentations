from __future__ import annotations

from typing import Protocol


class TrainingLogger(Protocol):
    def info(self, message: str) -> None:
        """Record a human-readable training message."""


class ConsoleTrainingLogger:
    def info(self, message: str) -> None:
        print(message)


class NullTrainingLogger:
    def info(self, message: str) -> None:
        del message
