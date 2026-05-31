from __future__ import annotations

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm

from uer.trainers.logging import ConsoleTrainingLogger, TrainingLogger


@dataclass(frozen=True)
class ClassificationResult:
    params: dict
    opt_state: optax.OptState
    accuracy: float


class ClassificationTrainer:
    def __init__(
        self,
        model,
        learning_rate: float = 1e-2,
        seed: int = 3,
        logger: TrainingLogger | None = None,
    ):
        self.model = model
        self.opt = optax.adam(learning_rate)
        self.seed = seed
        self.logger = logger or ConsoleTrainingLogger()

    def fit(self, graphs, steps: int = 200) -> ClassificationResult:
        def loss_fn(params, batch):
            logits = self.model.apply(params, batch)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.globals)
            return jnp.mean(loss), logits

        @jax.jit
        def update_fn(params, opt_state, batch):
            grad_fn = jax.grad(loss_fn, has_aux=True)
            grads, logits = grad_fn(params, batch)
            accuracy = jnp.mean(jnp.argmax(logits, axis=1) == batch.globals)
            updates, opt_state = self.opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, accuracy

        params = jax.jit(self.model.init)(jax.random.PRNGKey(self.seed), graphs)
        opt_state = self.opt.init(params)

        wall = time.perf_counter()
        _, _, accuracy = update_fn(params, opt_state, graphs)
        self.logger.info(f"initial accuracy = {100 * accuracy:.0f}%")
        self.logger.info(f"compilation took {time.perf_counter() - wall:.1f}s")

        for _ in tqdm(range(steps)):
            params, opt_state, accuracy = update_fn(params, opt_state, graphs)
            if accuracy == 1.0:
                break

        self.logger.info(f"final accuracy = {100 * accuracy:.0f}%")
        return ClassificationResult(params=params, opt_state=opt_state, accuracy=float(accuracy))
