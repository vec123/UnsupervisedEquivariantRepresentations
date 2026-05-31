from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from uer.geometry.transforms import apply_batched_transform, random_se3
from uer.trainers.logging import ConsoleTrainingLogger, TrainingLogger


@dataclass(frozen=True)
class AugmentationConfig:
    translation_range: float = 4.0
    permute_nodes: bool = False


@dataclass(frozen=True)
class ReconstructionMetrics:
    step: int
    loss: float
    invariant_delta: float
    frame_delta: float
    translation_delta: float


@dataclass(frozen=True)
class ReconstructionResult:
    state: train_state.TrainState
    predictions: jnp.ndarray
    history: list[ReconstructionMetrics] = field(default_factory=list)


class ReconstructionTrainer:
    def __init__(
        self,
        encoder,
        decoder,
        radius: float,
        learning_rate: float = 1e-3,
        augmentation: AugmentationConfig | None = None,
        seed: int = 0,
        logger: TrainingLogger | None = None,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.radius = radius
        self.learning_rate = learning_rate
        self.augmentation = augmentation or AugmentationConfig()
        self.seed = seed
        self.logger = logger or ConsoleTrainingLogger()

    def apply_group_action(self, canonical_positions, rotation_frame, translation_frame):
        rotated = jnp.einsum("bij,bkj->bki", rotation_frame, canonical_positions)
        return rotated + translation_frame[:, jnp.newaxis, :]

    def loss_fn(self, params, graph):
        invariant, rotation_frame, _, translation_frame = self.encoder.apply(
            {"params": params["encoder"]},
            graph,
        )
        canonical_positions = self.decoder.apply(
            {"params": params["decoder"]},
            invariant.array,
        )
        pred_positions = self.apply_group_action(
            canonical_positions,
            rotation_frame,
            translation_frame,
        )
        target_positions = graph.nodes.reshape(pred_positions.shape)
        loss = jnp.mean(jnp.square(pred_positions - target_positions))
        return loss, (pred_positions, canonical_positions, invariant, rotation_frame, translation_frame)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, graph):
        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params, graph)
        return state.apply_gradients(grads=grads), loss, aux

    def initialize(self, graphs):
        rng = jax.random.PRNGKey(self.seed)
        rng, encoder_key, decoder_key = jax.random.split(rng, 3)
        encoder_vars = self.encoder.init(encoder_key, graphs)
        invariant, _, _, _ = self.encoder.apply(encoder_vars, graphs)
        decoder_vars = self.decoder.init(decoder_key, invariant.array)
        params = {
            "encoder": encoder_vars["params"],
            "decoder": decoder_vars["params"],
        }
        state = train_state.TrainState.create(
            apply_fn=None,
            params=params,
            tx=optax.adam(self.learning_rate),
        )
        return rng, state

    def fit(self, graphs, num_steps: int = 1000, log_every: int = 100) -> ReconstructionResult:
        rng, state = self.initialize(graphs)
        latest_predictions = graphs.nodes.reshape((int(graphs.n_node.shape[0]), -1, 3))
        history = []
        self.logger.info(f"Starting reconstruction training for {num_steps} steps...")

        for step in range(num_steps):
            rng, aug_key, transform_key = jax.random.split(rng, 3)
            batch_size = int(graphs.n_node.shape[0])
            rotations, translations = random_se3(
                aug_key,
                batch_size,
                translation_range=self.augmentation.translation_range,
            )
            graphs_aug = apply_batched_transform(
                transform_key,
                graphs,
                rotations,
                translations,
                radius=self.radius,
                permute=self.augmentation.permute_nodes,
            )
            state, loss, aux = self.train_step(state, graphs_aug)
            latest_predictions, _, _, _, _ = aux

            if step % log_every == 0 or step == num_steps - 1:
                metrics = self.evaluate_equivariance(state.params, graphs, graphs_aug, rotations, translations)
                metrics = ReconstructionMetrics(
                    step=step,
                    loss=float(loss),
                    invariant_delta=metrics.invariant_delta,
                    frame_delta=metrics.frame_delta,
                    translation_delta=metrics.translation_delta,
                )
                history.append(metrics)
                self.logger.info(
                    f"Step {step:4d} | Loss: {metrics.loss:.6f} | "
                    f"Inv: {metrics.invariant_delta:.2e} | "
                    f"Frame: {metrics.frame_delta:.2e} | "
                    f"Transl: {metrics.translation_delta:.2e}"
                )

        return ReconstructionResult(state=state, predictions=latest_predictions, history=history)

    def evaluate_equivariance(self, params, original_graphs, augmented_graphs, rotations, translations):
        inv_orig, rotation_orig, _, translation_orig = self.encoder.apply(
            {"params": params["encoder"]},
            original_graphs,
        )
        inv_aug, rotation_aug, _, translation_aug = self.encoder.apply(
            {"params": params["encoder"]},
            augmented_graphs,
        )
        invariant_delta = jnp.mean(jnp.abs(inv_orig.array - inv_aug.array))
        rotation_expected = jnp.einsum("bij,bjk->bik", rotations, rotation_orig)
        frame_delta = jnp.mean(jnp.abs(rotation_aug - rotation_expected))
        translation_expected = jnp.einsum("bij,bj->bi", rotations, translation_orig) + translations
        translation_delta = jnp.mean(jnp.abs(translation_aug - translation_expected))
        return ReconstructionMetrics(
            step=-1,
            loss=0.0,
            invariant_delta=float(invariant_delta),
            frame_delta=float(frame_delta),
            translation_delta=float(translation_delta),
        )
