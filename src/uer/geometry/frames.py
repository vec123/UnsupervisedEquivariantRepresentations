from __future__ import annotations

import jax.numpy as jnp


def orthogonal_basis(v):
    condition = jnp.abs(v[..., 0]) < 0.9
    helper = jnp.where(
        condition[..., None],
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
    )
    u = jnp.cross(v, helper)
    u = u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + 1e-8)
    w = jnp.cross(v, u)
    return u, w


def rotation_matrix_from_two_vectors(v1, v2):
    u = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    dot = jnp.einsum("bi,bi->b", u, v2)[..., None]
    w_raw = v2 - dot * u
    w = w_raw / (jnp.linalg.norm(w_raw, axis=-1, keepdims=True) + 1e-8)
    _, fallback_w = orthogonal_basis(u)
    is_degenerate = jnp.linalg.norm(w_raw, axis=-1) < 1e-4
    w = jnp.where(is_degenerate[..., None], fallback_w, w)
    last_v = jnp.cross(u, w)
    return jnp.stack([u, w, last_v], axis=-1)
