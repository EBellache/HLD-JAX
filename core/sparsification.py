import jax.numpy as jnp
from jax import jit, vmap


@jit
def apply_sparsification(data, threshold=0.05):
    """
    Applies structured sparsification by removing low-amplitude components.

    Args:
        data (jax.numpy.array): Input tensor to be sparsified.
        threshold (float): Minimum magnitude to retain elements.

    Returns:
        jax.numpy.array: Sparsified data.
    """
    return jnp.where(jnp.abs(data) > threshold, data, 0)


import jax.numpy as jnp
from jax import jit


@jit
def bistable_sparsification(data, low_threshold=0.05, high_threshold=0.15):
    """
    Applies bistable sparsification by pushing values toward one of two stable states.

    Args:
        data (jax.numpy.array): Input wavelet-transformed bioelectric signal.
        low_threshold (float): Lower threshold for suppression.
        high_threshold (float): Upper threshold for bistability.

    Returns:
        jax.numpy.array: Sparsified signal with forced bistability.
    """
    # Suppress low-energy states
    sparsified_data = jnp.where(jnp.abs(data) < low_threshold, 0, data)

    # Push data into one of two stable attractor states
    sparsified_data = jnp.where(jnp.abs(sparsified_data) > high_threshold,
                                jnp.sign(sparsified_data), sparsified_data)

    return sparsified_data


@jit
def adaptive_sparsification(data, entropy_field, scaling_factor=0.1):
    """
    Dynamically adjusts sparsification threshold based on entropy accumulation.

    Args:
        data (jax.numpy.array): Input tensor.
        entropy_field (jax.numpy.array): Local entropy measurement field.
        scaling_factor (float): Controls how entropy affects sparsification.

    Returns:
        jax.numpy.array: Adaptively sparsified data.
    """
    adaptive_threshold = scaling_factor * jnp.abs(entropy_field)
    return jnp.where(jnp.abs(data) > adaptive_threshold, data, 0)


@jit
def fibonacci_sparsification(data, phi=(1 + jnp.sqrt(5)) / 2, num_levels=5):
    """
    Applies a Fibonacci-based hierarchical sparsification method.

    Args:
        data (jax.numpy.array): Input tensor.
        phi (float): Golden ratio scaling factor.
        num_levels (int): Number of hierarchical sparsification levels.

    Returns:
        jax.numpy.array: Hierarchically sparsified data.
    """
    scales = jnp.array([phi ** (-i) for i in range(num_levels)])
    return vmap(lambda s: apply_sparsification(data * s))(scales).sum(axis=0)
