import jax.numpy as jnp


def enforce_phase_constraints(k, projection_surface):
    """
    Enforces holographic phase constraints for projection fidelity.

    Args:
        k (jax.numpy.array): Fourier mode wavevector.
        projection_surface (jax.numpy.array): The holographic projection screen.

    Returns:
        Filtered projection enforcing phase constraints.
    """
    return jnp.exp(-k ** 2) * projection_surface  # Applies Gaussian smoothing
