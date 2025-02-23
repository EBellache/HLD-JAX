import jax.numpy as jnp


def alternate_tqf_coupling(k, field, coupling_constant):
    """
    Tests a scenario where TQF directly interacts with gauge fields instead of acting as a substrate.

    Args:
        k (jax.numpy.array): Fourier wavevector.
        field (jax.numpy.array): The interacting field.
        coupling_constant (float): Strength of direct coupling.

    Returns:
        Modified field that now interacts directly with TQF.
    """
    return field + coupling_constant * k
