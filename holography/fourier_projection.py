import jax.numpy as jnp
from jax import jit

@jit
def holographic_projection(x, t, params):
    """
    Computes the holographic projection function as a Fourier sum.
    Args:
        x: Spatial coordinate
        t: Time coordinate
        params: Tuple of projection parameters (A_k, k_values, phi_t)
    Returns:
        Projection intensity at (x, t)
    """
    A_k, k_values, phi_t = params
    projection = jnp.sum(A_k * jnp.sin(k_values * x + phi_t * t))
    return projection

@jit
def compute_quantum_foam(phase_field):
    """
    Computes quantum foam effects from holographic interference.
    Args:
        phase_field: Fourier phase representation of the projection
    Returns:
        Quantum artifact intensity map
    """
    return jnp.abs(jnp.fft.ifft(phase_field)) ** 2
