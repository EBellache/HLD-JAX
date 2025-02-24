import jax.numpy as jnp
from jax import jit


@jit
def fibonacci_sparsification_mask(size, phi=(1 + jnp.sqrt(5)) / 2):
    """Generates a sparsification mask based on Fibonacci scaling."""
    indices = jnp.floor(phi ** jnp.arange(size)) % size
    mask = jnp.zeros(size)
    mask = mask.at[indices.astype(int)].set(1)
    return mask


@jit
def apply_fibonacci_sparsification(signal):
    """Applies Fibonacci sparsification to a given signal by masking certain frequencies."""
    size = signal.shape[0]
    mask = fibonacci_sparsification_mask(size)
    return signal * mask


@jit
def apply_wavelet_sparsification(wavelet_coeffs, threshold=0.1):
    """Applies sparsification by zeroing out low-magnitude wavelet coefficients."""
    abs_coeffs = jnp.abs(wavelet_coeffs)
    mask = jnp.where(abs_coeffs > threshold * jnp.max(abs_coeffs), 1, 0)
    return wavelet_coeffs * mask
