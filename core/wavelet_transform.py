import jax.numpy as jnp
from jax import jit, grad, vmap, pmap


@jit
def fibonacci_scales(num_scales, phi=(1 + jnp.sqrt(5)) / 2):
    """Generate Fibonacci-scaled wavelet scales."""
    return jnp.floor(phi ** jnp.arange(num_scales))


@jit
def fibonacci_wavelet_transform(signal, wavelet_function, num_scales=10):
    """Computes wavelet transform using Fibonacci-scaled wavelets."""
    scales = fibonacci_scales(num_scales)
    return vmap(lambda s: wavelet_function(signal / s))(scales)
