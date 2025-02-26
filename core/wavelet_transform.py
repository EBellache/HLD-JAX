import jax.numpy as jnp
from jax import jit, vmap, lax


def body_fun(carry, _):
    x, y = carry
    return (y, x + y), _


@jit
def fib(n):
    _, (fib_n_minus_1, fib_n) = lax.scan(body_fun, (0, 1), None, length=n)
    return fib_n


@jit
def fibonacci_scales(num_scales, phi=(1 + jnp.sqrt(5)) / 2):
    """Generate Fibonacci-scaled wavelet scales."""
    return jnp.floor(phi ** jnp.arange(num_scales))


@jit
def standard_wavelet_transform(signal, wavelet_function, num_scales=10):
    """
    Computes a standard wavelet transform of a given signal.

    Args:
        signal (jax.numpy.array): Input bioelectric signal.
        wavelet_function (function): Wavelet basis function to be used.
        num_scales (int): Number of decomposition scales.

    Returns:
        jax.numpy.array: Wavelet-transformed coefficients.
    """
    transformed_signal = []

    for j in range(num_scales):
        scale = 2 ** j  # Standard dyadic wavelet scaling
        wavelet_coefficients = wavelet_function(scale * signal)
        transformed_signal.append(wavelet_coefficients)

    return jnp.array(transformed_signal)


@jit
def fibonacci_wavelet_transform(signal, wavelet_function, num_scales=10):
    """Computes wavelet transform using Fibonacci-scaled wavelets."""
    scales = fibonacci_scales(num_scales)
    return vmap(lambda s: wavelet_function(signal / s))(scales)


@jit
def hyperbolic_wavelet_transform(signal, num_scales=10, alpha=0.7):
    """
    Computes a hyperbolic wavelet transform with Fibonacci-based scaling.

    Args:
        signal (jax.numpy.array): Input bioelectric signal.
        num_scales (int): Number of decomposition scales.
        alpha (float): Anisotropy parameter (0 < alpha < 2).

    Returns:
        jax.numpy.array: Wavelet-transformed coefficients.
    """
    transformed_signal = []

    for j in range(num_scales):
        scale_x = fib(j) ** alpha
        scale_y = fib(j) ** (2 - alpha)

        wavelet_x = jnp.exp(-scale_x * jnp.abs(signal)) * jnp.cos(scale_x * signal)
        wavelet_y = jnp.exp(-scale_y * jnp.abs(signal)) * jnp.cos(scale_y * signal)

        transformed_signal.append(wavelet_x * wavelet_y)

    return jnp.array(transformed_signal)
