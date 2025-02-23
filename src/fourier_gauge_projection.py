import jax.numpy as jnp


def etch_gauge_field_on_tqf(k, gauge_field, TQF_pressure):
    """
    Simulates how gauge fields imprint onto the time-like quantum fluid.

    Args:
        k (jax.numpy.array): Fourier mode wavevector.
        gauge_field (jax.numpy.array): Gauge field structure in Fourier space.
        TQF_pressure (float): Pressure exerted by the time-like quantum fluid.

    Returns:
        The etched projection of the gauge field onto the TQF.
    """
    return gauge_field * jnp.exp(-TQF_pressure * k ** 2)
