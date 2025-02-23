import jax.numpy as jnp
from jax import jit


@jit
def compute_holographic_potential(rho_k, hbar=1.0, m=1.0):
    """
    Computes the Holographic Potential (MQP) in Fourier space.

    Args:
        rho_k (jax.numpy.array): Fourier transform of probability density.
        hbar (float): Reduced Planck's constant.
        m (float): Effective mass.

    Returns:
        Holographic potential Q in Fourier space.
    """
    k2 = jnp.abs(jnp.fft.fftfreq(rho_k.shape[0])) ** 2  # Square of wavevector magnitude
    Q_k = - (hbar ** 2 / (2 * m)) * (k2 * rho_k / (rho_k + 1e-8))  # Bohmian-style correction
    return Q_k
