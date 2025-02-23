import jax.numpy as jnp
from jax import jit


@jit
def compute_emergent_potential(rho_k, hbar=1.0, m=1.0):
    """
    Computes the Holographic Potential (MQP) in Fourier space.

    Args:
        rho_k (jax.numpy.array): Fourier transform of probability density.
        hbar (float): Reduced Planck's constant.
        m (float): Effective mass.

    Returns:
        Holographic potential Q in Fourier space.
    """
    if rho_k.shape[0] < 2:
        raise ValueError("rho_k must have more than one element for Fourier transform.")

    rho_k = rho_k / jnp.max(rho_k)  # Normalize
    k2 = jnp.abs(jnp.fft.fftfreq(rho_k.shape[0])) ** 2
    Q_k = - (hbar ** 2 / (2 * m)) * (k2 * rho_k / (rho_k + 1e-8))
    return Q_k


@jit
def compute_mqp_with_equilibrium_constraint(
    rho_k, hbar=1.0, m=1.0, gamma=0.9
):
    """
    Computes MQP with a constraint that pulls the system towards equilibrium.

    Args:
        rho_k (jax.numpy.array): Fourier transform of probability density.
        hbar (float): Reduced Planck's constant.
        m (float): Effective mass.
        gamma (float): Memory persistence factor (determines deviation from equilibrium).

    Returns:
        jax.numpy.array: MQP corrected for memory-induced free energy effects.
    """
    k2 = jnp.abs(jnp.fft.fftfreq(rho_k.shape[0])) ** 2

    # Compute standard MQP
    Q_k = - (hbar ** 2 / (2 * m)) * (k2 * rho_k / (rho_k + 1e-8))

    # Add equilibrium restoring term
    equilibrium_correction = gamma * jnp.abs(Q_k)

    return Q_k - equilibrium_correction
