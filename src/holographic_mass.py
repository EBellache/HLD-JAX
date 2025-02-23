import jax.numpy as jnp
from jax import jit
from mqp_correction import compute_emergent_potential

LAMBDA_HOLO = 1.0  # Holographic coupling constant


@jit
def compute_holographic_mass(rho_k, hbar=1.0, m=1.0, lambda_holo=LAMBDA_HOLO):
    """
    Computes holographic mass as an emergent effect from MQP.

    Returns:
        jax.numpy.array: Holographic mass Mh(x) in real space.
    """
    rho_k = rho_k / jnp.max(rho_k)  # Normalize
    Q_k = compute_emergent_potential(rho_k, hbar, m)
    Mh_k = lambda_holo * Q_k

    return Mh_k
