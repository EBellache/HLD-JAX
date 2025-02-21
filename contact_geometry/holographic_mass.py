import jax.numpy as jnp
from jax import jit, grad
from quantum_artifacts import compute_mqp

@jit
def compute_holographic_mass(x, lambda_holo=1.0):
    """
    Computes effective mass as a function of the Macroscopic Quantum Potential (MQP).
    Args:
        x: Position in projection space
        lambda_holo: Holographic coupling constant
    Returns:
        Mass-like correction term
    """
    mqp_potential = compute_mqp(x)
    return lambda_holo * mqp_potential
