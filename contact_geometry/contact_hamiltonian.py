import jax.numpy as jnp
from jax import grad, jit

@jit
def contact_hamiltonian(q, p, S, params):
    """
    Computes the contact Hamiltonian evolution equations.
    Args:
        q: Generalized coordinate
        p: Generalized momentum
        S: Projection constraint variable
        params: Tuple (H0, lambda_factor)
    Returns:
        (dq/dt, dp/dt, dS/dt)
    """
    H0, lambda_factor = params
    H = H0 + lambda_factor * S
    dq_dt = jnp.gradient(H, p)
    dp_dt = -jnp.gradient(H, q) + lambda_factor * p
    dS_dt = p * dq_dt - H
    return dq_dt, dp_dt, dS_dt
