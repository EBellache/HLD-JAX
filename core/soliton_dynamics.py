import jax.numpy as jnp
from jax import jit, grad, vmap, pmap


@jit
def soliton_wave_equation(S, v=1.0, g=0.1, dt=0.01):
    """Evolves a solitonic wave equation in a tetradic lattice."""
    laplacian = jnp.gradient(jnp.gradient(S))
    torsion_term = g * jnp.cross(S, jnp.gradient(S))
    return S + dt * (v ** 2 * laplacian + torsion_term)
