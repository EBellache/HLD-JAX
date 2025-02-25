import jax.numpy as jnp
from jax import jit


@jit
def soliton_wave_equation(S, v=1.0, g=0.1, dt=0.01):
    """Evolves solitonic wave in a triadic lattice."""
    laplacian = jnp.gradient(jnp.gradient(S))
    torsion_term = g * jnp.cross(S, jnp.gradient(S))
    return S + dt * (v ** 2 * laplacian + torsion_term)
