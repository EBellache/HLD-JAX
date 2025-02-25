import jax.numpy as jnp
from jax import jit, pmap


@jit
def update_triad_lattice(lattice, deformation_field, phase_field, alpha=0.1, phase_inc=0.01):
    """Updates triadic lattice under deformation with U(1) phase accumulation."""
    phase_shift = jnp.exp(1j * phase_inc)
    return pmap(lambda ts, df, pf: phase_shift * (ts + alpha * df + pf))(lattice, deformation_field, phase_field)
