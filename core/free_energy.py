import jax.numpy as jnp
from jax import jit


@jit
def compute_phase_entropy(phase_field):
    """Computes entropy-based phase coherence in the triadic lattice."""
    phase_distribution = jnp.abs(phase_field) / jnp.sum(jnp.abs(phase_field))
    return -jnp.sum(phase_distribution * jnp.log(phase_distribution + 1e-8))


@jit
def compute_free_energy(U, S, sparsification_term):
    """Computes free energy considering entropy and sparsification effects."""
    return U - S - sparsification_term
