import jax.numpy as jnp
from jax import jit, grad, vmap, pmap


@jit
def compute_phase_entropy(phase_field):
    """Computes an entropy-based metric to track phase coherence in the tetrad lattice."""
    phase_distribution = jnp.abs(phase_field) / jnp.sum(jnp.abs(phase_field))
    entropy = -jnp.sum(phase_distribution * jnp.log(phase_distribution + 1e-8))
    return entropy


@jit
def compute_free_energy(U, S, sparsification_term):
    """Computes free energy with entropy and sparsification effects."""
    return U - S - sparsification_term
