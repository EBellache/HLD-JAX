import jax.numpy as jnp
from jax import jit, grad, vmap
from lie_algebra.lie_tensor_algebra import su_n_generator, batched_lie_bracket


@jit
def gauge_force_tensor(n):
    """
    Computes the Lie algebra force tensors for SU(N) gauge interactions.
    Args:
        n: Dimension of gauge group (e.g., N=3 for SU(3) QCD, N=2 for weak force)
    Returns:
        Tensor representation of gauge force structure
    """
    return jnp.stack([su_n_generator(n, i, j) for i in range(n) for j in range(n)])


@jit
def compute_gauge_interactions(force_tensors, coupling_constants):
    """
    Computes interaction terms for force carriers based on symmetry constraints.
    Args:
        force_tensors: Lie algebra tensors for gauge groups
        coupling_constants: Array of gauge coupling strengths (e.g., α_S, α_W, α_EM)
    Returns:
        Effective interaction tensor
    """
    interaction_tensor = jnp.sum(
        jnp.array([c * batched_lie_bracket(force_tensors, force_tensors)
                   for c in coupling_constants]), axis=0)
    return interaction_tensor
