import jax
import jax.numpy as jnp
from jax import jit


@jit
def su_n_generator(n, i, j):
    """
    Generates SU(n) Lie algebra generator matrices in tensor format.
    Args:
        n: Group dimension
        i, j: Generator indices
    Returns:
        (n, n) Lie algebra matrix tensor
    """
    G = jnp.zeros((n, n))
    G = G.at[i, j].set(1.0)
    G = G.at[j, i].set(-1.0)
    return G

# Compute Lie brackets in a batched manner using vmap
@jit
def batched_lie_bracket(A, B):
    """
    Computes the batched Lie bracket [A, B] = AB - BA.
    Optimized for parallel execution on GPU/TPU.
    Args:
        A, B: Batched Lie algebra elements (batch_size, n, n)
    Returns:
        Batched Lie bracket result
    """
    return jax.vmap(lambda A, B: jnp.dot(A, B) - jnp.dot(B, A))(A, B)

# Exponential map using JAX's GPU-accelerated matrix exponentiation
@jit
def lie_exponential_map(G, t):
    """
    Computes exp(tG) for a Lie algebra element G.
    Uses JAX GPU-optimized matrix exponentiation.
    Args:
        G: Lie algebra generator tensor
        t: Scaling parameter
    Returns:
        Lie group element exp(tG)
    """
    return jax.scipy.linalg.expm(t * G)
