import jax.numpy as jnp
from jax import jit, pmap


@jit
def holographic_projection(field, projection_matrix):
    """Computes holographic projection of a given field."""
    return jnp.dot(projection_matrix, field)


@jit
def compute_holographic_mass(soliton_field, lattice_density, coupling=1.0):
    """Computes the holographic mass of a soliton wave in the triadic lattice."""
    return coupling * jnp.sum(jnp.abs(soliton_field) ** 2 * lattice_density)


@jit
@pmap
def gauge_field_projection(field, gauge_group):
    """Projects gauge fields onto the triad lattice."""
    if gauge_group == "U(1)":
        return field
    elif gauge_group == "SU(2)":
        return jnp.cross(field, field)
    elif gauge_group == "SU(3)":
        return jnp.matmul(field, field) - jnp.trace(field) * jnp.eye(field.shape[0]) / field.shape[0]
    else:
        raise ValueError("Unsupported gauge group.")
