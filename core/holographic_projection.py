import jax.numpy as jnp
from jax import jit, grad, vmap, pmap


@jit
def holographic_projection(field, projection_matrix):
    """Computes holographic projection of a given field."""
    return jnp.dot(projection_matrix, field)


@jit
def compute_holographic_mass(soliton_field, lattice_density, coupling_constant=1.0):
    """Computes the holographic mass of a soliton wave interacting with the tetrad lattice."""
    energy_density = jnp.abs(soliton_field) ** 2
    return coupling_constant * jnp.sum(energy_density * lattice_density)

@jit
@pmap
def gauge_field_projection(field, gauge_group):
    """Projects gauge fields onto the tetrad lattice based on symmetry group, optimized for parallel GPU execution."""
    if gauge_group == "U(1)":
        return field  # Abelian fields pass through unchanged
    elif gauge_group == "SU(2)":
        return jnp.cross(field, field)  # Introduces self-interaction effects
    elif gauge_group == "SU(3)":
        return jnp.matmul(field, field) - jnp.trace(field) * jnp.eye(field.shape[0]) / field.shape[0]  # Confinement effects
    else:
        raise ValueError("Unsupported gauge group.")