import jax.numpy as jnp
from jax import jit
from contact_geometry.contact_forces import compute_gauge_interactions
from contact_geometry.mqp_corrections import corrected_gauge_force


@jit
def project_mqp_corrected_force(x, metric_tensor, gauge_interaction, projection_function):
    """
    Projects MQP-corrected gauge interactions into the holographic substrate.
    Args:
        x: Position in projection space
        metric_tensor: Metric defining projection constraints
        gauge_interaction: Standard Model gauge interaction tensor
        projection_function: Holographic mapping function
    Returns:
        Projected interaction with MQP corrections
    """
    corrected_force = corrected_gauge_force(x, metric_tensor, gauge_interaction)
    return projection_function(corrected_force)


@jit
def project_force_to_hologram(interaction_tensor, projection_function):
    """
    Projects gauge interactions from contact geometry space into the holographic substrate.
    Args:
        interaction_tensor: Tensor of gauge interactions from Lie algebra constraints
        projection_function: Holographic mapping function
    Returns:
        Projected force field in holographic space
    """
    return projection_function(interaction_tensor)


@jit
def holographic_projection_mapping(x, interaction_tensor):
    """
    Defines the mapping of gauge interactions onto the holographic Fourier basis.
    Args:
        x: Position in holographic space
        interaction_tensor: Tensor from contact geometry
    Returns:
        Fourier-projected interaction effects
    """
    return jnp.sum(interaction_tensor * jnp.sin(x))
