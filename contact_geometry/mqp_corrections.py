import jax.numpy as jnp
from jax import jit, grad
from holography.quantum_artifacts import compute_mqp
from geodesics.geodesic_solver import christoffel_symbols

@jit
def compute_mqp_force(x, metric_tensor):
    """
    Computes the MQP correction term as a gravity-like force.
    Args:
        x: Position in projection space
        metric_tensor: Metric defining curvature
    Returns:
        MQP force correction F_MQP
    """
    mqp_potential = compute_mqp(x)
    gamma = christoffel_symbols(metric_tensor)  # Compute Christoffel symbols
    return -jnp.einsum("ijk,j", gamma, grad(mqp_potential))

@jit
def corrected_gauge_force(x, metric_tensor, gauge_interaction):
    """
    Computes the gauge force corrected by MQP gravity-like effects.
    Args:
        x: Position in space
        metric_tensor: Projection space metric
        gauge_interaction: Standard Model gauge interaction tensor
    Returns:
        Gauge force corrected by MQP
    """
    F_mqp = compute_mqp_force(x, metric_tensor)
    return gauge_interaction + F_mqp
