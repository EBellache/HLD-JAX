from jax import grad
from jax import jit
from jax import vmap
import jax.numpy as jnp

from contact_geometry.mqp_corrections import compute_mqp_force
from holography.fourier_projection import holographic_projection_mapping
from lie_algebra.non_abelian_mqp import corrected_non_abelian_forces


@jit
def geodesic_equation(x, v, metric_tensor, gauge_interaction):
    """
    Computes geodesic motion with projected holographic constraints.
    """
    # Compute MQP and gauge corrections in contact space
    F_mqp = compute_mqp_force(x, metric_tensor)
    F_gauge = corrected_non_abelian_forces(x, gauge_interaction)

    # Project these forces onto the holographic screen
    projected_force = holographic_projection_mapping(x, F_gauge + F_mqp)

    return -jnp.einsum("ijk,i,j", metric_tensor, v, v) + projected_force

@jit
def solve_geodesic(x0, v0, metric_tensor, gauge_interaction, num_steps=100, dt=0.01):
    """
    Solves the geodesic equation with non-Abelian forces using RK4 integration.
    """
    trajectory = [x0]
    x, v = x0, v0
    for _ in range(num_steps):
        k1 = dt * geodesic_equation(x, v, metric_tensor, gauge_interaction)
        k2 = dt * geodesic_equation(x + 0.5*dt*v, v + 0.5*k1, metric_tensor, gauge_interaction)
        k3 = dt * geodesic_equation(x + 0.5*dt*v, v + 0.5*k2, metric_tensor, gauge_interaction)
        k4 = dt * geodesic_equation(x + dt*v, v + k3, metric_tensor, gauge_interaction)
        v += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += dt * v
        trajectory.append(x)
    return jnp.array(trajectory)

@jit
def christoffel_symbols(metric_tensor):
    """
    Computes Christoffel symbols of the second kind for a given metric tensor.
    Args:
        metric_tensor: (n, n) array representing the metric of the projection space
    Returns:
        Christoffel symbols Γ^μ_{ρσ} as a (n, n, n) tensor
    """
    inv_metric = jnp.linalg.inv(metric_tensor)
    d_metric = vmap(grad(lambda g: g))(metric_tensor)  # Compute metric derivatives
    return 0.5 * (jnp.einsum("ml,rls->mrs", inv_metric, d_metric) +
                  jnp.einsum("ml,srl->mrs", inv_metric, d_metric) -
                  jnp.einsum("ml,rsr->mrs", inv_metric, d_metric))

