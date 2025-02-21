import jax.numpy as jnp
from jax import grad
from jax import jit
from jax import vmap

from contact_geometry.holographic_mass import compute_holographic_mass
from contact_geometry.mqp_corrections import compute_mqp_force
from holography.fourier_projection import holographic_projection_mapping
from lie_algebra.non_abelian_mqp import corrected_non_abelian_forces


@jit
def geodesic_equation(x, v, metric_tensor, gauge_interaction):
    """
    Computes geodesic motion with projected holographic constraints,
    incorporating holographic mass corrections.

    Args:
        x: Position in space
        v: Velocity
        metric_tensor: Projection space metric
        gauge_interaction: Non-Abelian gauge force tensor
    Returns:
        Acceleration due to curvature, MQP, gauge interactions, and holographic mass effects.
    """
    # Compute MQP and gauge corrections in contact space
    F_mqp = compute_mqp_force(x, metric_tensor)
    F_gauge = corrected_non_abelian_forces(x, gauge_interaction)

    # Compute Holographic Mass Correction (acts as an inertia-like effect)
    M_holo = compute_holographic_mass(x)

    # Project these forces onto the holographic screen
    projected_force = holographic_projection_mapping(x, F_gauge + F_mqp)

    # Apply Holographic Mass as a scaling factor on force corrections
    return (-jnp.einsum("ijk,i,j", metric_tensor, v, v) + projected_force) / (1 + M_holo)




@jit
def solve_geodesic(x0, v0, metric_tensor, gauge_interaction, num_steps=100, dt=0.01):
    """
    Solves the geodesic equation with non-Abelian forces using RK4 integration,
    incorporating holographic mass as a dynamic correction.

    Args:
        x0: Initial position
        v0: Initial velocity
        metric_tensor: Projection space metric
        gauge_interaction: Non-Abelian gauge force tensor
        num_steps: Number of integration steps
        dt: Time step size
    Returns:
        Geodesic trajectory with holographic mass corrections.
    """
    trajectory = [x0]
    x, v = x0, v0
    for _ in range(num_steps):
        # Compute local mass correction
        M_holo = compute_holographic_mass(x)

        # RK4 integration with mass-scaling
        k1 = dt * geodesic_equation(x, v, metric_tensor, gauge_interaction) / (1 + M_holo)
        k2 = dt * geodesic_equation(x + 0.5 * dt * v, v + 0.5 * k1, metric_tensor, gauge_interaction) / (1 + M_holo)
        k3 = dt * geodesic_equation(x + 0.5 * dt * v, v + 0.5 * k2, metric_tensor, gauge_interaction) / (1 + M_holo)
        k4 = dt * geodesic_equation(x + dt * v, v + k3, metric_tensor, gauge_interaction) / (1 + M_holo)

        v += (k1 + 2 * k2 + 2 * k3 + k4) / 6
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

