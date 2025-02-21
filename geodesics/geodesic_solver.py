import jax.numpy as jnp
from jax import jit, grad, vmap
from holography.quantum_artifacts import compute_mqp

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

@jit
def geodesic_equation(x, v, metric_tensor):
    """
    Computes the right-hand side of the geodesic equation, incorporating MQP.
    Args:
        x: Position in projection space
        v: Velocity along the geodesic
        metric_tensor: Metric defining the space
    Returns:
        Acceleration due to curvature and MQP force
    """
    gamma = christoffel_symbols(metric_tensor)
    mqp_force = compute_mqp(x)  # Compute MQP force dynamically
    return -jnp.einsum("ijk,i,j", gamma, v, v) + mqp_force

@jit
def solve_geodesic(x0, v0, metric_tensor, num_steps=100, dt=0.01):
    """
    Solves the geodesic equation using numerical integration (RK4 method).
    Args:
        x0: Initial position
        v0: Initial velocity
        metric_tensor: Metric defining projection space
        num_steps: Number of integration steps
        dt: Step size
    Returns:
        Array of geodesic trajectory points
    """
    trajectory = [x0]
    x, v = x0, v0
    for _ in range(num_steps):
        k1 = dt * geodesic_equation(x, v, metric_tensor)
        k2 = dt * geodesic_equation(x + 0.5*dt*v, v + 0.5*k1, metric_tensor)
        k3 = dt * geodesic_equation(x + 0.5*dt*v, v + 0.5*k2, metric_tensor)
        k4 = dt * geodesic_equation(x + dt*v, v + k3, metric_tensor)
        v += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += dt * v
        trajectory.append(x)
    return jnp.array(trajectory)
