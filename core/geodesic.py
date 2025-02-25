import jax.numpy as jnp
from jax import jit, grad, vmap

@jit
def compute_geodesic(initial_state, metric_tensor, steps=100, step_size=0.01):
    """
    Computes geodesic trajectory given a metric tensor in the triad lattice.

    Args:
        initial_state (jax.numpy.array): Initial position and velocity.
        metric_tensor (function): Function computing the metric tensor at a given state.
        steps (int): Number of integration steps.
        step_size (float): Step size for numerical integration.

    Returns:
        jax.numpy.array: Computed geodesic trajectory.
    """
    def equation(state):
        pos, vel = state[:3], state[3:]
        metric = metric_tensor(pos)
        christoffel = -0.5 * jnp.einsum('ijk,j->ik', grad(metric, argnums=0)(pos), vel)
        return jnp.concatenate([vel, christoffel @ vel])

    trajectory = [initial_state]
    state = initial_state
    for _ in range(steps):
        state = state + step_size * equation(state)
        trajectory.append(state)
    return jnp.array(trajectory)


@jit
def compute_smearing_effect(geodesic_trajectory, phase_field, smearing_coefficient=0.05):
    """
    Computes the smearing effect on geodesic trajectories due to phase accumulation.

    Args:
        geodesic_trajectory (jax.numpy.array): Array of geodesic points.
        phase_field (jax.numpy.array): Phase accumulation field.
        smearing_coefficient (float): Strength of the smearing effect.

    Returns:
        jax.numpy.array: Smearing-affected geodesic.
    """
    phase_gradient = jnp.gradient(phase_field)
    return geodesic_trajectory + smearing_coefficient * phase_gradient[:geodesic_trajectory.shape[0]]


@jit
def apply_perturbation(geodesic_trajectory, perturbation_field, perturbation_strength=0.1):
    """
    Applies external perturbations to geodesic trajectories.

    Args:
        geodesic_trajectory (jax.numpy.array): Array of geodesic points.
        perturbation_field (jax.numpy.array): External perturbation field.
        perturbation_strength (float): Magnitude of the perturbation.

    Returns:
        jax.numpy.array: Perturbed geodesic trajectory.
    """
    perturbation = perturbation_strength * perturbation_field[:geodesic_trajectory.shape[0]]
    return geodesic_trajectory + perturbation
