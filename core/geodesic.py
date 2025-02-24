import jax.numpy as jnp
from jax import jit, grad


@jit
def compute_geodesic_trajectory(initial_state, metric_tensor, timesteps=100, step_size=0.01):
    """
    Computes geodesic trajectory given an initial state and metric tensor.

    Args:
        initial_state (jax.numpy.array): Initial position and velocity.
        metric_tensor (function): Function that computes the metric tensor at a given state.
        timesteps (int): Number of steps to compute.
        step_size (float): Integration step size.

    Returns:
        jax.numpy.array: Geodesic trajectory over time.
    """

    def geodesic_equation(state):
        position, velocity = state[:len(state) // 2], state[len(state) // 2:]
        metric = metric_tensor(position)
        christoffel = -0.5 * jnp.einsum('ijk,j->ik', grad(metric, argnums=0)(position), velocity)
        return jnp.concatenate([velocity, christoffel @ velocity])

    trajectory = [initial_state]
    state = initial_state
    for _ in range(timesteps):
        state = state + step_size * geodesic_equation(state)
        trajectory.append(state)
    return jnp.array(trajectory)


@jit
def apply_perturbation(trajectory, perturbation):
    """
    Applies an external perturbation to the geodesic trajectory.

    Args:
        trajectory (jax.numpy.array): Original geodesic trajectory.
        perturbation (function): Function defining the perturbation over time.

    Returns:
        jax.numpy.array: Perturbed trajectory.
    """
    return trajectory + perturbation(jnp.arange(len(trajectory)))


@jit
def compute_smearing_effect(trajectory, diffusion_coefficient=0.05):
    """
    Computes smearing effects due to entropy accumulation along a geodesic.

    Args:
        trajectory (jax.numpy.array): Geodesic trajectory.
        diffusion_coefficient (float): Strength of diffusion applied over time.

    Returns:
        jax.numpy.array: Smeared trajectory.
    """
    noise = diffusion_coefficient * jnp.cumsum(jnp.random.normal(size=trajectory.shape), axis=0)
    return trajectory + noise


@jit
def compute_su2_su3_geodesics(initial_state, su2_torsion_field, su3_helical_dislocation, timesteps=100, step_size=0.01):
    """
    Computes geodesic trajectory under SU(2) and SU(3) gauge interactions using Loop Quantum Gravity principles.

    Args:
        initial_state (jax.numpy.array): Initial position and velocity.
        su2_torsion_field (function): Function defining SU(2) torsion in the lattice.
        su3_helical_dislocation (function): Function defining SU(3) helical curvature and dislocation.
        timesteps (int): Number of steps to compute.
        step_size (float): Integration step size.

    Returns:
        jax.numpy.array: Geodesic trajectory over time.
    """

    def geodesic_equation(state):
        position, velocity = state[:len(state) // 2], state[len(state) // 2:]
        torsion_correction = su2_torsion_field(position) @ velocity
        helical_dislocation = su3_helical_dislocation(position) @ velocity
        return jnp.concatenate([velocity, torsion_correction + helical_dislocation])

    trajectory = [initial_state]
    state = initial_state
    for _ in range(timesteps):
        state = state + step_size * geodesic_equation(state)
        trajectory.append(state)
    return jnp.array(trajectory)
