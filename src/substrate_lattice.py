import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

# Enable GPU execution
jax.config.update("jax_platform_name", "gpu")

import jax.numpy as jnp
from jax import jit


@jit
def update_lattice_memory_with_persistence(
        substrate_lattice_state, new_memory, gamma=0.9
):
    """
    Updates the substrate lattice by applying a moving average process.

    Args:
        substrate_lattice_state (jax.numpy.array): The persistent memory lattice.
        new_memory (jax.numpy.array): New imprint stored in memory.
        gamma (float): Memory persistence constant.

    Returns:
        jax.numpy.array: Updated lattice state with memory persistence.
    """
    # Memory field accumulation over time
    updated_lattice = gamma * substrate_lattice_state + (1 - gamma) * new_memory

    return updated_lattice


def apply_lattice_field_interaction(substrate_lattice_state, k, field, coupling_constant=0.05):
    """
    Modifies field behavior based on persistent substrate lattice imprints.
    """
    field_modification = jnp.exp(-k ** 2) * substrate_lattice_state * coupling_constant
    return field + field_modification


@jit
def process_lattice_interaction(substrate_lattice_state, k_grid, field_grid, energy_input_grid):
    """
    Parallel processing of substrate lattice interactions across all Fourier modes.

    Args:
        substrate_lattice_state (jax.numpy.array): Persistent lattice memory across grid.
        k_grid (jax.numpy.array): Grid of Fourier wavevectors.
        field_grid (jax.numpy.array): Grid of interacting fields.
        energy_input_grid (jax.numpy.array): Energy injected into the lattice per grid point.

    Returns:
        Updated substrate lattice state and modified field grid.
    """
    # Vectorized lattice memory update across all Fourier modes
    new_lattice_state = vmap(update_lattice_memory, in_axes=(0, 0, 0, 0))(
        substrate_lattice_state, k_grid, field_grid, energy_input_grid
    )

    # Apply non-local lattice memory influence on field evolution
    modified_field = vmap(apply_lattice_field_interaction, in_axes=(0, 0, 0))(
        new_lattice_state, k_grid, field_grid
    )

    return new_lattice_state, modified_field


def simulate_lattice_evolution(substrate_lattice_state, k_grid, field_grid, energy_input_grid, steps=100):
    """
    Simulates the evolution of the Substrate Lattice over multiple steps.

    Args:
        substrate_lattice_state (jax.numpy.array): Initial lattice memory state.
        k_grid (jax.numpy.array): Grid of Fourier wavevectors.
        field_grid (jax.numpy.array): Grid of interacting fields.
        energy_input_grid (jax.numpy.array): Energy injected into the lattice per grid point.
        steps (int): Number of simulation steps.

    Returns:
        Final substrate lattice state and field configuration.
    """

    def step_fn(carry, _):
        substrate_lattice_state, field_grid = carry
        substrate_lattice_state, field_grid = process_lattice_interaction(
            substrate_lattice_state, k_grid, field_grid, energy_input_grid
        )
        return (substrate_lattice_state, field_grid), None

    (final_lattice_state, final_field_grid), _ = lax.scan(step_fn, (substrate_lattice_state, field_grid), None,
                                                          length=steps)

    return final_lattice_state, final_field_grid


def initialize_lattice_grid(grid_size):
    """
    Initializes substrate lattice memory state and field grid for a given simulation size.

    Args:
        grid_size (int): Number of Fourier modes.

    Returns:
        Initialized substrate lattice state, k_grid, field_grid, and energy input grid.
    """
    k_grid = jnp.linspace(-5, 5, grid_size)  # Fourier wavevector grid
    substrate_lattice_state = jnp.zeros(grid_size)  # Initial lattice memory state
    field_grid = jnp.zeros(grid_size)  # Initial field state
    energy_input_grid = jnp.exp(-k_grid ** 2)  # Energy injection (Gaussian localized input)

    return substrate_lattice_state, k_grid, field_grid, energy_input_grid
