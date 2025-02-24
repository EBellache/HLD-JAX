import jax.numpy as jnp
from core.substrate_lattice import (
    update_lattice_memory,
    apply_lattice_field_interaction,
    process_lattice_interaction,
    initialize_lattice_grid,
)


def test_update_lattice_memory():
    lattice_state = jnp.zeros(10)
    k = jnp.linspace(-1, 1, 10)
    field = jnp.ones(10)
    energy_input = jnp.full(10, 0.1)

    new_state = update_lattice_memory(lattice_state, k, field, energy_input)

    assert new_state.shape == (10,)


def test_apply_lattice_field_interaction():
    lattice_state = jnp.ones(10)
    k = jnp.linspace(-1, 1, 10)
    field = jnp.zeros(10)

    modified_field = apply_lattice_field_interaction(lattice_state, k, field)

    assert modified_field.shape == (10,)


def test_initialize_lattice_grid():
    lattice_state, k_grid, field_grid, energy_input_grid = initialize_lattice_grid(10)
    assert lattice_state.shape == (10,)
    assert k_grid.shape == (10,)
    assert field_grid.shape == (10,)
    assert energy_input_grid.shape == (10,)
