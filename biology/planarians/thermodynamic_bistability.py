import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jit
from  core.free_energy import compute_phase_entropy, compute_free_energy


# Define Bistable Potential Function
@jit
def bistable_potential(state, alpha=1.0, beta=-2.0):
    """
    Defines a bistable potential function for phase transitions.

    Args:
        state (jax.numpy.array): Morphogenetic state variable.
        alpha (float): Potential strength coefficient.
        beta (float): Bistability tuning coefficient.

    Returns:
        jax.numpy.array: Potential energy landscape.
    """
    return alpha * (state ** 4) + beta * (state ** 2)


# Compute Free Energy in a Bistable System
@jit
def bistable_free_energy(state, temperature=1.0):
    """
    Computes free energy of the bistable system.

    Args:
        state (jax.numpy.array): Morphogenetic state variable.
        temperature (float): System temperature controlling entropy.

    Returns:
        jax.numpy.array: Computed free energy.
    """
    U = bistable_potential(state)
    S = compute_phase_entropy(state)
    return compute_free_energy(U, S, sparsification_term=0.05)


# Generate Phase Transition Data
state_values = jnp.linspace(-2, 2, 100)
free_energy_values = bistable_free_energy(state_values)

# Visualization of the Bistable Potential and Free Energy
plt.figure(figsize=(10, 5))
plt.plot(state_values, bistable_potential(state_values), label='Bistable Potential', linestyle='dashed')
plt.plot(state_values, free_energy_values, label='Free Energy', linestyle='solid')
plt.xlabel('State Variable')
plt.ylabel('Energy')
plt.title('Thermodynamic Bistability: Potential and Free Energy')
plt.legend()
plt.show()

# Print Free Energy Results
print("Computed Free Energy Values:", free_energy_values)
