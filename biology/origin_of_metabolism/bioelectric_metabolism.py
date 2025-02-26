import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt


@jit
def bioelectric_field(time, initial_potential=1.0, decay_rate=0.005):
    """
    Models the emergence of a bioelectric field as the first computational layer.
    """
    return initial_potential * jnp.exp(-decay_rate * time)


@jit
def krebs_cycle_activation(bioelectric_potential, k_activation=0.1):
    """
    Models the activation of an early metabolic cycle as a response to bioelectric gradients.
    """
    return k_activation * bioelectric_potential / (1 + bioelectric_potential)


@jit
def atp_production(metabolic_activity, efficiency=0.8):
    """
    Simulates ATP production as an output of early metabolism.
    """
    return efficiency * metabolic_activity


# Simulate the time evolution
time_steps = jnp.linspace(0, 500, 500)
bioelectric_potentials = bioelectric_field(time_steps)
metabolic_activity = krebs_cycle_activation(bioelectric_potentials)
atp_levels = atp_production(metabolic_activity)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(time_steps, bioelectric_potentials, label="Bioelectric Field (U(1))", linestyle="solid")
plt.plot(time_steps, metabolic_activity, label="Metabolic Activation (Krebs)", linestyle="dashed")
plt.plot(time_steps, atp_levels, label="ATP Production (SU(2))", linestyle="dotted")
plt.xlabel("Time")
plt.ylabel("Activity Level")
plt.title("Bioelectric Field Inducing Metabolism and ATP Production")
plt.legend()
plt.show()

# Print final ATP production levels
print("Final ATP Levels:", atp_levels[-1])
