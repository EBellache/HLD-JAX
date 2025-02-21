import jax.numpy as jnp
from contact_geometry.mqp_corrections import corrected_gauge_force
from holography.projected_interactions import project_mqp_corrected_force

# Define gauge interactions (SU(3), SU(2), U(1))
gauge_groups = [3, 2, 1]
coupling_constants = jnp.array([0.118, 0.033, 1/137])  # QCD, Weak, Electromagnetic

# Define example metric tensor (projection space)
metric_tensor = jnp.eye(4)

# Define test spatial position
x_values = jnp.linspace(-10, 10, 500)

# Compute corrected force interactions
gauge_interaction = jnp.array([0.1, 0.05, 0.01])  # Example gauge forces
corrected_force = corrected_gauge_force(x_values, metric_tensor, gauge_interaction)

# Project into holographic space
projected_forces = project_mqp_corrected_force(x_values, metric_tensor, gauge_interaction, lambda t: jnp.sin(t))

# Print results
print("Projected Force Fields with MQP Corrections:")
print(projected_forces[:10])
