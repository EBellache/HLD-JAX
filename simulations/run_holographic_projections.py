import jax.numpy as jnp
from holography.fourier_projection import holographic_projection

# Define projection parameters
A_k = jnp.array([1.0, 0.8, 0.6, 0.4, 0.2])
k_values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
phi_t = 0.5
params = (A_k, k_values, phi_t)

# Compute projection
x_values = jnp.linspace(-10, 10, 500)
t_value = 0.0
projection_values = holographic_projection(x_values, t_value, params)

# Print first few values
print(projection_values[:10])
