import jax.numpy as jnp
from jax import jit, grad, vmap, pmap


@jit
def update_tetrad_lattice(tetrad_state, deformation_field, phase_field, alpha=0.1, phase_increment=0.01):
    """Updates tetrad lattice under deformation field effects with U(1) phase accumulation and optimized for GPU execution."""
    phase_shift = jnp.exp(1j * phase_increment)
    return pmap(lambda ts, df, pf: phase_shift * (ts + alpha * df + pf))(tetrad_state, deformation_field, phase_field)


# Lattice Pressure Computation (formerly MQP)
@jit
def compute_lattice_pressure(rho_k, hbar=1.0, m=1.0):
    """Computes lattice pressure as an emergent effect from tetrad fluctuations."""
    return - (hbar ** 2 / (2 * m)) * (jnp.abs(jnp.fft.fftfreq(rho_k.shape[0])) ** 2 * rho_k / (rho_k + 1e-8))
