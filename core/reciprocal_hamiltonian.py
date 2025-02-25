import jax.numpy as jnp
from jax import jit, grad

@jit
def reciprocal_hamiltonian_wavelet(k, wavelet_coeffs, H, gauge_field, rho_k, lambda_p, hbar=1.0, m=1.0):
    """
    Computes Fourier-space Hamiltonian evolution using wavelet coefficients in the triad lattice.

    Args:
        k (jax.numpy.array): Fourier wavevector.
        wavelet_coeffs (jax.numpy.array): Wavelet-transformed momentum representation.
        H (function): Hamiltonian function.
        gauge_field (jax.numpy.array): Gauge field modifying momentum.
        rho_k (jax.numpy.array): Local lattice density in Fourier space.
        lambda_p (float): Dissipation parameter.
        hbar (float): Reduced Planck's constant.
        m (float): Effective mass parameter.

    Returns:
        tuple: (dk/dt, dp_wavelet/dt) - Time evolution of wavevector and wavelet momentum.
    """
    p_wavelet = (wavelet_coeffs - gauge_field) / (1.0 + rho_k)  # Gauge-covariant wavelet transformation
    lattice_pressure = - (hbar ** 2 / (2 * m)) * (jnp.abs(jnp.fft.fftfreq(rho_k.shape[0])) ** 2 * rho_k / (rho_k + 1e-8))

    dk_dt = grad(H, argnums=1)(k, p_wavelet)
    dp_wavelet_dt = -grad(H, argnums=0)(k, p_wavelet) + lambda_p * p_wavelet - lattice_pressure

    return dk_dt, dp_wavelet_dt
