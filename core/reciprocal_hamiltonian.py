import jax.numpy as jnp
from jax import jit, grad, vmap, pmap


@jit
def reciprocal_hamiltonian(k, p_tilde, H, gauge_field, rho_k, lambda_p, hbar=1.0, m=1.0):
    """Computes time evolution of Fourier-space Hamiltonian with lattice pressure correction."""
    p_gauge = (p_tilde - gauge_field) / (1.0 + rho_k)
    lattice_pressure = - (hbar ** 2 / (2 * m)) * (
                jnp.abs(jnp.fft.fftfreq(rho_k.shape[0])) ** 2 * rho_k / (rho_k + 1e-8))
    dk_dt = grad(H, argnums=1)(k, p_gauge)
    dp_tilde_dt = -grad(H, argnums=0)(k, p_gauge) + lambda_p * p_gauge - lattice_pressure
    return dk_dt, dp_tilde_dt


@jit
def reciprocal_hamiltonian_wavelet(W, H, lambda_p, hbar=1.0, m=1.0):
    "Computes time evolution of wavelet-space Hamiltonian."
    # Compute pressure correction in wavelet space
    P_wavelet = - (hbar ** 2 / (2 * m)) * jnp.sum(jnp.gradient(jnp.gradient(W)) / (W + 1e-8))

    # Compute evolution
    dW_dt = -grad(H, argnums=0)(W) + lambda_p * W - P_wavelet
    return dW_dt
