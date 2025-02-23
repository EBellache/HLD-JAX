from jax import grad, jit

from holographic_mass import compute_holographic_mass


@jit
def gauge_covariant_momentum(p_tilde, gauge_field, rho_k):
    """
    Compute the gauge-invariant momentum transformation, now including holographic mass.

    Args:
        p_tilde (jax.numpy.array): Projected Fourier-space momentum.
        gauge_field (jax.numpy.array): Gauge field modifying the momentum.
        rho_k (jax.numpy.array): Fourier-space density (used to compute holographic mass).

    Returns:
        jax.numpy.array: Mass-adjusted, gauge-invariant momentum.
    """
    M_h = compute_holographic_mass(rho_k) + 1e-8  # Prevent division errors
    return (p_tilde - gauge_field) / M_h  # Adjusted for dynamic mass


@jit
def gauge_reciprocal_hamiltonian(k, p_tilde, H, gauge_field, rho_k, lambda_p, substrate_pressure=0):
    """
    Computes time evolution of Fourier-space Hamiltonian with gauge interactions and holographic mass.

    Args:
        k (jax.numpy.array): Fourier wavevector.
        p_tilde (jax.numpy.array): Projected momentum in Fourier space.
        H (function): Hamiltonian function.
        gauge_field (jax.numpy.array): Gauge field modifying momentum.
        rho_k (jax.numpy.array): Fourier-space density (used to compute holographic mass).
        lambda_p (float): Dissipation parameter.
        substrate_pressure (float): Pressure from the substrate lattice.

    Returns:
        (dk/dt, dp_tilde/dt) - Time evolution of wavevector and momentum.
    """
    p_gauge = gauge_covariant_momentum(p_tilde, gauge_field, rho_k)

    dk_dt = grad(H, argnums=1)(k, p_gauge)
    dp_tilde_dt = -grad(H, argnums=0)(k, p_gauge) + lambda_p * p_gauge - substrate_pressure * k

    return dk_dt, dp_tilde_dt


@jit
def evolve_gauge_hamiltonian_with_mass(k, p_tilde, field, H, gauge_field, rho_k, lambda_p):
    """
    Evolves the reciprocal Hamiltonian with gauge interactions and holographic mass.

    Returns:
        Updated (k, p_tilde) after mass-dependent, gauge-covariant evolution.
    """
    dk_dt, dp_tilde_dt = gauge_reciprocal_hamiltonian(k, p_tilde, H, gauge_field, rho_k, lambda_p)

    k_new = k + dk_dt
    p_tilde_new = p_tilde + dp_tilde_dt

    return k_new, p_tilde_new
