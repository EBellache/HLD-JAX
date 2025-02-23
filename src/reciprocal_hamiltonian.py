from jax import grad, jit


@jit
def reciprocal_hamiltonian(k, p_tilde, H, lambda_p, TQF_pressure=0):
    """
    Computes time evolution in Fourier-transformed holographic projection space.

    Args:
        k (jax.numpy.array): Fourier wavevector.
        p_tilde (jax.numpy.array): Projected holographic momentum.
        H (function): Hamiltonian function.
        lambda_p (float): Dissipation parameter.
        TQF_pressure (float): Pressure exerted by the time-like quantum fluid.

    Returns:
        dk/dt, dp_tilde/dt
    """
    dk_dt = grad(H, argnums=1)(k, p_tilde)
    dp_tilde_dt = -grad(H, argnums=0)(k, p_tilde) + lambda_p * p_tilde - TQF_pressure * k

    return dk_dt, dp_tilde_dt
