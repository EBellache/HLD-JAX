import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import expm


# 1️⃣ U(1) Projection - Abelian Gauge Symmetry
@jit
def u1_projection(field, theta):
    """
    Applies a U(1) phase transformation to the holographic field.
    This represents electromagnetism in Fourier space.

    Args:
        field (jax.numpy.ndarray): The Fourier-transformed field.
        theta (float): Phase shift parameter.

    Returns:
        jax.numpy.ndarray: Phase-shifted field.
    """
    return field * jnp.exp(1j * theta)


# 2️⃣ SU(2) Projection - Non-Abelian Gauge Symmetry
@jit
def su2_projection(field, alpha, beta, gamma):
    """
    Applies an SU(2) rotation to the Fourier holographic field.
    This corresponds to weak force interactions.

    Args:
        field (jax.numpy.ndarray): The Fourier-transformed field.
        alpha, beta, gamma (float): SU(2) rotation parameters.

    Returns:
        jax.numpy.ndarray: Rotated field.
    """
    # Define Pauli Matrices
    sigma_x = jnp.array([[0, 1], [1, 0]])
    sigma_y = jnp.array([[0, -1j], [1j, 0]])
    sigma_z = jnp.array([[1, 0], [0, -1]])

    # Generate SU(2) transformation matrix
    su2_matrix = expm(1j * (alpha * sigma_x + beta * sigma_y + gamma * sigma_z))

    # Apply transformation to the field
    return jnp.einsum('ij,...j->...i', su2_matrix, field)


# 3️⃣ SU(3) Projection - Strong Force (Color Charge)
@jit
def su3_projection(field, g_params):
    """
    Applies an SU(3) rotation to the Fourier holographic field.
    This models interactions in the strong force.

    Args:
        field (jax.numpy.ndarray): The Fourier-transformed field.
        g_params (list of floats): Eight SU(3) transformation parameters.

    Returns:
        jax.numpy.ndarray: Rotated field.
    """
    # Define Gell-Mann Matrices
    lambda_1 = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    lambda_2 = jnp.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    lambda_3 = jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    lambda_4 = jnp.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    lambda_5 = jnp.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
    lambda_6 = jnp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    lambda_7 = jnp.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
    lambda_8 = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) * (1 / jnp.sqrt(3))

    # Construct SU(3) generator
    su3_generator = (
            g_params[0] * lambda_1 + g_params[1] * lambda_2 + g_params[2] * lambda_3 +
            g_params[3] * lambda_4 + g_params[4] * lambda_5 + g_params[5] * lambda_6 +
            g_params[6] * lambda_7 + g_params[7] * lambda_8
    )

    # Compute SU(3) transformation matrix
    su3_matrix = expm(1j * su3_generator)

    # Apply transformation to the field
    return jnp.einsum('ij,...j->...i', su3_matrix, field)


def apply_gauge_projections(field, u1_theta, su2_params, su3_params):
    """
    Applies U(1), SU(2), and SU(3) gauge transformations sequentially.

    Args:
        field (jax.numpy.ndarray): Fourier holographic field.
        u1_theta (float): U(1) phase shift.
        su2_params (tuple): (alpha, beta, gamma) SU(2) rotation parameters.
        su3_params (list): SU(3) transformation parameters.

    Returns:
        jax.numpy.ndarray: Gauge-transformed field.
    """
    field_u1 = u1_projection(field, u1_theta)
    field_su2 = su2_projection(field_u1, *su2_params)
    field_su3 = su3_projection(field_su2, su3_params)

    return field_su3
