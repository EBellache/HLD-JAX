import pytest
import jax.numpy as jnp
from core.gauge_reciprocal_hamiltonian import (
    gauge_covariant_momentum,
    gauge_reciprocal_hamiltonian,
)


@pytest.fixture
def test_data():
    """Fixture to provide test inputs for functions."""
    k = jnp.array([1.0, 0.5])
    p_tilde = jnp.array([0.2, 0.3])
    gauge_field = jnp.array([0.1, -0.1])
    rho_k = jnp.array([0.8, 1.2])
    lambda_p = 0.05
    substrate_pressure = 0.02
    return k, p_tilde, gauge_field, rho_k, lambda_p, substrate_pressure


def test_gauge_covariant_momentum(test_data):
    """Test gauge transformation correctly modifies momentum."""
    _, p_tilde, gauge_field, rho_k, _, _ = test_data
    p_gauge = gauge_covariant_momentum(p_tilde, gauge_field, rho_k)
    expected_p_gauge = (p_tilde - gauge_field) / (1.0 + rho_k)  # Expected mass normalization
    assert jnp.allclose(p_gauge, expected_p_gauge), "Gauge covariant momentum incorrect"


def test_gauge_reciprocal_hamiltonian(test_data):
    """Test Hamiltonian time evolution equations are correctly computed."""
    k, p_tilde, gauge_field, rho_k, lambda_p, substrate_pressure = test_data
    dk_dt, dp_tilde_dt = gauge_reciprocal_hamiltonian(k, p_tilde,
                                                      lambda k, p: 0.5 * jnp.dot(k, k) + 0.5 * jnp.dot(p, p),
                                                      gauge_field, rho_k, lambda_p, substrate_pressure)
    assert dk_dt.shape == k.shape, "Incorrect shape for dk/dt"
    assert dp_tilde_dt.shape == p_tilde.shape, "Incorrect shape for dp_tilde/dt"
