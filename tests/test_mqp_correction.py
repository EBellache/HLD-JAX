import pytest
import jax.numpy as jnp
from src.mqp_correction import compute_emergent_potential


def test_compute_emergent_potential():
    """Test MQP calculation with expected behavior."""
    rho_k = jnp.array([0.5, 1.0, 1.5])
    result = compute_emergent_potential(rho_k)
    assert result.shape == rho_k.shape, "Output shape mismatch"
    assert jnp.all(result < 0), "MQP must be negative due to Bohmian correction"
