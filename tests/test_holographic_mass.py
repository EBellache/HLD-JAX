import pytest
import jax.numpy as jnp
from core.holographic_mass import compute_holographic_mass


def test_compute_holographic_mass():
    """Test holographic mass calculation."""
    rho_k = jnp.array([1.0, 2.0, 3.0])
    Mh = compute_holographic_mass(rho_k)
    assert Mh.shape == rho_k.shape, "Incorrect shape for computed mass"
    assert jnp.all(Mh > 0), "Mass should be positive"
