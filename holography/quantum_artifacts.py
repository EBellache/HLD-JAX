import jax.numpy as jnp
from jax import jit
from holography.fourier_projection import holographic_projection
import jax

@jit
def compute_quantum_foam(phase_field):
    """
    Computes quantum foam effects from holographic interference patterns.
    Args:
        phase_field: Fourier phase representation of the projection
    Returns:
        Quantum foam intensity map
    """
    return jnp.abs(jnp.fft.ifft(phase_field)) ** 2

@jit
def wave_particle_duality(x, t, params):
    """
    Computes the emergent wave-particle duality by extracting Fourier components.
    Args:
        x: Spatial coordinate
        t: Time coordinate
        params: Holographic projection parameters
    Returns:
        Probability amplitude from interference pattern
    """
    projection_wave = holographic_projection(x, t, params)
    return jnp.abs(projection_wave) ** 2  # Probability density function

@jit
def compute_mqp(density_field):
    """
    Computes the Macroscopic Quantum Potential (MQP) from projected field density.
    Args:
        density_field: The projected density field (analogous to wavefunction modulus)
    Returns:
        MQP as a correction potential
    """
    laplacian = jnp.gradient(jnp.gradient(density_field))
    return - (jnp.gradient(laplacian) / density_field)

@jit
def compute_entanglement_entropy(phase_field):
    """
    Computes entanglement entropy from phase correlations in the projection field.
    Args:
        phase_field: Fourier phase representation of the holographic projection
    Returns:
        Entanglement entropy measure
    """
    spectral_density = jnp.abs(jnp.fft.fft(phase_field)) ** 2
    probability_distribution = spectral_density / jnp.sum(spectral_density)
    entropy = -jnp.sum(probability_distribution * jnp.log(probability_distribution + 1e-10))
    return entropy

@jit
def holographic_decoherence(phase_field, decoherence_rate):
    """
    Models holographic decoherence by introducing stochastic phase noise.
    Args:
        phase_field: Fourier phase representation of projection
        decoherence_rate: Strength of phase noise
    Returns:
        Decohered phase field
    """
    noise = jax.random.normal(size=phase_field.shape) * decoherence_rate
    return phase_field + noise
