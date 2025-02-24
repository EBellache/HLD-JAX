import jax.numpy as jnp
from jax import jit, grad, vmap, random

# Parameters for bioelectric evolution
lambda_decay = 0.05  # Phase dissipation rate
alpha_diffusion = 0.1  # Bioelectric diffusion strength
beta_nonlinearity = 0.5  # Nonlinear bistability control

# Parameters for tetrad lattice memory
theta_phase_shift = 0.01  # Phase accumulation enforcing irreversibility

# Parameters for SU(2) epigenetic switching
gamma_stability = 0.1  # Stabilization factor
delta_coupling = 0.3  # Bioelectric-epigenetic coupling strength

# Initialize lattice
Nx, Ny = 50, 50  # Grid size
key = random.PRNGKey(42)
Phi = random.uniform(key, shape=(Nx, Ny)) * 2 * jnp.pi - jnp.pi  # Bioelectric field
L = jnp.zeros((Nx, Ny))  # Tetrad lattice memory
S = jnp.zeros((Nx, Ny))  # Epigenetic state (SU(2) switch)


@jit
def update_bioelectric_field(Phi, L):
    """Evolves bioelectric field with phase memory effects."""
    laplacian = jnp.roll(Phi, 1, axis=0) + jnp.roll(Phi, -1, axis=0) + \
                jnp.roll(Phi, 1, axis=1) + jnp.roll(Phi, -1, axis=1) - 4 * Phi
    dPhi_dt = -lambda_decay * Phi + alpha_diffusion * laplacian + beta_nonlinearity * jnp.cos(Phi)
    return Phi + dPhi_dt


@jit
def update_tetrad_memory(L, Phi):
    """Updates lattice memory with phase accumulation."""
    phase_shift = jnp.exp(1j * theta_phase_shift)
    return phase_shift * L + Phi


@jit
def update_epigenetic_switch(S, Phi):
    """Evolves epigenetic bistable states based on bioelectric field strengths."""
    dS_dt = -gamma_stability * S + delta_coupling * jnp.tanh(Phi)
    return S + dS_dt


# Run simulation for T steps
T = 100
for t in range(T):
    Phi = update_bioelectric_field(Phi, L)
    L = update_tetrad_memory(L, Phi)
    S = update_epigenetic_switch(S, Phi)

# Print final states
print("Final Bioelectric Field:", Phi)
print("Final Tetrad Memory:", L)
print("Final Epigenetic State:", S)
