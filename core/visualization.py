import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, grad
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def visualize_geodesic(trajectory, title="Geodesic Path", color="b"):
    """Visualizes geodesic trajectories with torsion & helical dislocation effects."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    ax.plot(x, y, z, label=title, color=color)
    ax.scatter(x[0], y[0], z[0], color='r', marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='g', marker='x', label='End')
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    ax.set_title(title), ax.legend()
    plt.show()