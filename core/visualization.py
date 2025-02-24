import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Function to visualize SU(2) geodesic trajectory
def visualize_su2_geodesic(trajectory):
    """Visualizes SU(2) geodesic trajectory with torsion effects."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    ax.plot(x, y, z, label='SU(2) Geodesic', color='b')
    ax.scatter(x[0], y[0], z[0], color='r', marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='g', marker='x', label='End')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("SU(2) Geodesic Path with Torsion")
    ax.legend()
    plt.show()


# Function to visualize SU(3) geodesic trajectory
def visualize_su3_geodesic(trajectory):
    """Visualizes SU(3) geodesic trajectory with helical dislocation effects."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    ax.plot(x, y, z, label='SU(3) Geodesic', color='purple')
    ax.scatter(x[0], y[0], z[0], color='r', marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='g', marker='x', label='End')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("SU(3) Geodesic Path with Helical Dislocation")
    ax.legend()
    plt.show()


# Function to animate SU(2) and SU(3) geodesics
def animate_geodesics(trajectory_su2, trajectory_su3, save_as=None):
    """Creates an animation showing the evolution of SU(2) and SU(3) geodesics."""
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    x_su2, y_su2, z_su2 = trajectory_su2[:, 0], trajectory_su2[:, 1], trajectory_su2[:, 2]
    x_su3, y_su3, z_su3 = trajectory_su3[:, 0], trajectory_su3[:, 1], trajectory_su3[:, 2]

    line_su2, = ax.plot([], [], [], label='SU(2) Geodesic', color='b')
    line_su3, = ax.plot([], [], [], label='SU(3) Geodesic', color='purple')

    def update(frame):
        line_su2.set_data(x_su2[:frame], y_su2[:frame])
        line_su2.set_3d_properties(z_su2[:frame])
        line_su3.set_data(x_su3[:frame], y_su3[:frame])
        line_su3.set_3d_properties(z_su3[:frame])
        return line_su2, line_su3

    anim = FuncAnimation(fig, update, frames=len(x_su2), interval=50, blit=False)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Animated SU(2) and SU(3) Geodesic Paths")
    ax.legend()

    if save_as:
        anim.save(save_as, writer='pillow')
    else:
        plt.show()
