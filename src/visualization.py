import jax.numpy as jnp
import vispy.scene
from vispy.scene import visuals
import numpy as np
from jax import jit


class TQF_Visualizer:
    def __init__(self, size=512):
        """
        Initializes visualization for the Time-Like Quantum Fluid (TQF).
        """
        self.size = size
        self.canvas = vispy.scene.SceneCanvas(keys='interactive', bgcolor='black', size=(800, 800))
        self.view = self.canvas.central_widget.add_view()
        self.image = visuals.Image(np.zeros((size, size)), cmap='plasma')
        self.view.add(self.image)
        self.view.camera = 'panzoom'
        self.canvas.show()

    @jit
    def simulate_tqf_projection(self, field_k, TQF_pressure):
        """
        Simulates how gauge fields imprint onto the time-like quantum fluid.

        Args:
            field_k (jax.numpy.array): Fourier-transformed gauge field.
            TQF_pressure (float): Pressure exerted by TQF.

        Returns:
            Real-space holographic imprint.
        """
        field_real = jnp.fft.ifftshift(
            jnp.fft.ifft2(jnp.fft.ifftshift(field_k * jnp.exp(-TQF_pressure * field_k ** 2))))
        return jnp.abs(field_real) / jnp.max(field_real)

    def update(self, field_k, TQF_pressure):
        """
        Updates visualization with a new TQF simulation.
        """
        field_real = np.array(self.simulate_tqf_projection(field_k, TQF_pressure))
        self.image.set_data(field_real)
        self.canvas.update()
