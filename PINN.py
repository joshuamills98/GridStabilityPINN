import tensorflow as tf
from derivativelayer import DerivativeLayer

"""
Here we define the entire model for the inputs and outputs
This will require the base Neural Network contained in:
basenetwork.py - this takes inputs t, P and outputs c
And the derivativelayer, contained in:
derivativelayer.py
"""


class PINNModel:

    """
    This will wrap the model up into its inputs and outputs  which are:
    Inputs:
    - Collocation inputs t, P, m and d over domain (txcol)
    - Inputs t and P over initial condition (txini)
    - Inputs t and P over boudnary condition (txbound)
    Outputs:
    - f_out: what we wish to minimize in order to regularize to the
    physics conditions (f_out = m*d2c/dt2 +d*dc/dt + BV1V2sin(c)-x)
    - c_ini: initial conditions of the model (for data regularization)
    - c_bound: boundary conditions of the model (for data regularization)
    """

    def __init__(self, base_model):

        self.base_model = base_model
        self.derivativelayer = DerivativeLayer(base_model)

    def build(self, BV1V2=0.2):

        # Create inputs:
        # Collocation points over domain (Time and Power)
        tP_col = tf.keras.layers.Input(shape=(4,))

        # Initial condition points
        tP_ini_1 = tf.keras.layers.Input(shape=(4, ))

        # First derivative initial condition
        tP_ini_2 = tf.keras.layers.Input(shape=(4, ))

        # f_out: (for conveciton diffusion equation we
        # only need first derivatives in x and t and second in x)
        c, dc_dt, d2c_dt2 = self.derivativelayer(tP_col)

        f_out = tf.reshape(
                    tP_col[:, 2], [-1, 1])*tf.reshape(d2c_dt2, [-1, 1]) + \
            tf.reshape(
                        tP_col[:, 3], [-1, 1])*tf.reshape(dc_dt, [-1, 1]) + \
            BV1V2*tf.math.sin(tf.reshape(c, [-1, 1])) - tf.reshape(
                tP_col[:, 1], [-1, 1])

        c_ini_1 = self.base_model(tP_ini_1)

        # c_ini_2:
        _, c_ini_2, _ = self.derivativelayer(tP_ini_2)

        # Pack all the inputs and outputs into model
        f_out = tf.reshape(f_out, [-1, 1])
        return tf.keras.models.Model(
            inputs=[tP_col, tP_ini_1, tP_ini_2],
            outputs=[f_out, c_ini_1, c_ini_2]
        )
