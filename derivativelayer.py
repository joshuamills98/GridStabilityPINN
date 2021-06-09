import tensorflow as tf

# Inherit from keras layers to use call method


class DerivativeLayer(tf.keras.layers.Layer):

    def __init__(self, model):

        self.model = model
        super(DerivativeLayer, self).__init__()

    def call(self, inputs):

        """
        Implement gradient tape method to compute derivatives
        inputs[:,0] is time
        inputs[:,1] is position
        inputs[:,2] is inertial constant
        inputs[:,3] is damping coefficient
        """

        t = tf.convert_to_tensor(inputs[:, 0], dtype='float32')
        P = tf.convert_to_tensor(inputs[:, 1], dtype='float32')
        m = tf.convert_to_tensor(inputs[:, 2], dtype='float32')
        d = tf.convert_to_tensor(inputs[:, 3], dtype='float32')

        with tf.GradientTape(persistent=True) as gg:
            gg.watch(t)

            # Calculate output (i.e. concentration)
            c = self.model(tf.stack([t, P, m, d], axis=1))

            # Derive within tape so you can use tape after
            dc_dt = gg.gradient(c, t)

        d2c_dt2 = gg.gradient(dc_dt, t)

        return c, dc_dt, d2c_dt2
