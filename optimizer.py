import scipy.optimize
import numpy as np
import tensorflow as tf


class Optimizer:

    """
    The optimizer will update the weights of the network by
    performing Adam optimization.
    The optimization algorithm will require the weights
    to be completely flattened to perform gradient updates
    """

    def __init__(self, model, x_train,
                 y_train, maxiter, regularizer=1,
                 factr=1e5, m=50, maxls=50):
        self.model = model
        self.x_train = [tf.constant(x, dtype=tf.float32) for x in x_train]
        self.y_train = [tf.constant(y, dtype=tf.float32) for y in y_train]

        self.regularizer = regularizer
        # Shapes of each weight layer
        self.shapes = [weights.shape for weights in self.model.get_weights()]
        self.num_weight_layers = len(self.shapes)
        self.maxiter = maxiter
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.metrics = ['loss']
        self.mse = tf.keras.losses.MeanSquaredError()
        # initialize the progress bar
        self.progbar = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps', stateful_metrics=self.metrics)
        self.progbar.set_params({
            'verbose': 1, 'epochs': 1, 'steps': 1, 'metrics': self.metrics})

    def set_weights(self, flattened_weights):

        """
        Time taken by this method = 0.004 seconds to update weights,
        store in init?
        """

        index = 0
        new_weights = []

        for layer in range(self.num_weight_layers):
            shape = self.shapes[layer]
            weight_index = np.prod(shape)
            new_weights.append(
                flattened_weights[index:index+weight_index].reshape(shape))
            index += weight_index

        self.model.set_weights(np.array(new_weights))

    @tf.function
    def tf_evaluate(self, x, y):

        """
        Evaluate current NN, given weights
        """

        with tf.GradientTape() as g:
            # Unpack y
            f_out_true, c_ini_1_true, c_ini_2_true = y

            # Get outputs manually to feed into loss
            f_out, c_ini_1, c_ini_2 = self.model(x)

            # Determine three components of loss
            loss = tf.reduce_mean(
                        tf.square(f_out-f_out_true)) + self.regularizer*(
                            tf.reduce_mean(tf.square(c_ini_1-c_ini_1_true)) +
                            tf.reduce_mean(tf.square(c_ini_2-c_ini_2_true)))
        grads = g.gradient(loss, self.model.trainable_variables)

        return loss, grads

    def evaluate(self, weights):

        """
        Evaluate loss and gradients for weights as ndarray.
        """

        self.set_weights(weights)
        # compute loss and gradients for weights
        loss, grads = self.tf_evaluate(self.x_train, self.y_train)
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype(
                                                                    'float64')
        return loss, grads

    def callback(self, weights):

        self.progbar.on_epoch_begin(10)
        loss, _ = self.evaluate(weights)
        self.progbar.on_epoch_end(10, logs=dict(zip(self.metrics, [loss])))

    def fit(self, iterations=None):
        """
        fit model using Keras optimizer
        """

        if iterations is None:
            iterations = self.maxiter

        initial_weights = np.concatenate(
            [w.flatten() for w in self.model.get_weights()])

        self.progbar.on_train_begin()
        self.progbar.on_epoch_begin(1)
        print(
            'Optimizer: L_BFGS Optimizer for {} iterations'.format(iterations))
        scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights,
                                     m=self.m,
                                     factr=self.factr, maxls=50,
                                     maxiter=iterations,
                                     callback=self.callback)
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()
