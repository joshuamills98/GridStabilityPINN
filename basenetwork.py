"""
Store the base Neural Network here
"""
import tensorflow as tf
import numpy as np
class Network:

    @classmethod
    def basemodel(cls,num_inputs = 4, hidden_layers = [20,20,20,20], activation_layers = 'tanh', num_outputs= 1,  ub = np.array([20, 0.18, 0.4, 0.15]), lb = np.array([0, 0.08, 0.1, 0.05])):

        #Input layer
        inputs = tf.keras.layers.Input(shape = (num_inputs,))

        #Use functional API to build model by chaining function calls

        x = inputs 
        x = 2.0*(x-lb)/(ub - lb) - 1.0
        
        for layer_size in hidden_layers:
            x = tf.keras.layers.Dense(layer_size, activation = activation_layers)(x)

        output = tf.keras.layers.Dense(num_outputs)(x)

        model = tf.keras.Model(inputs = inputs, outputs = output)

        return model


def set_flattened_weights(model, flattened_weights):
    index = 0
    new_weights = []
    shapes = [weights.shape for weights in model.get_weights()]
    num_weight_layers = len(shapes)
    
    for layer in range(num_weight_layers):
        shape = shapes[layer]
        weight_index = np.prod(shape)
        new_weights.append(flattened_weights[index:index+weight_index].reshape(shape))
        index += weight_index

    model.set_weights(np.array(new_weights))

    return(model)