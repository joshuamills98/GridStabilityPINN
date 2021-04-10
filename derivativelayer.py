import tensorflow as tf 

"""
This file will compute the gradients in order to compute the f_loss and u_loss

"""

class DerivativeLayer(tf.keras.layers.Layer): #Inherit from keras layers to use call method

    def __init__(self,model):

        self.model = model
        super(DerivativeLayer,self).__init__()

    def call(self, inputs):
        """
        Implement gradient tape method to compute derivatives
        
        inputs[:,0] is time
        inputs[:,1] is position
        
        """
        t = tf.convert_to_tensor(inputs[:,0], dtype = 'float32')  # Time
        P = tf.convert_to_tensor(inputs[:,1], dtype = 'float32')  # Power
        m = tf.convert_to_tensor(inputs[:,2], dtype = 'float32')  # Inertial constant of system
        d = tf.convert_to_tensor(inputs[:,3], dtype = 'float32')  # Damping constant of generator
        
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(t)
            c = self.model(tf.stack([t,P,m,d],axis=1))  # calculate output (i.e. concentration)
            dc_dt = gg.gradient(c, t)  # Derive within tape so you can use tape after 

        d2c_dt2 = gg.gradient(dc_dt,t)

        return c, dc_dt, d2c_dt2
