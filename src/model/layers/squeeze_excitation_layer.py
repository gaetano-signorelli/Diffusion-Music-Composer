import tensorflow as tf
from tensorflow.keras import layers

class SqueezeAndExcitationLayer(layers.Layer):

    def __init__(self, n_filters, ratio=16):

        super(SqueezeAndExcitationLayer, self).__init__()

        self.global_avg_pool_layer = layers.GlobalAveragePooling2D()

        self.reshape_layer = layers.Reshape((1,1,n_filters))

        self.dense_layer_1 = layers.Dense(n_filters // ratio, activation='relu',
                                kernel_initializer='he_normal', use_bias=False)
        self.dense_layer_2 = layers.Dense(n_filters, activation='sigmoid',
                                kernel_initializer='he_normal', use_bias=False)

        self.multiply_layer = layers.Multiply()


    @tf.function
    def call(self, x):

        se_x = self.global_avg_pool_layer(x)

        se_x = self.reshape_layer(se_x)

        se_x = self.dense_layer_1(se_x)
        se_x = self.dense_layer_2(se_x)

        se_x = self.multiply_layer([x,se_x])

        return se_x
