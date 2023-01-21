import tensorflow as tf
from tensorflow.keras import layers, activations

from src.model.layers.conv_block_layer import ConvolutionalBlockLayer

class DownSampleLayer(layers.Layer):

    def __init__(self, input_shape, out_channels, kernel_size):

        super(DownSampleLayer, self).__init__()

        self.h = input_shape[0]
        self.w = input_shape[1] // 2
        self.c = input_shape[2]

        self.out_c = out_channels

        self.max_pool_layer = layers.MaxPooling2D(pool_size=(1,2))

        self.conv_block_1 = ConvolutionalBlockLayer(kernel_size, self.c, residual=True)
        self.conv_block_2 = ConvolutionalBlockLayer(kernel_size, self.out_c)

        self.dense_layer = layers.Dense(self.out_c)

    @tf.function
    def call(self, inputs):

        x = inputs[0]
        t = inputs[1]

        x = self.max_pool_layer(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        t = activations.swish(t)
        t = self.dense_layer(t)

        t = tf.expand_dims(t, axis=-1)
        t = tf.tile(t, tf.constant([1,1,self.h*self.w]))
        t = layers.Reshape((self.h, self.w, self.out_c))(t)

        output = x + t

        return output
