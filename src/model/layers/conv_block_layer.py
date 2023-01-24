import tensorflow as tf
from tensorflow.keras import layers, activations

from src.model.layers.squeeze_excitation_layer import SqueezeAndExcitationLayer

class ConvolutionalBlockLayer(layers.Layer):

    def __init__(self, kernel_size, out_channels, mid_channels=None, residual=False):

        super(ConvolutionalBlockLayer, self).__init__()

        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels

        self.conv_layer_1 = layers.Conv2D(mid_channels,
                                        kernel_size=(1,kernel_size),
                                        padding="valid",
                                        use_bias=False)

        self.conv_layer_2 = layers.Conv2D(out_channels,
                                        kernel_size=(1,kernel_size),
                                        padding="valid",
                                        use_bias=False)

        self.group_norm_layer_1 = layers.LayerNormalization()
        self.group_norm_layer_2 = layers.LayerNormalization()

        self.squeeze_excitation_layer_1 = SqueezeAndExcitationLayer(mid_channels)
        self.squeeze_excitation_layer_2 = SqueezeAndExcitationLayer(out_channels)

        w_padding = (kernel_size - 1)
        self.padding_tensor_width = tf.constant([ [0,0], [0,0], [w_padding,0], [0,0]])
        #Width padding is only on the left side, making this a Causal convolution, like for WaveNet

    @tf.function
    def call(self, x):

        padded_x = tf.pad(x, self.padding_tensor_width, "CONSTANT")

        conv_x = self.conv_layer_1(padded_x)
        conv_x = self.group_norm_layer_1(conv_x)
        conv_x = activations.gelu(conv_x)
        conv_x = self.squeeze_excitation_layer_1(conv_x)

        padded_x = tf.pad(conv_x, self.padding_tensor_width, "CONSTANT")

        conv_x = self.conv_layer_2(padded_x)
        conv_x = self.group_norm_layer_2(conv_x)
        conv_x = self.squeeze_excitation_layer_2(conv_x)

        if self.residual:
            return activations.gelu(x + conv_x)

        else:
            return conv_x
