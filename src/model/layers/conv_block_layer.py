import tensorflow as tf
from tensorflow.keras import layers, activations

from src.model.layers.squeeze_excitation_layer import SqueezeAndExcitationLayer

class ConvolutionalBlockLayer(layers.Layer):

    def __init__(self, kernel_size, out_channels, mid_channels=None, residual=False):

        super(ConvolutionalBlockLayer, self).__init__()

        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels

        self.conv_layer_a1 = layers.Conv2D(mid_channels,
                                        kernel_size=(3,kernel_size),
                                        padding="valid",
                                        use_bias=False)

        self.conv_layer_a2 = layers.Conv2D(out_channels,
                                        kernel_size=(3,kernel_size),
                                        padding="valid",
                                        use_bias=False)

        self.conv_layer_b1 = layers.Conv2D(mid_channels,
                                        kernel_size=(3,kernel_size),
                                        padding="valid",
                                        use_bias=False)

        self.conv_layer_b2 = layers.Conv2D(out_channels,
                                        kernel_size=(3,kernel_size),
                                        padding="valid",
                                        use_bias=False)

        self.conv_layer_c1 = layers.Conv2D(mid_channels,
                                        kernel_size=(3,kernel_size),
                                        padding="valid",
                                        use_bias=False)

        self.conv_layer_c2 = layers.Conv2D(out_channels,
                                        kernel_size=(3,kernel_size),
                                        padding="valid",
                                        use_bias=False)

        self.group_norm_layer_1 = layers.LayerNormalization()
        self.group_norm_layer_2 = layers.LayerNormalization()

        self.squeeze_excitation_layer_1 = SqueezeAndExcitationLayer(mid_channels)
        self.squeeze_excitation_layer_2 = SqueezeAndExcitationLayer(out_channels)

        self.padding_tensor_height = tf.constant([ [0,0], [1,1], [0,0], [0,0]])
        w_padding = (kernel_size - 1)
        self.padding_tensor_width = tf.constant([ [0,0], [0,0], [w_padding,0], [0,0]])
        #Width padding is only on the left side, making this a Causal convolution, like for WaveNet

        self.get_a = layers.Cropping2D(cropping=( (0,2), (0,0)))
        self.get_b = layers.Cropping2D(cropping=( (1,1), (0,0)))
        self.get_c = layers.Cropping2D(cropping=( (2,0), (0,0)))

        self.concatenation_layer = layers.Concatenate(axis=1)

    @tf.function
    def call(self, x):

        padded_x = tf.pad(x, self.padding_tensor_height, "REFLECT")
        padded_x = tf.pad(padded_x, self.padding_tensor_width, "CONSTANT")

        a = self.get_a(padded_x)
        b = self.get_b(padded_x)
        c = self.get_c(padded_x)

        conv_a = self.conv_layer_a1(a)
        conv_b = self.conv_layer_b1(b)
        conv_c = self.conv_layer_c1(c)

        conv_x = self.concatenation_layer([conv_a,conv_b,conv_c])
        conv_x = self.group_norm_layer_1(conv_x)
        conv_x = activations.gelu(conv_x)
        conv_x = self.squeeze_excitation_layer_1(conv_x)

        padded_x = tf.pad(conv_x, self.padding_tensor_height, "REFLECT")
        padded_x = tf.pad(padded_x, self.padding_tensor_width, "CONSTANT")

        a = self.get_a(padded_x)
        b = self.get_b(padded_x)
        c = self.get_c(padded_x)

        conv_a = self.conv_layer_a2(a)
        conv_b = self.conv_layer_b2(b)
        conv_c = self.conv_layer_c2(c)

        conv_x = self.concatenation_layer([conv_a,conv_b,conv_c])
        conv_x = self.group_norm_layer_2(conv_x)
        conv_x = self.squeeze_excitation_layer_2(conv_x)

        if self.residual:
            return activations.gelu(x + conv_x)

        else:
            return conv_x
