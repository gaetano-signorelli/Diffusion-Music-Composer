import tensorflow as tf
from tensorflow.keras import layers, activations

class SelfAttentionLayer(layers.Layer):

    def __init__(self, input_shape, n_heads):

        super(SelfAttentionLayer, self).__init__()

        self.h = input_shape[0]
        self.w = input_shape[1]
        self.c = input_shape[2]

        self.self_attention_layer = layers.MultiHeadAttention(n_heads, self.c)

        self.layer_normalization_1 = layers.LayerNormalization()
        self.layer_normalization_2 = layers.LayerNormalization()

        self.dense_layer_1 = layers.Dense(self.c)
        self.dense_layer_2 = layers.Dense(self.c)

        self.reshape_layer_1 = layers.Reshape((self.h*self.w, self.c))
        self.reshape_layer_2 = layers.Reshape((self.h, self.w, self.c))

    @tf.function
    def call(self, x):

        x = self.reshape_layer_1(x)

        norm_x = self.layer_normalization_1(x)

        attention_x = self.self_attention_layer(norm_x, norm_x)
        attention_x = attention_x + x

        linear_attention_x = self.layer_normalization_2(attention_x)
        linear_attention_x = self.dense_layer_1(linear_attention_x)
        linear_attention_x = activations.gelu(linear_attention_x)
        linear_attention_x = self.dense_layer_2(linear_attention_x)

        output = linear_attention_x + attention_x
        output = self.reshape_layer_2(output)

        return output
