import tensorflow as tf
from tensorflow.keras import layers

class PositionalEncodingLayer(layers.Layer):

    def __init__(self, embedding_size):

        super(PositionalEncodingLayer, self).__init__()

        self.embedding_size = embedding_size

        self.conc_layer = layers.Concatenate(axis=-1)

    @tf.function
    def call(self, x):

        half_dim = self.embedding_size // 2
        emb_range = tf.range(half_dim, dtype=tf.float32)

        x = tf.repeat(x, repeats=half_dim, axis=-1)
        x = tf.cast(x, dtype=tf.float32)

        embeddings = -tf.math.log(10000.0) / (half_dim - 1)
        embeddings = tf.math.exp(emb_range * embeddings)

        embeddings = x * embeddings

        emb_sin = tf.math.sin(embeddings)
        emb_cos = tf.math.cos(embeddings)

        positional_encoding = self.conc_layer([emb_sin, emb_cos])

        return positional_encoding
