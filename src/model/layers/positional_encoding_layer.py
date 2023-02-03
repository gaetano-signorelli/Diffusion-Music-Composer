import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class TimePositionalEncodingLayer(layers.Layer):

    def __init__(self, embedding_size):

        super(TimePositionalEncodingLayer, self).__init__()

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

class SpatialPositionalEncodingLayer(layers.Layer):

    def __init__(self, length, embedding_size):

        super(SpatialPositionalEncodingLayer, self).__init__()

        self.length = length
        self.embedding_size = embedding_size

        self.dense_layer = layers.Dense(self.embedding_size)

    def __get_angles(self, pos, i, d_model):

        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        return pos * angle_rates

    def __get_encoding(self):

        # Compute the angles of each position
        angle_rads = self.__get_angles(np.arange(self.length)[:, np.newaxis],
                                np.arange(self.embedding_size)[np.newaxis, :],
                                self.embedding_size)

        # Compute the sin of angles of even positions
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Compute the cosine of angles of odd positions
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        # Get positional layer_encoder for each position
        pos_encoding = angle_rads[np.newaxis, ...]

        return pos_encoding

    def call(self, x):

        pos_encoding = self.__get_encoding()
        pos_encoding = tf.expand_dims(pos_encoding, axis=1)
        pos_encoding = self.dense_layer(pos_encoding)

        return x + pos_encoding
