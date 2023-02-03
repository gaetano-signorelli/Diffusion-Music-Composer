import tensorflow as tf
from tensorflow.keras import layers, Model

from src.model.layers.conv_block_layer import ConvolutionalBlockLayer
from src.model.layers.down_sample_layer import DownSampleLayer
from src.model.layers.up_sample_layer import UpSampleLayer
from src.model.layers.positional_encoding_layer import TimePositionalEncodingLayer, SpatialPositionalEncodingLayer
from src.model.layers.self_attention_layer import SelfAttentionLayer

class UNet(Model):

    def __init__(self, input_shape, n_heads, time_embedding_size):

        super().__init__()

        self.h = input_shape[0]
        self.w = input_shape[1]
        self.c = input_shape[2]

        assert self.w % 8 == 0

        self.time_positional_encoding_layer = TimePositionalEncodingLayer(time_embedding_size)

        self.conv_input = ConvolutionalBlockLayer(7,64)
        #self.spatial_positional_encoding_layer_1 = SpatialPositionalEncodingLayer(self.w, 64)

        self.down1 = DownSampleLayer((self.h,self.w,64), 128, 5)
        #self.spatial_positional_encoding_layer_2 = SpatialPositionalEncodingLayer(self.w//2, 128)
        self.sa1 = SelfAttentionLayer((self.h,self.w//2,128), n_heads)
        self.down2 = DownSampleLayer((self.h,self.w//2,128), 256, 5)
        #self.spatial_positional_encoding_layer_3 = SpatialPositionalEncodingLayer(self.w//4, 256)
        self.sa2 = SelfAttentionLayer((self.h,self.w//4,256), n_heads)
        self.down3 = DownSampleLayer((self.h,self.w//4,256), 256, 5)
        #self.spatial_positional_encoding_layer_4 = SpatialPositionalEncodingLayer(self.w//8, 256)
        self.sa3 = SelfAttentionLayer((self.h,self.w//8,256), n_heads)

        self.conv_mid_1 = ConvolutionalBlockLayer(3,512)
        self.conv_mid_2 = ConvolutionalBlockLayer(3,512)
        self.conv_mid_3 = ConvolutionalBlockLayer(3,256)

        self.up1 = UpSampleLayer((self.h,self.w//8,256), 128, 5)
        self.sa4 = SelfAttentionLayer((self.h,self.w//4,128), n_heads)
        self.up2 = UpSampleLayer((self.h,self.w//4,128), 64, 5)
        self.sa5 = SelfAttentionLayer((self.h,self.w//2,64), n_heads)
        self.up3 = UpSampleLayer((self.h,self.w//2,64), 64, 5)
        self.sa6 = SelfAttentionLayer((self.h,self.w,64), n_heads)

        self.conv_output = layers.Conv2D(self.c, kernel_size=1)

    @tf.function
    def call(self, inputs):

        x = inputs[0]
        t = inputs[1]

        t = self.time_positional_encoding_layer(t)

        x1 = self.conv_input(x)
        #x1 = self.spatial_positional_encoding_layer_1(x1)

        x2 = self.down1([x1, t])
        #x2 = self.spatial_positional_encoding_layer_2(x2)
        x2 = self.sa1(x2)

        x3 = self.down2([x2, t])
        #x3 = self.spatial_positional_encoding_layer_3(x3)
        x3 = self.sa2(x3)

        x4 = self.down3([x3, t])
        #x4 = self.spatial_positional_encoding_layer_4(x4)
        x4 = self.sa3(x4)
        x4 = self.conv_mid_1(x4)
        x4 = self.conv_mid_2(x4)
        x4 = self.conv_mid_3(x4)

        x = self.up1([x4, x3, t])
        x = self.sa4(x)
        x = self.up2([x, x2, t])
        x = self.sa5(x)
        x = self.up3([x, x1, t])
        x = self.sa6(x)

        output = self.conv_output(x)

        return output
