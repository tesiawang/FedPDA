# -*- coding: utf-8 -*-
import sionna
import tensorflow as tf
from Utils.Models.CNN.CNNMainNetSRB import CNNMainNetSRB

class CNNMainNet(tf.keras.Model):
    def __init__(self):
        super(CNNMainNet, self).__init__(name='CNNMainNet')
        # Input convolution
        self._input_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding='same', activation=None)
        # Residual blocks
        self._res_block_0 = CNNMainNetSRB(num_filters=64, dilation=[1,1])
        self._res_block_1 = CNNMainNetSRB(num_filters=64, dilation=[1,1])
        self._res_block_2 = CNNMainNetSRB(num_filters=128, dilation=[2,3], skip_conv11 = True)
        self._res_block_3 = CNNMainNetSRB(num_filters=128, dilation=[2,3])
        self._res_block_4 = CNNMainNetSRB(num_filters=256, dilation=[2,3], skip_conv11 = True)
        self._res_block_5 = CNNMainNetSRB(num_filters=256, dilation=[3,6])
        self._res_block_6 = CNNMainNetSRB(num_filters=256, dilation=[2,3])
        self._res_block_7 = CNNMainNetSRB(num_filters=128, dilation=[2,3], skip_conv11 = True)
        self._res_block_8 = CNNMainNetSRB(num_filters=128, dilation=[2,3])
        self._res_block_9 = CNNMainNetSRB(num_filters=64, dilation=[1,1], skip_conv11 = True)
        self._res_block_10 = CNNMainNetSRB(num_filters=64, dilation=[1,1])
        # Output conv
        self._output_conv = tf.keras.layers.Conv2D(filters=4,
                                                   kernel_size=[3,3],
                                                   padding='same',
                                                   activation=None)
    def call(self, inputs):
        batch_pilots_rg, batch_y, batch_N0 = inputs
        # Shape of batch_pilots_rg (complex64): [batch_size, num_ofdm_symbols, fft_size, 1]
        # Shape of batch_y (complex64):         [batch_size, num_ofdm_symbols, fft_size]
        # Shape of batch_N0 (float32):          [batch_size]
        batch_y = sionna.utils.insert_dims(batch_y, num_dims=1, axis=-1)
        batch_N0 = sionna.utils.log10(batch_N0)
        batch_N0 = sionna.utils.insert_dims(batch_N0, num_dims=3, axis=1)
        batch_N0 = tf.tile(batch_N0, [1, batch_y.shape[1], batch_y.shape[2], 1])
        z = tf.concat([tf.math.real(batch_y),
                       tf.math.imag(batch_y),
                       tf.math.real(batch_pilots_rg),
                       tf.math.imag(batch_pilots_rg),
                       batch_N0], axis=-1)
        # Input conv
        input_conv_out = self._input_conv(z)
        # Residual blocks
        res0_out = self._res_block_0(inputs=input_conv_out)
        res1_out = self._res_block_1(inputs=res0_out)
        res2_out = self._res_block_2(inputs=res1_out)
        res3_out = self._res_block_3(inputs=res2_out)
        res4_out = self._res_block_4(inputs=res3_out)
        res5_out = self._res_block_5(inputs=res4_out)
        res6_out = self._res_block_6(inputs=res5_out)
        res7_out = self._res_block_7(inputs=res6_out)
        res8_out = self._res_block_8(inputs=res7_out)
        res9_out = self._res_block_9(inputs=res8_out)
        res10_out = self._res_block_10(inputs=res9_out)
        # Output conv
        output = self._output_conv(res10_out)
        return output

class CNNMainNetInfer(tf.keras.Model):
    def __init__(self):
        super(CNNMainNetInfer, self).__init__(name='CNNMainNetInfer')
        # Input convolution
        self._input_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding='same', activation=None)
        # Residual blocks
        self._res_block_0 = CNNMainNetSRB(num_filters=64, dilation=[1,1])
        self._res_block_1 = CNNMainNetSRB(num_filters=64, dilation=[1,1])
        self._res_block_2 = CNNMainNetSRB(num_filters=128, dilation=[2,3], skip_conv11 = True)
        self._res_block_3 = CNNMainNetSRB(num_filters=128, dilation=[2,3])
        self._res_block_4 = CNNMainNetSRB(num_filters=256, dilation=[2,3], skip_conv11 = True)
        self._res_block_5 = CNNMainNetSRB(num_filters=256, dilation=[3,6])
        self._res_block_6 = CNNMainNetSRB(num_filters=256, dilation=[2,3])
        self._res_block_7 = CNNMainNetSRB(num_filters=128, dilation=[2,3], skip_conv11 = True)
        self._res_block_8 = CNNMainNetSRB(num_filters=128, dilation=[2,3])
        self._res_block_9 = CNNMainNetSRB(num_filters=64, dilation=[1,1], skip_conv11 = True)
        self._res_block_10 = CNNMainNetSRB(num_filters=64, dilation=[1,1])
        # Output conv
        self._output_conv = tf.keras.layers.Conv2D(filters=4,
                                                   kernel_size=[3,3],
                                                   padding='same',
                                                   activation=None)
    def call(self, inputs):
        batch_pilots_rg, batch_y, batch_N0 = inputs
        # Shape of batch_pilots_rg (complex64): [batch_size, num_ofdm_symbols, fft_size, 1]
        # Shape of batch_y (complex64):         [batch_size, num_ofdm_symbols, fft_size]
        # Shape of batch_N0 (float32):          [batch_size]
        batch_y = sionna.utils.insert_dims(batch_y, num_dims=1, axis=-1)
        batch_N0 = sionna.utils.log10(batch_N0)
        batch_N0 = sionna.utils.insert_dims(batch_N0, num_dims=3, axis=1)
        batch_N0 = tf.tile(batch_N0, [1, batch_y.shape[1], batch_y.shape[2], 1])
        z = tf.concat([tf.math.real(batch_y),
                       tf.math.imag(batch_y),
                       tf.math.real(batch_pilots_rg),
                       tf.math.imag(batch_pilots_rg),
                       batch_N0], axis=-1)
        # Input conv
        input_conv_out = self._input_conv(z)
        # Residual blocks
        res0_middle_out, res0_out = self._res_block_0(inputs=input_conv_out, mode='all')
        res1_middle_out, res1_out  = self._res_block_1(inputs=res0_out, mode='all')
        res2_middle_out, res2_out  = self._res_block_2(inputs=res1_out, mode='all')
        res3_middle_out, res3_out  = self._res_block_3(inputs=res2_out, mode='all')
        res4_middle_out, res4_out  = self._res_block_4(inputs=res3_out, mode='all')
        res5_middle_out, res5_out  = self._res_block_5(inputs=res4_out, mode='all')
        res6_middle_out, res6_out  = self._res_block_6(inputs=res5_out, mode='all')
        res7_middle_out, res7_out  = self._res_block_7(inputs=res6_out, mode='all')
        res8_middle_out, res8_out  = self._res_block_8(inputs=res7_out, mode='all')
        res9_middle_out, res9_out  = self._res_block_9(inputs=res8_out, mode='all')
        res10_middle_out, res10_out  = self._res_block_10(inputs=res9_out, mode='all')
        # Output conv
        output = self._output_conv(res10_out)
        return [input_conv_out, res0_middle_out, res0_out, res1_middle_out, res1_out, res2_middle_out, res2_out, \
                res3_middle_out, res3_out, res4_middle_out, res4_out, res5_middle_out, res5_out, res6_middle_out, res6_out, \
                res7_middle_out, res7_out, res8_middle_out, res8_out, res9_middle_out, res9_out, res10_middle_out, res10_out, output]