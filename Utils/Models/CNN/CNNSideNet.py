# -*- coding: utf-8 -*-
import sionna
import tensorflow as tf
from Utils.Models.CNN.CNNSideNetSRB import CNNSideNetSRB

def zero_layer(_):
    return 0

class CNNSideNet(tf.keras.Model):
    def __init__(self):
        super(CNNSideNet, self).__init__(name='CNNSideNet')
        SRB_num_filters = [16, 16, 16, 32, 32, 64, 64, 64, 32, 32, 16, 16]
        skip_connection_num_filters = [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                       64, 64, 64, 64, 32, 32, 32, 32, 16, 16, 16, 16]
        # Input convolution
        self._input_conv = tf.keras.layers.Conv2D(filters=SRB_num_filters[0],
                                                  kernel_size=[3,3],
                                                  padding='same',
                                                  activation=None)
        # Residual blocks
        self._res_block_0 = CNNSideNetSRB(num_filters=SRB_num_filters[1], dilation=[1,1])
        self._res_block_1 = CNNSideNetSRB(num_filters=SRB_num_filters[2], dilation=[1,1])
        self._res_block_2 = CNNSideNetSRB(num_filters=SRB_num_filters[3], dilation=[2,3], skip_conv11 = True)
        self._res_block_3 = CNNSideNetSRB(num_filters=SRB_num_filters[4], dilation=[2,3])
        self._res_block_4 = CNNSideNetSRB(num_filters=SRB_num_filters[5], dilation=[2,3], skip_conv11 = True)
        self._res_block_5 = CNNSideNetSRB(num_filters=SRB_num_filters[6], dilation=[3,6])
        self._res_block_6 = CNNSideNetSRB(num_filters=SRB_num_filters[7], dilation=[2,3])
        self._res_block_7 = CNNSideNetSRB(num_filters=SRB_num_filters[8], dilation=[2,3], skip_conv11 = True)
        self._res_block_8 = CNNSideNetSRB(num_filters=SRB_num_filters[9], dilation=[2,3])
        self._res_block_9 = CNNSideNetSRB(num_filters=SRB_num_filters[10], dilation=[1,1], skip_conv11 = True)
        self._res_block_10 = CNNSideNetSRB(num_filters=SRB_num_filters[11], dilation=[1,1])

        # Output conv
        self._output_conv = tf.keras.layers.Conv2D(filters=4,
                                                   kernel_size=[3,3],
                                                   padding='same',
                                                   activation=None)
        # Skip connections
        self._skip_connections = []
        for i in range(23):
            if skip_connection_num_filters[i]==0:
                self._skip_connections.append(zero_layer)
            else:
                self._skip_connections.append(tf.keras.layers.Conv2D(filters=skip_connection_num_filters[i],
                                                                     kernel_size=[1,1],
                                                                     padding='same',
                                                                     activation=None))
    def call(self, inputs, main_net_out):
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
        input_conv_out = self._input_conv(z) + self._skip_connections[0](main_net_out.pop(0))
        # Residual blocks
        res0_out = self._res_block_0(inputs=input_conv_out,
                                     main_net_SRB_middle_out=self._skip_connections[1](main_net_out.pop(0))) + self._skip_connections[2](main_net_out.pop(0))
        res1_out = self._res_block_1(inputs=res0_out,
                                     main_net_SRB_middle_out=self._skip_connections[3](main_net_out.pop(0))) + self._skip_connections[4](main_net_out.pop(0))
        res2_out = self._res_block_2(inputs=res1_out,
                                     main_net_SRB_middle_out=self._skip_connections[5](main_net_out.pop(0))) + self._skip_connections[6](main_net_out.pop(0))
        res3_out = self._res_block_3(inputs=res2_out,
                                     main_net_SRB_middle_out=self._skip_connections[7](main_net_out.pop(0))) + self._skip_connections[8](main_net_out.pop(0))
        res4_out = self._res_block_4(inputs=res3_out,
                                     main_net_SRB_middle_out=self._skip_connections[9](main_net_out.pop(0))) + self._skip_connections[10](main_net_out.pop(0))
        res5_out = self._res_block_5(inputs=res4_out,
                                     main_net_SRB_middle_out=self._skip_connections[11](main_net_out.pop(0))) + self._skip_connections[12](main_net_out.pop(0))
        res6_out = self._res_block_6(inputs=res5_out,
                                     main_net_SRB_middle_out=self._skip_connections[13](main_net_out.pop(0))) + self._skip_connections[14](main_net_out.pop(0))
        res7_out = self._res_block_7(inputs=res6_out, 
                                     main_net_SRB_middle_out=self._skip_connections[15](main_net_out.pop(0))) + self._skip_connections[16](main_net_out.pop(0))
        res8_out = self._res_block_8(inputs=res7_out,
                                     main_net_SRB_middle_out=self._skip_connections[17](main_net_out.pop(0))) + self._skip_connections[18](main_net_out.pop(0))
        res9_out = self._res_block_9(inputs=res8_out,
                                     main_net_SRB_middle_out=self._skip_connections[19](main_net_out.pop(0))) + self._skip_connections[20](main_net_out.pop(0))
        res10_out = self._res_block_10(inputs=res9_out,
                                       main_net_SRB_middle_out=self._skip_connections[21](main_net_out.pop(0))) + self._skip_connections[22](main_net_out.pop(0))
        # Output conv
        output = self._output_conv(res10_out) + main_net_out.pop(0)
        return output