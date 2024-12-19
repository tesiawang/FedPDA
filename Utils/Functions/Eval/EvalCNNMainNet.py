# -*- coding: utf-8 -*-
import sionna
import tensorflow as tf
import numpy as np
from Utils.Functions.Common.io import load_batch_data
from Utils.Configurations.BasicConfig import BasicConfig

class EvalCNNMainNet(tf.keras.Model):
    def __init__(self,
                 config: BasicConfig=BasicConfig(),
                 main_net: tf.keras.Model=None):
        super(EvalCNNMainNet, self).__init__(name='EvalCNNMainNet')
        self._config = config
        self._main_net = main_net

    @tf.function
    def eval_one_batch(self,
                       batch_b: np.ndarray,
                       batch_pilots_rg: np.ndarray,
                       batch_y: np.ndarray,
                       batch_N0: np.ndarray):
        # Shape of batch_b (float32):           [batch_size, 1, 1, k]
        # Shape of batch_pilots_rg (complex64): [batch_size, num_ofdm_symbols, fft_size, 1]
        # Shape of batch_y (complex64):         [batch_size, num_ofdm_symbols, fft_size]
        # Shape of batch_N0 (float32):          [batch_size]
        batch_llr = self._main_net([batch_pilots_rg, batch_y, batch_N0])
        batch_llr = sionna.utils.insert_dims(batch_llr, 2, axis=1)
        batch_llr = self._config._rg_demapper(batch_llr)
        batch_llr = tf.reshape(batch_llr, [batch_llr.shape[0], 1, 1, self._config._n])
        batch_b_hat = self._config._decoder(batch_llr)
        return tf.reduce_sum(tf.cast(tf.math.not_equal(batch_b, batch_b_hat), dtype=tf.int32))/tf.size(batch_b)

    def eval(self, eval_data_pathes: list=[]):
        ber = np.zeros(1, dtype=np.float64)
        for batch_data_path in eval_data_pathes:
            ebno_db, batch_b, batch_tx_codeword_bits, batch_x_rg, batch_pilots_rg, batch_h_freq, batch_y, batch_N0, batch_h_ls_est = load_batch_data(batch_data_path)
            ber += self.eval_one_batch(batch_b, batch_pilots_rg, batch_y, batch_N0).numpy()
        return ber/len(eval_data_pathes)