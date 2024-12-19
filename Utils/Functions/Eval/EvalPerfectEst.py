# -*- coding: utf-8 -*-
import sionna
import tensorflow as tf
import numpy as np
from Utils.Functions.Common.io import load_batch_data, get_file_pathes_in_folder
from Utils.Configurations.BasicConfig import BasicConfig

class EvalPerfectEst(tf.keras.Model):
    def __init__(self,
                 config: BasicConfig):
        super(EvalPerfectEst, self).__init__(name='EvalPerfectEst')
        self._config = config

    @tf.function
    def infer_one_batch(self,
                        batch_b: np.ndarray,
                        batch_h_freq: np.ndarray,
                        batch_y: np.ndarray,
                        batch_N0: np.ndarray):
        # Shape of batch_b (float32):        [batch_size, 1, 1, k]
        # Shape of batch_h_freq (complex64): [batch_size, num_ofdm_symbols, fft_size]
        # Shape of batch_y (complex64):      [batch_size, num_ofdm_symbols, fft_size]
        # Shape of batch_N0 (float32):       [batch_size]
        batch_y = sionna.utils.insert_dims(batch_y, num_dims=2, axis=1)
        batch_h_freq = sionna.utils.insert_dims(batch_h_freq, num_dims=4, axis=1)
        batch_var = 0.0
        batch_x_hat, batch_no_eff = self._config._lmmse_equ([batch_y, batch_h_freq, batch_var, batch_N0])
        # Shape of batch_x_hat (complex):  [batch_size, 1, 1, n/num_bits_per_symbol]
        # Shape of batch_no_eff (complex): [batch_size, 1, 1, n/num_bits_per_symbol]

        batch_llr = self._config._demapper([batch_x_hat, batch_no_eff])
        batch_b_hat = self._config._decoder(batch_llr)
        return tf.reduce_sum(tf.cast(tf.math.not_equal(batch_b, batch_b_hat), dtype=tf.int32))/tf.size(batch_b)

    def eval(self, eval_data_pathes: list=[]):
        ber = np.zeros(1, dtype=np.float64)
        for batch_data_path in eval_data_pathes:
            ebno_db, batch_b, batch_tx_codeword_bits, batch_x_rg, batch_pilots_rg, batch_h_freq, batch_y, batch_N0, batch_h_ls_est = load_batch_data(batch_data_path)
            ber += self.infer_one_batch(batch_b, batch_h_freq, batch_y, batch_N0).numpy()
        return ber/len(eval_data_pathes)