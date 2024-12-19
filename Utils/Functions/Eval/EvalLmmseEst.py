# -*- coding: utf-8 -*-
import sionna
import tensorflow as tf
import numpy as np
from Utils.Functions.Common.io import load_batch_data, get_file_pathes_in_folder
from Utils.Configurations.BasicConfig import BasicConfig

# def sample_channel(batch_size:int=32,
#                    config: BasicConfig=BasicConfig()):
#     cir = config._tdl(batch_size, config._rg.num_ofdm_symbols, 1/config._rg.ofdm_symbol_duration)
#     batch_h_freq = sionna.channel.cir_to_ofdm_channel(config._frequencies, *cir, normalize=True)
#     return batch_h_freq[:,0,:,0,0]

# @tf.function
# def estimate_covariance_matrices(num_iterations:int=100,
#                                  batch_size:int=1000,
#                                  fft_size:int=264,
#                                  num_ofdm_symbols:int=14):
#     freq_cov_mat = tf.zeros([fft_size, fft_size], tf.complex64)
#     time_cov_mat = tf.zeros([num_ofdm_symbols, num_ofdm_symbols], tf.complex64)
#     for _ in tf.range(num_iterations):
#         # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
#         h_samples = sample_channel(batch_size)

#         # ----------------------- Estimate frequency covariance ---------------------- #
#         h_samples_ = tf.transpose(h_samples, [0,1,3,2])                   # Shape of [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
#         freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True) # Shape of [batch size, num_rx_ant, fft_size, fft_size]
#         freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0,1))         # Shape of [fft_size, fft_size]
#         freq_cov_mat += freq_cov_mat_                                     # Shape of [fft_size, fft_size]

#         # ------------------------- Estimate time covariance ------------------------- #
#         time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)   # Shape of [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
#         time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0,1))         # Shape of [num_ofdm_symbols, num_ofdm_symbols]
#         time_cov_mat += time_cov_mat_                                     # Shape of [num_ofdm_symbols, num_ofdm_symbols]

#     freq_cov_mat /= tf.complex(tf.cast(num_ofdm_symbols*num_iterations, tf.float32), 0.0)
#     time_cov_mat /= tf.complex(tf.cast(fft_size*num_iterations, tf.float32), 0.0)
#     return freq_cov_mat, time_cov_mat

class EvalLmmseEst(tf.keras.Model):
    def __init__(self,
                 config: BasicConfig):
        super(EvalLmmseEst, self).__init__(name='EvalLmmseEst')
        self._config = config
        self._time_cov_mat = sionna.ofdm.tdl_time_cov_mat(model=config._tdl_model,
                                                          speed=config._max_speed,
                                                          carrier_frequency=config._carrier_frequency,
                                                          ofdm_symbol_duration=config._rg.ofdm_symbol_duration,
                                                          num_ofdm_symbols=config._num_ofdm_symbols)
        self._freq_cov_mat = sionna.ofdm.tdl_freq_cov_mat(model=config._tdl_model,
                                                          subcarrier_spacing=config._subcarrier_spacing,
                                                          fft_size=config._fft_size,
                                                          delay_spread=config._delay_spread)
        lmmse_interpolator = sionna.ofdm.LMMSEInterpolator(pilot_pattern=config._rg.pilot_pattern,
                                                           cov_mat_time=self._time_cov_mat,
                                                           cov_mat_freq=self._freq_cov_mat)
        self._lmmse_est = sionna.ofdm.LSChannelEstimator(config._rg, interpolator=lmmse_interpolator)

    @tf.function
    def infer_one_batch(self,
                        batch_b: np.ndarray,
                        batch_y: np.ndarray,
                        batch_N0: np.ndarray):
        # Shape of batch_b (float32):    [batch_size, 1, 1, k]
        # Shape of batch_y (complex64):  [batch_size, num_ofdm_symbols, fft_size]
        # Shape of batch_N0 (float32):   [batch_size]

        batch_y = sionna.utils.insert_dims(batch_y, num_dims=2, axis=1)
        batch_h_lmmse_est, batch_var_lmmse_est = self._lmmse_est([batch_y, batch_N0]) # Shape of batch_h_lmmse_est (complex64): [batch_size, 1, 1, 1, 1, num_ofdm_symbols, fft_size]
        batch_x_hat, batch_no_eff = self._config._lmmse_equ([batch_y, batch_h_lmmse_est, batch_var_lmmse_est, batch_N0])
        # Shape of batch_x_hat (complex64):  [batch_size, 1, 1, n/num_bits_per_symbol]
        # Shape of batch_no_eff (complex64): [batch_size, 1, 1, n/num_bits_per_symbol]

        batch_llr = self._config._demapper([batch_x_hat, batch_no_eff])
        batch_b_hat = self._config._decoder(batch_llr)
        return tf.reduce_sum(tf.cast(tf.math.not_equal(batch_b, batch_b_hat), dtype=tf.int32))/tf.size(batch_b)

    def eval(self, eval_data_pathes: list=[]):
        ber = np.zeros(1, dtype=np.float64)
        for batch_data_path in eval_data_pathes:
            ebno_db, batch_b, batch_tx_codeword_bits, batch_x_rg, batch_pilots_rg, batch_h_freq, batch_y, batch_N0, batch_h_ls_est = load_batch_data(batch_data_path)
            ber += self.infer_one_batch(batch_b, batch_y, batch_N0).numpy()
        return ber/len(eval_data_pathes)