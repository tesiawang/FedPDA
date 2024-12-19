# -*- coding: utf-8 -*-
import sionna
import tensorflow as tf
import numpy as np
from pathlib import Path
from Utils.Configurations.BasicConfig import BasicConfig
from Utils.Functions.Common.io import save_pkl_file, load_batch_data

class DataPruner:
    def __init__(self,
                 config: BasicConfig=BasicConfig(),
                 main_net_infer: tf.keras.Model=None,
                 side_net: tf.keras.Model=None):
        self._config = config
        self._main_net_infer = main_net_infer
        self._side_net = side_net

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
        main_net_out = self._main_net_infer([batch_pilots_rg, batch_y, batch_N0])
        batch_llr = self._side_net(inputs=[batch_pilots_rg, batch_y, batch_N0], main_net_out=main_net_out)
        batch_llr = sionna.utils.insert_dims(batch_llr, 2, axis=1)
        batch_llr = self._config._rg_demapper(batch_llr)
        batch_llr = tf.reshape(batch_llr, [batch_llr.shape[0], 1, 1, self._config._n])
        batch_b_hat = self._config._decoder(batch_llr)
        batch_ber = tf.reduce_mean(tf.cast(tf.math.not_equal(batch_b, batch_b_hat), dtype=tf.int32), axis=(-1,-2,-3)) # Shape: [batch_size]
        return batch_ber

    @tf.function
    def get_batch_loss(self,
                       batch_pilots_rg: np.ndarray,
                       batch_y: np.ndarray,
                       batch_N0: np.ndarray,
                       batch_tx_codeword_bits: np.ndarray):
        # Shape of batch_pilots_rg (complex64):       [batch_size, num_ofdm_symbols, fft_size, 1]
        # Shape of batch_y (complex64):               [batch_size, num_ofdm_symbols, fft_size]
        # Shape of batch_N0 (float32):                [batch_size]
        # Shape of batch_tx_codeword_bits (float32):  [batch_size, 1, 1, n]
        main_net_out = self._main_net_infer([batch_pilots_rg, batch_y, batch_N0])
        batch_llr = self._side_net(inputs=[batch_pilots_rg, batch_y, batch_N0], main_net_out=main_net_out)
        batch_llr = sionna.utils.insert_dims(batch_llr, 2, axis=1)
        batch_llr = self._config._rg_demapper(batch_llr)
        batch_llr = tf.reshape(batch_llr, [batch_tx_codeword_bits.shape[0], 1, 1, self._config._n])
        batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(batch_tx_codeword_bits, batch_llr)
        batch_loss = tf.reduce_mean(batch_loss, axis=(-1,-2,-3)) # Shape: [batch_size]
        return batch_loss

    def loss_based_prune(self,
                         pruning_ratio: float=0.8,
                         saved_batch_size: int=32,
                         original_data_pathes: list=[],
                         pruned_data_folder: str=None):
        original_data_pathes = sorted(original_data_pathes)

        # Data pruning based on loss
        all_data_loss = np.array([])
        all_data_ebno_db = np.array([])

        for i in range(len(original_data_pathes)):
            batch_data_path = original_data_pathes[i]
            ebno_db, batch_b, batch_tx_codeword_bits, batch_x_rg, batch_pilots_rg, batch_h_freq, batch_y, batch_N0, batch_h_ls_est = load_batch_data(batch_data_path)
            batch_loss = self.get_batch_loss(batch_pilots_rg, batch_y, batch_N0, batch_tx_codeword_bits).numpy()
            all_data_loss = np.concatenate((all_data_loss, batch_loss))
            all_data_ebno_db = np.concatenate((all_data_ebno_db, ebno_db))

        indices = np.argsort(all_data_loss)[::-1]
        res_indices = indices[:int(len(indices)*(1-pruning_ratio))]

        # Get the pruned data
        res = {i: [] for i in range(len(original_data_pathes))}
        for idx in res_indices:
            batch_data_idx = idx // ebno_db.shape[0]
            idx_in_batch = idx % ebno_db.shape[0]
            res[batch_data_idx].append(idx_in_batch)

        # Save the pruned data
        saved_idx = 0
        for initial in range(len(original_data_pathes)):
            if res[initial] != []:
                ebno_db, batch_b, batch_tx_codeword_bits, batch_x_rg, batch_pilots_rg, batch_h_freq, batch_y, batch_N0, batch_h_ls_est = load_batch_data(original_data_pathes[initial])
                pruned_ebno_db = ebno_db[res[initial]]
                pruned_batch_b = batch_b[res[initial]]
                pruned_batch_tx_codeword_bits = batch_tx_codeword_bits[res[initial]]
                pruned_batch_x_rg = batch_x_rg[res[initial]]
                pruned_batch_pilots_rg = batch_pilots_rg[res[initial]]
                pruned_batch_h_freq = batch_h_freq[res[initial]]
                pruned_batch_y = batch_y[res[initial]]
                pruned_batch_N0 = batch_N0[res[initial]]
                pruned_batch_h_ls_est = batch_h_ls_est[res[initial]]
                break

        for i in range(initial+1, len(original_data_pathes)):
            ebno_db, batch_b, batch_tx_codeword_bits, batch_x_rg, batch_pilots_rg, batch_h_freq, batch_y, batch_N0, batch_h_ls_est = load_batch_data(original_data_pathes[i])
            pruned_ebno_db = np.concatenate((pruned_ebno_db, ebno_db[res[i]]), axis=0)
            pruned_batch_b = np.concatenate((pruned_batch_b, batch_b[res[i]]), axis=0)
            pruned_batch_tx_codeword_bits = np.concatenate((pruned_batch_tx_codeword_bits, batch_tx_codeword_bits[res[i]]), axis=0)
            pruned_batch_x_rg = np.concatenate((pruned_batch_x_rg, batch_x_rg[res[i]]), axis=0)
            pruned_batch_pilots_rg = np.concatenate((pruned_batch_pilots_rg, batch_pilots_rg[res[i]]), axis=0)
            pruned_batch_h_freq = np.concatenate((pruned_batch_h_freq, batch_h_freq[res[i]]), axis=0)
            pruned_batch_y = np.concatenate((pruned_batch_y, batch_y[res[i]]), axis=0)
            pruned_batch_N0 = np.concatenate((pruned_batch_N0, batch_N0[res[i]]), axis=0)
            pruned_batch_h_ls_est = np.concatenate((pruned_batch_h_ls_est, batch_h_ls_est[res[i]]), axis=0)

            if pruned_ebno_db.shape[0] > saved_batch_size:
                save_pkl_file({'ebno_db': pruned_ebno_db[:saved_batch_size],
                               'batch_b': pruned_batch_b[:saved_batch_size],
                               'batch_tx_codeword_bits': pruned_batch_tx_codeword_bits[:saved_batch_size],
                               'batch_x_rg': pruned_batch_x_rg[:saved_batch_size],
                               'batch_pilots_rg': pruned_batch_pilots_rg[:saved_batch_size],
                               'batch_h_freq': pruned_batch_h_freq[:saved_batch_size],
                               'batch_y': pruned_batch_y[:saved_batch_size],
                               'batch_N0': pruned_batch_N0[:saved_batch_size],
                               'batch_h_ls_est': pruned_batch_h_ls_est[:saved_batch_size]},
                               Path(pruned_data_folder).joinpath("batch_%d.pkl" % saved_idx))
                pruned_ebno_db = pruned_ebno_db[saved_batch_size:]
                pruned_batch_b = pruned_batch_b[saved_batch_size:]
                pruned_batch_tx_codeword_bits = pruned_batch_tx_codeword_bits[saved_batch_size:]
                pruned_batch_x_rg = pruned_batch_x_rg[saved_batch_size:]
                pruned_batch_pilots_rg = pruned_batch_pilots_rg[saved_batch_size:]
                pruned_batch_h_freq = pruned_batch_h_freq[saved_batch_size:]
                pruned_batch_y = pruned_batch_y[saved_batch_size:]
                pruned_batch_N0 = pruned_batch_N0[saved_batch_size:]
                pruned_batch_h_ls_est = pruned_batch_h_ls_est[saved_batch_size:]
                saved_idx += 1

        if pruned_ebno_db.shape[0] > 0:
            save_pkl_file({'ebno_db': pruned_ebno_db,
                           'batch_b': pruned_batch_b,
                           'batch_tx_codeword_bits': pruned_batch_tx_codeword_bits,
                           'batch_x_rg': pruned_batch_x_rg,
                           'batch_pilots_rg': pruned_batch_pilots_rg,
                           'batch_h_freq': pruned_batch_h_freq,
                           'batch_y': pruned_batch_y,
                           'batch_N0': pruned_batch_N0,
                           'batch_h_ls_est': pruned_batch_h_ls_est},
                           Path(pruned_data_folder).joinpath("batch_%d.pkl" % saved_idx))
        return {'all_data_loss': all_data_loss, 'all_data_ebno_db': all_data_ebno_db}