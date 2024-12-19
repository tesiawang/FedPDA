# -*- coding: utf-8 -*-
import sionna
import tensorflow as tf
import numpy as np
from pathlib import Path
from Utils.Configurations.BasicConfig import BasicConfig
from Utils.Functions.Common.io import load_batch_data, log_print, save_pkl_file

class TrainCNNMainNet:
    def __init__(self,
                 config: BasicConfig=BasicConfig(),
                 main_net: tf.keras.Model=None,
                 num_epoches: int=30,
                 lr: float=1e-3,
                 train_data_pathes: str=None,
                 val_data_pathes: str=None,
                 model_save_folder: str=None):
        self._config = config
        self._main_net = main_net

        # Training parameters
        self._num_epoches = num_epoches
        self._lr = lr
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self._train_data_pathes = train_data_pathes
        self._val_data_pathes = val_data_pathes
        self._model_save_folder = model_save_folder
        self._epoch_loss_list = np.zeros(num_epoches, dtype=np.float32)

    @tf.function
    def train_one_batch(self,
                        batch_pilots_rg: np.ndarray,
                        batch_y: np.ndarray,
                        batch_N0: np.ndarray,
                        batch_tx_codeword_bits: np.ndarray):
        # Shape of batch_pilots_rg (complex64):       [batch_size, num_ofdm_symbols, fft_size, 1]
        # Shape of batch_y (complex64):               [batch_size, num_ofdm_symbols, fft_size]
        # Shape of batch_N0 (float32):                [batch_size]
        # Shape of batch_tx_codeword_bits (float32):  [batch_size, 1, 1, n]
        with tf.GradientTape() as tape:
            batch_llr = self._main_net([batch_pilots_rg, batch_y, batch_N0])
            batch_llr = sionna.utils.insert_dims(batch_llr, 2, axis=1)
            batch_llr = self._config._rg_demapper(batch_llr)
            batch_llr = tf.reshape(batch_llr, [batch_tx_codeword_bits.shape[0], 1, 1, self._config._n])
            batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(batch_tx_codeword_bits, batch_llr)
            batch_loss = tf.reduce_mean(batch_loss)
        # Computing and applying gradients
        grads = tape.gradient(batch_loss, self._main_net.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._main_net.trainable_variables))
        return batch_loss

    def train(self, stopping_num:int=5, save_mode:str='epoch'):
        checkpoint_idx = 0
        ber_array = np.zeros(self._max_epoches, dtype=np.float128)
        stopping_counter = 0

        for epoch in range(self._num_epoches):
            # -------------------------------- Validation -------------------------------- #
            if self._val_data_pathes is not None:
                for batch_data_path in self._val_data_pathes:
                    ebno_db, batch_b, batch_tx_codeword_bits, batch_x_rg, batch_pilots_rg, batch_h_freq, batch_y, batch_N0, batch_h_ls_est = load_batch_data(batch_data_path)
                    ber_array[epoch] += self.eval_one_batch(batch_b, batch_pilots_rg, batch_y, batch_N0).numpy()
                ber_array[epoch] = ber_array[epoch]/len(self._val_data_pathes)
                save_pkl_file(ber_array, Path(self._model_save_folder).joinpath("ber_array.pkl"))

                # Early stopping
                if epoch==0:
                    stopping_counter = 0
                else:
                    if ber_array[epoch] < np.min(ber_array[:epoch]):
                        stopping_counter = 0
                    else:
                        stopping_counter += 1
                if stopping_counter==stopping_num:
                    break
            else:
                log_print("|====> No validation data is provided, skip the validation process.", color='y')

            # --------------------------------- Training --------------------------------- #
            epoch_loss = 0.
            np.random.shuffle(self._train_data_pathes)
            for batch_data_path in self._train_data_pathes:
                ebno_db, batch_b, batch_tx_codeword_bits, batch_x_rg, batch_pilots_rg, batch_h_freq, batch_y, batch_N0, batch_h_ls_est = load_batch_data(batch_data_path)
                batch_loss = self.train_one_batch(batch_pilots_rg,
                                                  batch_y,
                                                  batch_N0,
                                                  batch_tx_codeword_bits)
                # Save checkpoints
                if save_mode=='iteration':
                    save_pkl_file(self._main_net.get_weights(), Path(self._model_save_folder).joinpath("updated_parameters_%d.pkl" % checkpoint_idx))
                    checkpoint_idx += 1

                epoch_loss += batch_loss
            self._epoch_loss_list[epoch] = epoch_loss/(len(self._train_data_pathes)*batch_N0.shape[0])
            log_print("|====> Main Net Training, Epoch %d, Loss: %.4f" % (epoch, self._epoch_loss_list[epoch]), color='g')

            # Save checkpoints
            if save_mode=='epoch':
                save_pkl_file(self._main_net.get_weights(), Path(self._model_save_folder).joinpath("updated_parameters_%d.pkl" % checkpoint_idx))
                checkpoint_idx += 1

            # Save loss info
            save_pkl_file(self._epoch_loss_list, Path(self._model_save_folder).joinpath("epoch_loss_list.pkl"))
        return self._main_net.get_weights()