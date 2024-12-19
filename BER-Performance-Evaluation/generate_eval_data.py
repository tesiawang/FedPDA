# -*- coding: utf-8 -*-
import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

root_path = str(Path(__file__).resolve().parent.parent)
parser = argparse.ArgumentParser()
parser.add_argument("--tdl_model", type=str, default="B", required=False)
parser.add_argument("--delay_spread", type=float, default=150e-9, required=False)
parser.add_argument("--min_speed", type=float, default=15, required=False)
parser.add_argument("--max_speed", type=float, default=20, required=False)

parser.add_argument("--min_ebNo", type=float, default=4, required=False)
parser.add_argument("--max_ebNo", type=float, default=12, required=False)
parser.add_argument("--num_ebNo_points", type=int, default=9, required=False)
parser.add_argument("--num_batches_per_dB", type=int, default=100, required=False)
parser.add_argument("--data_save_folder", type=str, default=root_path+"/BER-Performance-Evaluation/Eval-Data/TDLB-150-20", required=False)

parser.add_argument("--gpu_number", type=int, default=0, required=False)
parser.add_argument("--memory_limit", type=float, default=24, required=False)
run_args = parser.parse_args()

# -------------------------------- GPU CONFIG -------------------------------- #
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(
        gpus[run_args.gpu_number],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(1024*run_args.memory_limit))])
  except RuntimeError as e:
    print(e)
# ---------------------------------------------------------------------------- #

import sys
sys.path.append(root_path)
from Utils.Functions.Common.BatchDataGenerator import BatchDataGenerator
from Utils.Configurations.BasicConfig import BasicConfig
from Utils.Functions.Common.io import save_pkl_file, log_print

if __name__ == '__main__':
    tf.random.set_seed(44)
    np.random.seed(44)

    # ---------------------------------------------------------------------------- #
    #                              Refresh Data Folder                             #
    # ---------------------------------------------------------------------------- #
    if os.path.exists(Path(run_args.data_save_folder)):
        shutil.rmtree(Path(run_args.data_save_folder))
        os.mkdir(Path(run_args.data_save_folder))
    else:
        os.mkdir(Path(run_args.data_save_folder))
    for dB in np.linspace(run_args.min_ebNo, run_args.max_ebNo, run_args.num_ebNo_points):
        os.mkdir(Path(run_args.data_save_folder).joinpath('%.2f' % dB))

    # ---------------------------------------------------------------------------- #
    #                                 Generate Data                                #
    # ---------------------------------------------------------------------------- #
    for dB in np.linspace(run_args.min_ebNo, run_args.max_ebNo, run_args.num_ebNo_points):
        log_print('Generating data for EbNo = %.2f dB' % dB, color='g')
        config = BasicConfig()
        generator = BatchDataGenerator(init_config=BasicConfig(tdl_model=run_args.tdl_model,
                                                               delay_spread=run_args.delay_spread,
                                                               min_speed=run_args.min_speed,
                                                               max_speed=run_args.max_speed),
                                   ebNo_dB_range=np.linspace(dB, dB, 1))
        for j in range(run_args.num_batches_per_dB):
            ebno_db, batch_b, batch_tx_codeword_bits, batch_x_rg, batch_pilots_rg, batch_h_freq, batch_y, batch_N0, batch_h_ls_est = generator.receive_data(32)
            batch_data = dict()
            batch_data['ebno_db'] = ebno_db
            batch_data['batch_b'] = batch_b
            batch_data['batch_tx_codeword_bits'] = batch_tx_codeword_bits
            batch_data['batch_x_rg'] = batch_x_rg
            batch_data['batch_pilots_rg'] = batch_pilots_rg
            batch_data['batch_h_freq'] = batch_h_freq
            batch_data['batch_y'] = batch_y
            batch_data['batch_N0'] = batch_N0
            batch_data['batch_h_ls_est'] = batch_h_ls_est
            save_pkl_file(batch_data, Path(run_args.data_save_folder).joinpath('%.2f' % dB, 'batch_%d.pkl' % j))