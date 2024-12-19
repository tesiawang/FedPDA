# -*- coding: utf-8 -*-
import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

root_path = str(Path(__file__).resolve().parent.parent)
parser = argparse.ArgumentParser()
parser.add_argument("--data_save_folder", type=str, default=root_path+"/Collaborative-Train/Online-Training-Data", required=False)
parser.add_argument("--min_ebNo", type=float, default=10, required=False)
parser.add_argument("--max_ebNo", type=float, default=13, required=False)
parser.add_argument("--num_ebNo_points", type=int, default=11, required=False)

parser.add_argument("--num_users_per_env", type=int, default=12, required=False)
parser.add_argument("--num_batches_per_user", type=int, default=200, required=False)

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
from Utils.Functions.Common.io import save_pkl_file

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)

    # ---------------------------------------------------------------------------- #
    #                              Refresh Data Folder                             #
    # ---------------------------------------------------------------------------- #
    if os.path.exists(Path(run_args.data_save_folder)):
        shutil.rmtree(Path(run_args.data_save_folder))
        os.mkdir(Path(run_args.data_save_folder))
    else:
        os.mkdir(Path(run_args.data_save_folder))

    os.mkdir(Path(run_args.data_save_folder).joinpath("Env-1"))
    os.mkdir(Path(run_args.data_save_folder).joinpath("Env-2"))
    for i in range(run_args.num_users_per_env):
        os.mkdir(Path(run_args.data_save_folder).joinpath("Env-1", "User-%d" % i))
    for i in range(run_args.num_users_per_env, 2*run_args.num_users_per_env):
        os.mkdir(Path(run_args.data_save_folder).joinpath("Env-2", "User-%d" % i))

    # ---------------------------------------------------------------------------- #
    #                                 Generate Data                                #
    # ---------------------------------------------------------------------------- #
    # Environment 1: TDLB-500-5
    # Environment 2: TDLB-150-20
    for i in range(run_args.num_users_per_env):
        generator = BatchDataGenerator(init_config=BasicConfig(tdl_model='B',
                                                               delay_spread=500e-9,
                                                               min_speed=0,
                                                               max_speed=5),
                                       ebNo_dB_range=np.linspace(run_args.min_ebNo, run_args.max_ebNo, run_args.num_ebNo_points))

        for j in range(run_args.num_batches_per_user):
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
            save_pkl_file(batch_data, Path(run_args.data_save_folder).joinpath("Env-1", "User-%d" % i, 'batch_%d.pkl' % j))

    for i in range(run_args.num_users_per_env, 2*run_args.num_users_per_env):
        generator = BatchDataGenerator(init_config=BasicConfig(tdl_model='B',
                                                               delay_spread=150e-9,
                                                               min_speed=15,
                                                               max_speed=20),
                                       ebNo_dB_range=np.linspace(run_args.min_ebNo, run_args.max_ebNo, run_args.num_ebNo_points))

        for j in range(run_args.num_batches_per_user):
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
            save_pkl_file(batch_data, Path(run_args.data_save_folder).joinpath("Env-2", "User-%d" % i, 'batch_%d.pkl' % j))