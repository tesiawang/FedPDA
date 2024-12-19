# -*- coding: utf-8 -*-
import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

root_path = str(Path(__file__).resolve().parent.parent.parent.parent)
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_main_net_path", type=str, default=root_path+"/Offline-Pretrain/Pretrained-Models/CNNMainNet/updated_parameters_29.pkl", required=False)
parser.add_argument("--side_net_path", type=str, default=root_path+"/Collaborative-Train/FedPDA/t0-t1/Cache/global_model_parameters_27.pkl", required=False)
parser.add_argument("--original_data_folder", type=str, default=root_path+"/Collaborative-Train/Online-Training-Data/Env-1", required=False)
parser.add_argument("--pruned_data_folder", type=str, default=root_path+"/Collaborative-Train/FedPDA/t0-t1/Pruned-Data", required=False)
parser.add_argument("--gpu_number", type=int, default=0, required=False)
parser.add_argument("--memory_limit", type=float, default=30, required=False)
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
from Utils.Functions.Common.io import get_file_pathes_in_folder, load_pkl_file, save_pkl_file
from Utils.Models.CNN.CNNMainNet import CNNMainNetInfer
from Utils.Models.CNN.CNNSideNet import CNNSideNet
from Utils.Functions.Common.DataPruner import DataPruner
from Utils.Configurations.BasicConfig import BasicConfig

# -------------------------------- GPU CONFIG -------------------------------- #
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(1024*run_args.memory_limit))])
  except RuntimeError as e:
    print(e)
# ---------------------------------------------------------------------------- #

if __name__=='__main__':
    if os.path.exists(run_args.pruned_data_folder):
        shutil.rmtree(run_args.pruned_data_folder)
        os.mkdir(run_args.pruned_data_folder)
    else:
        os.mkdir(run_args.pruned_data_folder)
    os.mkdir(Path(run_args.pruned_data_folder).joinpath('0.80'))
    os.mkdir(Path(run_args.pruned_data_folder).joinpath('0.90'))
    os.mkdir(Path(run_args.pruned_data_folder).joinpath('0.95'))

    # ------------------------------------ #
    #              Load Models             #
    # ------------------------------------ #
    cnn_main_net = CNNMainNetInfer()
    batch_pilots_rg = np.zeros((16, BasicConfig()._num_ofdm_symbols, BasicConfig()._fft_size, 1), dtype=np.complex64)
    batch_y = np.zeros((16, BasicConfig()._num_ofdm_symbols, BasicConfig()._fft_size), dtype=np.complex64)
    batch_N0 = np.zeros(16, dtype=np.float32)
    main_net_out = cnn_main_net([batch_pilots_rg, batch_y, batch_N0])
    cnn_main_net.set_weights(load_pkl_file(run_args.pretrained_main_net_path))

    cnn_side_net = CNNSideNet()
    cnn_side_net([batch_pilots_rg, batch_y, batch_N0], main_net_out=main_net_out)
    cnn_side_net.set_weights(load_pkl_file(run_args.side_net_path))

    # ------------------------------------ #
    #            Prune the Data            #
    # ------------------------------------ #
    pruner = DataPruner(main_net_infer=cnn_main_net, side_net=cnn_side_net)
    pruner.loss_based_prune(pruning_ratio=0.8,
                            original_data_pathes=get_file_pathes_in_folder(run_args.original_data_folder),
                            pruned_data_folder=Path(run_args.pruned_data_folder).joinpath('0.80'))
    pruner.loss_based_prune(pruning_ratio=0.9,
                            original_data_pathes=get_file_pathes_in_folder(run_args.original_data_folder),
                            pruned_data_folder=Path(run_args.pruned_data_folder).joinpath('0.90'))
    all_data_loss = pruner.loss_based_prune(pruning_ratio=0.95,
                                            original_data_pathes=get_file_pathes_in_folder(run_args.original_data_folder),
                                            pruned_data_folder=Path(run_args.pruned_data_folder).joinpath('0.95'))
    save_pkl_file(all_data_loss, Path(run_args.pruned_data_folder).joinpath('all_data_loss.pkl'))