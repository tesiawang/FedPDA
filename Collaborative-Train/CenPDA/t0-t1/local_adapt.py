# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

root_path = str(Path(__file__).resolve().parent.parent.parent.parent)
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_main_net_path", type=str, default=root_path+"/Offline-Pretrain/Pretrained-Models/CNNMainNet/updated_parameters_29.pkl", required=False)
parser.add_argument("--pretrained_side_net_path", type=str, default=root_path+"/Offline-Pretrain/Pretrained-Models/CNNSideNet/Pretrained-CNNSideNet/updated_parameters_29.pkl", required=False)
parser.add_argument("--model_save_folder", type=str, default=root_path+"/Collaborative-Train/CenPDA/t0-t1/Local-Adapted-Models", required=False)
parser.add_argument("--max_epoches", type=int, default=30, required=False)
parser.add_argument("--lr", type=float, default=1e-3, required=False)
parser.add_argument("--training_data_folder", type=str, default=root_path+"/Collaborative-Train/Online-Training-Data/Env-1", required=False)
parser.add_argument("--val_data_folder", type=str, default=root_path+"/Collaborative-Train/Online-Val-Data/Env-1", required=False)
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
from Utils.Functions.Common.io import get_file_pathes_in_folder, load_pkl_file
from Utils.Models.CNN.CNNMainNet import CNNMainNetInfer
from Utils.Models.CNN.CNNSideNet import CNNSideNet
from Utils.Functions.Train.TrainCNNSideNet import TrainCNNSideNet
from Utils.Configurations.BasicConfig import BasicConfig

if __name__=='__main__':
    critical_users = [i for i in range(12)]

    cnn_main_net = CNNMainNetInfer()
    batch_pilots_rg = np.zeros((16, BasicConfig()._num_ofdm_symbols, BasicConfig()._fft_size, 1), dtype=np.complex64)
    batch_y = np.zeros((16, BasicConfig()._num_ofdm_symbols, BasicConfig()._fft_size), dtype=np.complex64)
    batch_N0 = np.zeros(16, dtype=np.float32)
    main_net_out = cnn_main_net([batch_pilots_rg, batch_y, batch_N0])
    cnn_main_net.set_weights(load_pkl_file(run_args.pretrained_main_net_path))

    cnn_side_net = CNNSideNet()
    cnn_side_net([batch_pilots_rg, batch_y, batch_N0], main_net_out=main_net_out)

    for user_id in critical_users:
        cnn_side_net.set_weights(load_pkl_file(run_args.pretrained_side_net_path))
        train_data_pathes = get_file_pathes_in_folder(Path(run_args.training_data_folder).joinpath("User-"+str(user_id)))
        val_data_pathes = get_file_pathes_in_folder(Path(run_args.val_data_folder).joinpath("User-"+str(user_id)))
        obj = TrainCNNSideNet(config=BasicConfig(),
                              main_net_infer=cnn_main_net,
                              side_net=cnn_side_net,
                              max_epoches=run_args.max_epoches,
                              lr=run_args.lr,
                              train_data_pathes=train_data_pathes,
                              val_data_pathes=val_data_pathes,
                              model_save_folder=Path(run_args.model_save_folder).joinpath("User-"+str(user_id)))
        obj.train(stopping_num=5)