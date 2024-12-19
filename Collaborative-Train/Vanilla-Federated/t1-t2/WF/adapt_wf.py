# -*- coding: utf-8 -*-
import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

root_path = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_main_net_path", type=str, default=root_path+"/Offline-Pretrain/Pretrained-Models/CNNMainNet/updated_parameters_29.pkl", required=False)
parser.add_argument("--side_net_path", type=str, default=root_path+"/Collaborative-Train/Vanilla-Federated/t0-t1/Cache/global_model_parameters_27.pkl", required=False)
parser.add_argument("--cache_folder", type=str, default=root_path+"/Collaborative-Train/Vanilla-Federated/t1-t2/WF/Cache", required=False)
parser.add_argument("--max_rounds", type=int, default=80, required=False)
parser.add_argument("--num_epoches", type=int, default=5, required=False)
parser.add_argument("--lr", type=float, default=1e-3, required=False)
parser.add_argument("--training_data_folder", type=str, default=root_path+"/Collaborative-Train/Online-Training-Data/Env-2", required=False)
parser.add_argument("--val_data_folder", type=str, default=root_path+"/Collaborative-Train/Online-Val-Data/Env-2", required=False)
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
from Utils.Functions.Train.TrainCNNSideNet import TrainCNNSideNet
from Utils.Functions.Eval.EvalCNNSideNet import EvalCNNSideNet
from Utils.Configurations.BasicConfig import BasicConfig
from sharpfed.functions import aggregate_model_updates

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
    critical_users = [i for i in range(12, 24)]
    # ---------------------------------------------------------------------------- #
    #                            Load Pretrained Models                            #
    # ---------------------------------------------------------------------------- #
    cnn_main_net = CNNMainNetInfer()
    batch_pilots_rg = np.zeros((16, BasicConfig()._num_ofdm_symbols, BasicConfig()._fft_size, 1), dtype=np.complex64)
    batch_y = np.zeros((16, BasicConfig()._num_ofdm_symbols, BasicConfig()._fft_size), dtype=np.complex64)
    batch_N0 = np.zeros(16, dtype=np.float32)
    main_net_out = cnn_main_net([batch_pilots_rg, batch_y, batch_N0])
    cnn_main_net.set_weights(load_pkl_file(run_args.pretrained_main_net_path))

    cnn_side_net = CNNSideNet()
    cnn_side_net([batch_pilots_rg, batch_y, batch_N0], main_net_out=main_net_out)
    cnn_side_net.set_weights(load_pkl_file(run_args.side_net_path))

    # ---------------------------------------------------------------------------- #
    #                          Initialize The Cache Folder                         #
    # ---------------------------------------------------------------------------- #
    if os.path.exists(run_args.cache_folder):
        shutil.rmtree(run_args.cache_folder)
        os.mkdir(run_args.cache_folder)
    else:
        os.mkdir(run_args.cache_folder)

    for user_id in critical_users:
        os.mkdir(Path(run_args.cache_folder).joinpath("%d" % user_id))
        initial_global_model_parameters = cnn_side_net.get_weights()
        save_pkl_file(initial_global_model_parameters, Path(run_args.cache_folder).joinpath("%d" % user_id, "global_model_parameters_0.pkl"))

    # ---------------------------------------------------------------------------- #
    #                              Federated Updating                              #
    # ---------------------------------------------------------------------------- #
    ber_rounds = np.zeros(run_args.max_rounds, dtype=np.float128)
    stopping_counter = 0

    for r in range(1, run_args.max_rounds+1):
        user_model_updates = dict()
        user_aggregation_weight = dict()
        user_layermask = dict()

        # ------------------------ Validation Info Collection ------------------------ #
        for user_id in critical_users:
            cnn_side_net.set_weights(load_pkl_file(Path(run_args.cache_folder).joinpath("%d" % user_id, "global_model_parameters_%d.pkl" % (r-1))))
            eval_data_pathes = get_file_pathes_in_folder(Path(run_args.val_data_folder).joinpath("User-%d" % user_id))
            obj = EvalCNNSideNet(config=BasicConfig(),
                                 main_net_infer=cnn_main_net,
                                 side_net=cnn_side_net)
            ber_rounds[r-1] += obj.eval(eval_data_pathes=eval_data_pathes)
        ber_rounds[r-1] = ber_rounds[r-1] / len(critical_users)
        save_pkl_file(ber_rounds, Path(run_args.cache_folder).joinpath("ber_rounds.pkl"))

        # Early stopping
        if r==1:
            pass
        else:
            if ber_rounds[r-1] < np.min(ber_rounds[:r-1]):
                stopping_counter = 0
            else:
                stopping_counter += 1
            if stopping_counter >= 5:
                break

        # -------------------------- Updating For Each User -------------------------- #
        for user_id in critical_users:
            user_aggregation_weight[user_id] = 1
            user_layermask[user_id] = [1 for _ in range(len(initial_global_model_parameters))]

            # Load the global model parameters of last round
            last_round_global_model_parameters = load_pkl_file(Path(run_args.cache_folder).joinpath("%d" % user_id, "global_model_parameters_%d.pkl" % (r-1)))
            cnn_side_net.set_weights(last_round_global_model_parameters)

            # Load training data of the current user
            train_data_pathes = get_file_pathes_in_folder(Path(run_args.training_data_folder).joinpath("User-%d" % user_id))

            # Training
            os.mkdir(Path(run_args.cache_folder).joinpath("%d" % user_id, "Updated-Parameters-Round%d" % r))
            obj = TrainCNNSideNet(config=BasicConfig(),
                                 main_net_infer=cnn_main_net,
                                 side_net=cnn_side_net,
                                 max_epoches=run_args.num_epoches,
                                 lr=run_args.lr,
                                 train_data_pathes=train_data_pathes,
                                 val_data_pathes=None,
                                 model_save_folder=Path(run_args.cache_folder).joinpath("%d" % user_id, "Updated-Parameters-Round%d" % r))
            updated_parameters = obj.train(save_mode='epoch')

            # Calculate the model updates
            model_updates = []
            for layer_idx in range(len(updated_parameters)):
                model_updates.append((updated_parameters[layer_idx]-last_round_global_model_parameters[layer_idx]).astype(np.float32))
            user_model_updates[user_id] = model_updates

        # ------------------------ Aggregate The Model Updates ----------------------- #
        aggregated_model_updates = aggregate_model_updates(client_model_updates=user_model_updates,
                                                           client_aggregation_weight=user_aggregation_weight,
                                                           client_layermask=user_layermask)
        last_round_global_model_parameters = load_pkl_file(Path(run_args.cache_folder).joinpath("%d" % critical_users[0], "global_model_parameters_%d.pkl" % (r-1)))
        updated_global_model_parameters = [last_round_global_model_parameters[layer_idx] + aggregated_model_updates[layer_idx]
                                           for layer_idx in range(len(last_round_global_model_parameters))]

        # ------------------ Distribute The Updated Model Parameters ----------------- #
        for user_id in critical_users:
            save_pkl_file(updated_global_model_parameters, Path(run_args.cache_folder).joinpath("%d" % user_id, "global_model_parameters_%d.pkl" % r))