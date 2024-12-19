# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

root_path = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_main_net_path", type=str, default=root_path+"/Offline-Pretrain/Pretrained-Models/CNNMainNet/updated_parameters_29.pkl", required=False)
parser.add_argument("--side_net_path", type=str, default=root_path+"/Collaborative-Train/Centralized/t1-t2/WF/Adapted-Models/updated_parameters_7.pkl", required=False)
parser.add_argument("--eval_data_folder", type=str, default=root_path+"/BER-Performance-Evaluation/Eval-Data/TDLB-150-20", required=False)
parser.add_argument("--min_ebNo", type=float, default=4, required=False)
parser.add_argument("--max_ebNo", type=float, default=12, required=False)
parser.add_argument("--num_ebNo_points", type=int, default=9, required=False)
parser.add_argument("--res_save_path", type=str, default=root_path+"/BER-Performance-Evaluation/Centralized-Eval/t1-t2/WF/eval_res_on_E2.pkl", required=False)
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
from Utils.Functions.Common.io import get_file_pathes_in_folder, load_pkl_file, save_pkl_file
from Utils.Models.CNN.CNNMainNet import CNNMainNetInfer
from Utils.Models.CNN.CNNSideNet import CNNSideNet
from Utils.Functions.Eval.EvalCNNSideNet import EvalCNNSideNet
from Utils.Configurations.BasicConfig import BasicConfig

if __name__=='__main__':
    cnn_main_net = CNNMainNetInfer()
    batch_pilots_rg = np.zeros((16, BasicConfig()._num_ofdm_symbols, BasicConfig()._fft_size, 1), dtype=np.complex64)
    batch_y = np.zeros((16, BasicConfig()._num_ofdm_symbols, BasicConfig()._fft_size), dtype=np.complex64)
    batch_N0 = np.zeros(16, dtype=np.float32)
    main_net_out = cnn_main_net([batch_pilots_rg, batch_y, batch_N0])
    cnn_main_net.set_weights(load_pkl_file(run_args.pretrained_main_net_path))

    cnn_side_net = CNNSideNet()
    cnn_side_net([batch_pilots_rg, batch_y, batch_N0], main_net_out=main_net_out)
    cnn_side_net.set_weights(load_pkl_file(run_args.side_net_path))

    obj = EvalCNNSideNet(config=BasicConfig(),
                         main_net_infer=cnn_main_net,
                         side_net=cnn_side_net)
    ber = dict()
    for dB in np.linspace(run_args.min_ebNo, run_args.max_ebNo, run_args.num_ebNo_points):
        eval_data_pathes = get_file_pathes_in_folder(Path(run_args.eval_data_folder).joinpath('%.2f' % dB))
        ber[dB] = obj.eval(eval_data_pathes)
    save_pkl_file(ber, run_args.res_save_path)