# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

root_path = str(Path(__file__).resolve().parent.parent.parent)
parser = argparse.ArgumentParser()
parser.add_argument("--eval_data_folder", type=str, default=root_path+"/BER-Performance-Evaluation/Eval-Data/TDLB-150-20", required=False)
parser.add_argument("--min_ebNo", type=float, default=4, required=False)
parser.add_argument("--max_ebNo", type=float, default=12, required=False)
parser.add_argument("--num_ebNo_points", type=int, default=9, required=False)
parser.add_argument("--res_save_path", type=str, default=root_path+"/BER-Performance-Evaluation/Conventional-Eval/eval_res_perf_csi_on_E2.pkl", required=False)
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
from Utils.Functions.Common.io import get_file_pathes_in_folder, save_pkl_file
from Utils.Functions.Eval.EvalPerfectEst import EvalPerfectEst
from Utils.Configurations.BasicConfig import BasicConfig

if __name__=='__main__':
    tf.random.set_seed(45)
    np.random.seed(45)

    obj = EvalPerfectEst(config=BasicConfig())
    ber = dict()
    for dB in np.linspace(run_args.min_ebNo, run_args.max_ebNo, run_args.num_ebNo_points):
        eval_data_pathes = get_file_pathes_in_folder(Path(run_args.eval_data_folder).joinpath('%.2f' % dB))
        ber[dB] = obj.eval(eval_data_pathes)
    save_pkl_file(ber, run_args.res_save_path)