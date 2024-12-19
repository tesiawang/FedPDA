# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

def log_print(text, color, end = '\n'):
    if color == 'r':
        print(colored(text, 'red'), end = end)
    elif color == 'g':
        print(colored(text, 'green'), end = end)
    elif color == 'b':
        print(colored(text, 'blue'), end = end)
    elif color == 'y':
        print(colored(text, 'yellow'), end = end)
    elif color == 'c':
        print(colored(text, 'cyan'), end = end)
    elif color == 'm':
        print(colored(text, 'magenta'), end = end)
    else:
        print(text, end = end)

def get_file_pathes_in_folder(directory) -> list:
    all_file_pathes = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            all_file_pathes.append(os.path.join(dirpath, filename))
    return all_file_pathes

def get_file_names_in_folder(directory) -> list:
    all_file_names = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            all_file_names.append(filename)
    return all_file_names

def load_pkl_file(path):
    file = open(path, "rb")
    file_to_load = pickle.load(file)
    file.close()
    return file_to_load

def save_pkl_file(file_to_save, path):
    file = open(path, "wb")
    pickle.dump(file_to_save, file)
    file.close()

def load_batch_data(path):
    f = open(path, "rb")
    batch_data = pickle.load(f)
    f.close()
    ebno_db = batch_data["ebno_db"]
    batch_b = batch_data["batch_b"]
    batch_tx_codeword_bits = batch_data["batch_tx_codeword_bits"]
    batch_x_rg = batch_data["batch_x_rg"]
    batch_pilots_rg = batch_data["batch_pilots_rg"]
    batch_h_freq = batch_data["batch_h_freq"]
    batch_y = batch_data["batch_y"]
    batch_N0 = batch_data["batch_N0"]
    batch_h_ls_est = batch_data["batch_h_ls_est"]
    return ebno_db, batch_b, batch_tx_codeword_bits, batch_x_rg, batch_pilots_rg, batch_h_freq, batch_y, batch_N0, batch_h_ls_est

def plot_ber(ebno_dbs, ber_list, label_list, save_path):
    plt.figure(figsize=(10,6))
    marker_list = ['o--', 'x--', 'v--', '*--', 'd--', 's--']
    for i in range(len(ber_list)):
        plt.semilogy(ebno_dbs, ber_list[i], marker_list[i], label=label_list[i])
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BER")
    plt.grid(which="both")
    plt.ylim((1e-8, 1.0))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()