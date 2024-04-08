import os
import sys
import argparse
import torch
import tqdm
    
#from ray import tune, air

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import utils
import datasets
import time_frequency as tf

def get_window_length_results(df, window_length, sr=8000):
    init_lambd = window_length / 6 * sr
    eps = 1e-5
    df_res = df[(df['config/init_lambd'] > (init_lambd - eps)) & (df['config/init_lambd'] < (init_lambd + eps))]

    return df_res

def produce_table_1(experiment_path, dataset_name):
    df = pd.read_csv(os.path.join(experiment_path, "{}.csv".format(dataset_name)))
    df_train = df[df['config/trainable'] == True]
    df_fixed = df[df['config/trainable'] == False]

    window_lengths = [0.010, 0.035, 0.300]

    print("Model & $l_{\lambda_{init}}$ & $l_{\lambda_{est}}$ & Method & Accuracy \\\\")
    print("\\hline \\hline")


    for window_length in window_lengths:
        df_train_win = get_window_length_results(df_train, window_length, sr=8000)
        df_fixed_win = get_window_length_results(df_fixed, window_length, sr=8000)

        mean_train_acc = df_train_win['test_accuracy'].mean() * 100
        std_train_acc = df_train_win['test_accuracy'].std() * 100

        mean_fixed_acc = df_fixed_win['test_accuracy'].mean() * 100
        std_fixed_acc = df_fixed_win['test_accuracy'].std() * 100

        min_lambd_est = df_train_win['best_lambd_est'].abs().min() * 6 / 8000
        max_lambd_est = df_train_win['best_lambd_est'].abs().max() * 6 / 8000

        row_format = "{} & {} ms & ({}, {}) ms & {} & ${:.1f} \pm {:.1f}$ \\\\"
        print(row_format.format(
            "LNet", int(window_length * 1000), int(min_lambd_est * 1000), int(max_lambd_est * 1000), 
            "DMEL", mean_train_acc, std_train_acc)
        )
        row_format = "{} & {} ms & {} ms & {} & ${:.1f} \pm {:.1f}$ \\\\"
        print(row_format.format(
            "LNet", int(window_length * 1000), int(window_length * 1000), 
            "baseline", mean_fixed_acc, std_fixed_acc)
        )
        print("\\hline")

def produce_table_2(experiment_path, dataset_name):
    df = pd.read_csv(os.path.join(experiment_path, "{}.csv".format(dataset_name)))
    #print(df)
    df_train = df[df['config/trainable'] == True]
    df_fixed = df[df['config/trainable'] == False]

    sigma_ref = 6.38
    #lambd_inits = [sigma_ref * 0.2, sigma_ref*0.6, sigma_ref, sigma_ref*1.8, sigma_ref*2.6]
    lambd_inits = [sigma_ref * 0.2, sigma_ref, sigma_ref*5.0]

    print("Model & $\lambda_{init}$ & $\lambda_{est}$ & Method & Accuracy \\\\")
    print("\\hline \\hline")


    for lambd_init in lambd_inits:
        df_train_win = df_train[df_train['config/init_lambd'] == lambd_init]
        df_fixed_win = df_fixed[df_fixed['config/init_lambd'] == lambd_init]

        mean_train_acc = df_train_win['test_accuracy'].mean() * 100
        std_train_acc = df_train_win['test_accuracy'].std() * 100

        mean_fixed_acc = df_fixed_win['test_accuracy'].mean() * 100
        std_fixed_acc = df_fixed_win['test_accuracy'].std() * 100

        min_lambd_est = df_train_win['best_lambd_est'].abs().min()
        max_lambd_est = df_train_win['best_lambd_est'].abs().max()

        mean_lambd_est = df_train_win['best_lambd_est'].abs().mean()
        std_lambd_est = df_train_win['best_lambd_est'].abs().std()

        row_format = "{} & {:.1f} & ({:.1f}, {:.1f}) & {} & ${:.1f} \pm {:.1f}$ \\\\"
        print(row_format.format(
            "LinearNet", lambd_init, min_lambd_est, max_lambd_est, 
            "DSPEC", mean_train_acc, std_train_acc)
        )
        row_format = "{} & {:.1f} & {:.1f} & {} & ${:.1f} \pm {:.1f}$ \\\\"
        print(row_format.format(
            "LinearNet", lambd_init, lambd_init, 
            "baseline", mean_fixed_acc, std_fixed_acc)
        )
        print("\\hline")

def produce_result_table(experiment_path, dataset_name):
    if dataset_name == 'audio_mnist':
        model_names = ['mel_conv_net', 'mel_linear_net']
    if dataset_name == 'esc50':
        model_names = ['panns_cnn6']

    print("############################################")
    print("Dataset : {}, and models: {}".format(dataset_name, model_names))
    print("############################################")

    # load test predictions
    df = pd.read_csv(os.path.join(experiment_path, "{}.csv".format(dataset_name)))
    predictionss = np.load(os.path.join(experiment_path, "{}_predictionss.npy".format(dataset_name)))
    labelss = np.load(os.path.join(experiment_path, "{}_labelss.npy".format(dataset_name)))

    column_width = 4
    figure_height = 3

    df = df[(df['config/dataset_name'] == dataset_name)]
    df = df[(df['config/init_lambd'] == 8000*0.025 / 6)]

    print("Trainable & True & False \\\\")
    for idx_column, model_name in enumerate(model_names):
        df_model = df[(df['config/model_name'] == model_name)]
        df_train = df_model[(df_model['config/trainable'] == True)]
        df_fixed = df_model[(df_model['config/trainable'] == False)]
        model_title = get_model_title(model_name)

        trainable_mean_acc = df_train['test_accuracy'].mean()
        trainable_std_acc = df_train['test_accuracy'].std()

        fixed_mean_acc = df_fixed['test_accuracy'].mean()
        fixed_std_acc = df_fixed['test_accuracy'].std()

        print("{} & ${:.2f} \\pm {:.2f}$ & ${:.2f} \\pm {:.2f}$ \\\\".format(
            model_title,
            trainable_mean_acc, trainable_std_acc,
            fixed_mean_acc, fixed_std_acc
        ))
        
       

def main():
    parser = argparse.ArgumentParser(description='Produce plots.')
    parser.add_argument('--ray_results_dir', help='The name of the ray results directory.', required=True, type=str)
    #parser.add_argument('--experiment_path', help='The name of the experiment directory.', required=True, type=str)
    #parser.add_argument('--dataset_name', help='The dataset name.', required=True, type=str)
    args = parser.parse_args()

    #experiment_path = os.path.join(args.experiment_path)
    print("ESC50")
    produce_table_1(os.path.join(args.ray_results_dir, 'esc50'), 'esc50')
    print("")
    
    print("A-MNIST")
    produce_table_1(os.path.join(args.ray_results_dir, 'audio_mnist'), 'audio_mnist')
    print("")

    print("time-frequency")
    produce_table_2(os.path.join(args.ray_results_dir, 'time_frequency'), 'time_frequency')
    print("")


def get_model_title(model_name):
    if model_name == 'conv_net':
        model_title = 'ConvNet'
    elif model_name == 'linear_net':
        model_title = 'LinearNet'
    elif model_name == 'mel_linear_net':
        model_title = 'MelLinearNet'
    elif model_name == 'mel_conv_net':
        model_title = 'MelConvNet'
    elif model_name == 'panns_cnn6':
        model_title = 'PANNs CNN6'
    else:
        raise ValueError("model_name: {} is not defined.".format(model_name))
        
    return model_title

def get_data_title(dataset_name):
    if dataset_name == 'time_frequency':
        dataset_title = "Gaussian-pulse dataset"
    elif dataset_name == 'audio_mnist':
        dataset_title = "Audio MNIST dataset"
    else:
        raise ValueError("dataset_name: {} is not defined.".format(dataset_name))

    return dataset_title

def get_title(dataset_name, model_name):

    dataset_title = get_data_title(dataset_name)
    model_title = get_model_title(model_name)
        
    return "{} with {}".format(dataset_title, model_title)

if __name__ == '__main__':
    main()
