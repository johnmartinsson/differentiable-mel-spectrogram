import os
import sys
import argparse
import torch
import tqdm
    
from ray import tune, air

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import utils
import datasets
import time_frequency as tf

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
    parser.add_argument('--experiment_path', help='The name of the experiment directory.', required=True, type=str)
    parser.add_argument('--dataset_name', help='The dataset name.', required=True, type=str)
    args = parser.parse_args()

    experiment_path = os.path.join(args.experiment_path)
    produce_result_table(experiment_path, args.dataset_name)

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
