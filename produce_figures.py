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

def produce_data_example_plot():
    # data
    sigma_ref = torch.tensor(6.38)
    dataset = datasets.GaussPulseDatasetTimeFrequency(
        sigma     = sigma_ref,
        n_points  = 128, 
        noise_std = torch.tensor(.0),
        n_samples = 500,
        f_center_max_offset=0,
        t_center_max_offset=0,
        demo=True, # disables a lot of variability, more pedagogical figure
    )

    plt.rcParams['text.usetex'] = True

    short_scale = 0.2
    long_scale = 5

    idx = 0
    n_classes = 3
    count = 0

    lambda_param = sigma_ref

    fig, ax = plt.subplots(3, 3, figsize=(8,3*2.7))

    while True:
    #for idx in range(20):
        x, y = dataset[idx]
            
        if count % n_classes == y:
            count += 1
            t1, f1, t2, f2 = dataset.locs[idx]
            
            s, w = tf.differentiable_spectrogram(x-torch.mean(x), lambd=lambda_param, return_window=True)
            utils.plot_spectrogram(s.detach().numpy(), ax[count-1, 0], decorate_axes=False)

            s, w = tf.differentiable_spectrogram(x-torch.mean(x), lambd=lambda_param*short_scale, return_window=True)
            utils.plot_spectrogram(s.detach().numpy(), ax[count-1, 1], decorate_axes=False)
            
            s, w = tf.differentiable_spectrogram(x-torch.mean(x), lambd=lambda_param*long_scale, return_window=True)
            utils.plot_spectrogram(s.detach().numpy(), ax[count-1, 2], decorate_axes=False)

            idx += 1
        else:
            idx += 1
        
        if count > 2:
            break
                 
    scales = [1.0, short_scale, long_scale]
    for i in range(3):
        ax[i, 0].set_ylabel('normalized frequency')
        ax[2, i].set_xlabel('time')
        ax[0, i].set_title(r'$\lambda = {0:.1f}$'.format(lambda_param * scales[i]))
        
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, 'data_example.pdf'), bbox_inches='tight')
        

def produce_accuracy_plot(experiment_path, data_dir, split='valid'):
    if 'audio_mnist' in experiment_path:
        dataset_name = 'audio_mnist'
        model_names = ['mel_linear_net', 'mel_conv_net']
    elif 'esc50' in experiment_path:
        dataset_name = 'esc50'
        model_names = ['panns_cnn6']
    elif 'time_frequency' in experiment_path:
        dataset_name = 'time_frequency'
        model_names = ['linear_net', 'conv_net']

    print("############################################")
    print("Dataset : {}, and models: {}".format(dataset_name, model_names))
    print("############################################")

    tuner = tune.Tuner.restore(path=experiment_path)
    print(experiment_path)
    result = tuner.fit()
    df = result.get_dataframe()


    if split == 'test':
        # make test predictions if they do not exist
        #if not os.path.exists(os.path.join(experiment_path, "{}.csv".format(dataset_name))):
        #    predict_test(df, dataset_name, data_dir, experiment_path)

        # load test predictions
        df = pd.read_csv(os.path.join(experiment_path, "{}.csv".format(dataset_name)))
        predictionss = np.load(os.path.join(experiment_path, "{}_predictionss.npy".format(dataset_name)))
        labelss = np.load(os.path.join(experiment_path, "{}_labelss.npy".format(dataset_name)))

    column_width = 4
    figure_height = 3

    #####################################################################
    # Accuracy plot
    #####################################################################
    df = df[(df['config/dataset_name'] == dataset_name)]

    fig, ax = plt.subplots(2, 2, figsize=(column_width*2, figure_height*2))

    for idx_column, model_name in enumerate(model_names):
        df_model = df[(df['config/model_name'] == model_name)]
        
        model_title = get_model_title(model_name)
        
        ax[0, idx_column].set_title(model_title)

        if split == 'valid':
            y_str = 'best_valid_acc'
            y_title = 'Validation accuracy'
        elif split == 'test':
            y_str = 'test_accuracy'
            y_title = 'Test accuracy'
        else:
            raise ValueError('split not found: ', split)
        
        sns.lineplot(data=df_model, x="config/init_lambd", 
                     y=y_str, marker='o', 
                     hue='config/trainable', ax=ax[0, idx_column])

        ax[0, idx_column].legend(loc='lower center', title='Trainable')

        g = sns.lineplot(data=df_model, x="config/init_lambd",
                         y='lambd_est', hue='config/trainable', 
                         marker="o", ax=ax[1, idx_column])

        ax[1, idx_column].legend(loc='upper left', title='Trainable')

    ax[0, 0].set_ylabel(y_title)
    ax[0, 0].set_xlabel("")
    ax[0, 1].set_ylabel("")
    ax[0, 1].set_xlabel("")
    ax[1, 0].set_ylabel(r'$\lambda_{est}$')
    ax[1, 0].set_xlabel(r'$\lambda_{init}$')
    ax[1, 1].set_ylabel("")
    ax[1, 1].set_xlabel(r'$\lambda_{init}$')

    if dataset_name == 'audio_mnist':
        ax[0, 0].set_ylim([0.75, 0.96])
        ax[0, 1].set_ylim([0.75, 0.96])
        
    if dataset_name == 'time_frequency':
        ax[0, 0].set_ylim([0.95, 1])
        ax[0, 1].set_ylim([0.95, 1])

    plt.tight_layout()
    fig_path = os.path.join(experiment_path, '{}_{}.pdf'.format(split, dataset_name))
    print("saving figure ... ", fig_path)
    plt.savefig(fig_path, bbox_inches='tight')

def produce_baseline_plot(experiment_path, dataset_name, model_names, data_dir, split='valid'):
    print("############################################")
    print("Dataset : {}, and models: {}".format(dataset_name, model_names))
    print("############################################")

    tuner = tune.Tuner.restore(path=experiment_path)
    print(experiment_path)
    result = tuner.fit()
    df = result.get_dataframe()

    if split == 'test':
        # make test predictions if they do not exist
        if not os.path.exists(os.path.join(experiment_path, "{}.csv".format(dataset_name))):
            predict_test(df, dataset_name, data_dir, experiment_path)

        # load test predictions
        df = pd.read_csv(os.path.join(experiment_path, "{}.csv".format(dataset_name)))
        predictionss = np.load(os.path.join(experiment_path, "{}_predictionss.npy".format(dataset_name)))
        labelss = np.load(os.path.join(experiment_path, "{}_labelss.npy".format(dataset_name)))

    column_width = 4
    figure_height = 3

    #####################################################################
    # Accuracy plot
    #####################################################################
    df = df[(df['config/dataset_name'] == dataset_name)]

    fig, ax = plt.subplots(1, 1, figsize=(column_width*1, figure_height*1))

    for idx_column, model_name in enumerate(model_names):
        df_model = df[(df['config/model_name'] == model_name)]
        
        model_title = get_model_title(model_name)
        
        #ax[0, idx_column].set_title(model_title)

        if split == 'valid':
            y_str = 'best_valid_acc'
            y_title = 'Validation accuracy'
        elif split == 'test':
            y_str = 'test_accuracy'
            y_title = 'Test accuracy'
        else:
            raise ValueError('split not found: ', split)
        
        sns.lineplot(data=df_model, x="config/init_lambd", 
                     y=y_str, marker='o', 
                     hue='config/trainable', ax=ax)

        #ax[0, idx_column].legend(loc='lower center', title='Trainable')

        #g = sns.lineplot(data=df_model, x="config/init_lambd",
        #                 y='lambd_est', hue='config/trainable', 
        #                 marker="o", ax=ax[1, idx_column])

        #ax[1, idx_column].legend(loc='upper left', title='Trainable')

    ax.set_ylabel(y_title)
    ax.set_xlabel(r'$\lambda_{init}$')

    if dataset_name == 'audio_mnist':
        ax.set_ylim([0.75, 0.96])
        ax.set_ylim([0.75, 0.96])
        
    if dataset_name == 'time_frequency':
        ax.set_ylim([0.95, 1])
        ax.set_ylim([0.95, 1])

    if dataset_name == 'esc50':
        ax.set_ylim([0.65, 0.9])
        ax.set_ylim([0.65, 0.9])

    plt.tight_layout()
    fig_path = os.path.join(experiment_path, '{}_{}.pdf'.format(split, dataset_name))
    print("saving figure ... ", fig_path)
    plt.savefig(fig_path, bbox_inches='tight')



def main():
    parser = argparse.ArgumentParser(description='Produce plots.')
    parser.add_argument('--experiment_path', help='The name of the experiment directory.', required=True, type=str)
    parser.add_argument('--data_dir', help='The absolute path to the audio-mnist data directory.', required=True, type=str)
    #parser.add_argument('--name', help='The name of the experiment', required=True, type=str)
    parser.add_argument('--split', help='The name of the split [train, valid].', required=True, type=str)
    args = parser.parse_args()

    # produce figure 1
    #produce_data_example_plot()

    # produce figure 2
    #experiment_path = os.path.join(args.ray_root_dir, 'time_frequency')
    #produce_accuracy_plot(experiment_path, data_dir=args.data_dir, split=args.split)

    # produce figure 3
    #experiment_path = os.path.join(args.ray_root_dir, 'audio_mnist')
    #produce_accuracy_plot(experiment_path, data_dir=args.data_dir, split=args.split)

    #produce_accuracy_plot(experiment_path, data_dir=args.data_dir, split=args.split)

    experiment_path = os.path.join(args.experiment_path)
    produce_baseline_plot(experiment_path, dataset_name='esc50', model_names=['panns_cnn6'], data_dir=args.data_dir, split=args.split)

    #print("")
    #print("the plots can now be be found in ./results/figures")


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
