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

def produce_plots(experiment_path, split='valid'):
    if 'audio_mnist' in experiment_path:
        dataset_name = 'audio_mnist'
        model_names = ['mel_linear_net', 'mel_conv_net']
    elif 'time_frequency' in experiment_path:
        dataset_name = 'time_frequency'
        model_names = ['linear_net', 'conv_net']

    tuner = tune.Tuner.restore(path=experiment_path)
    result = tuner.fit()
    df = result.get_dataframe()

    if not os.path.exists('./results/figures'):
        os.makedirs('./results/figures')

    if split == 'test':
        # make test predictions if they do not exist
        if not os.path.exists("results/{}.csv".format(dataset_name)):
            predict_test(df, dataset_name)

        # load test predictions
        df = pd.read_csv("results/{}.csv".format(dataset_name))
        predictionss = np.load("results/{}_predictionss.npy".format(dataset_name))
        labelss = np.load("results/{}_labelss.npy".format(dataset_name))

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
    plt.savefig('results/figures/{}_{}.pdf'.format(split, dataset_name), bbox_inches='tight')

def predict_test(df, dataset_name):
    df['test_accuracy'] = 0
    predictionss = []
    labelss = []

    print("making test predictions (takes a couple of minutes on GPU) ...")
    for row in tqdm.tqdm(df.iterrows()):
        idx = row[0]
        #print(row)
        #print("Model = ", row[1]['config/model_name'])
        #print("reported best valid acc: ", row[1]['best_valid_acc'])
        labels, predictions = utils.get_predictions_by_row(row, split='test', device='cuda:1')
        test_acc = np.mean(labels == predictions)
        df.at[idx, 'test_accuracy'] = test_acc
        #print("test acc: ", test_acc)
        
        predictionss.append(predictions)
        labelss.append(labels)

    df.to_csv("results/{}.csv".format(dataset_name))
    predictionss = np.array(predictionss)
    labelss = np.array(labelss)
    np.save("results/{}_predictionss.npy".format(dataset_name), predictionss)
    np.save("results/{}_labelss.npy".format(dataset_name), labelss)

def main():
    parser = argparse.ArgumentParser(description='Produce plots.')
    parser.add_argument('-n','--name', help='The name of the hyperparamter search experiment.', required=True, type=str)
    parser.add_argument('-d','--ray_root_dir', help='The name of the root directory to save the ray search results.', required=True, type=str)
    parser.add_argument('-s','--split', help='The name of the split [train, valid].', required=True, type=str)
    args = parser.parse_args()

    experiment_path = os.path.join(args.ray_root_dir, args.name)

    produce_plots(experiment_path, args.split)

def get_model_title(model_name):
    if model_name == 'conv_net':
        model_title = 'ConvNet'
    elif model_name == 'linear_net':
        model_title = 'LinearNet'
    elif model_name == 'mel_linear_net':
        model_title = 'MelLinearNet'
    elif model_name == 'mel_conv_net':
        model_title = 'MelConvNet'
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
