import torch

from functools import partial

from ray import tune
from ray import air
from ray.tune import CLIReporter

import argparse

import datasets
import models
import train
import utils
import search_spaces

def run_experiment(config):
    # data
    if config['dataset_name'] == 'esc50':
        n_classes = 50

        dataset = datasets.ESC50Dataset(
            source_dir    = '/home/john/gits/differentiable-time-frequency-transforms/data/esc50',
            resample_rate = config['resample_rate']
        )

    elif config['dataset_name'] == 'audio_mnist':
        n_classes = 10

        dataset = datasets.AudioMNISTDataset(
            source_dir    = '/home/john/gits/differentiable-time-frequency-transforms/data/audio-mnist',
        )

    else:
        sigma_ref = torch.tensor(config['sigma_ref'])

        # time-frequency redundancy settings
        size       = (config['n_points']+1, config['n_points']+1)

        # random offset 1/5 of tf-image in each direction
        if config['center_offset']:
            f_center_max_offset = 0.1
            t_center_max_offset = config['n_points']/5
        else:
            f_center_max_offset = 0.0
            t_center_max_offset = 0.0


        if config['dataset_name'] == 'time':
            n_classes = 2
            dataset = datasets.GaussPulseDatasetTime(
                sigma     = config['sigma_ref'],
                n_points  = config['n_points'],
                noise_std = torch.tensor(config['noise_std']),
                n_samples = config['n_samples'], 
                f_center_max_offset = f_center_max_offset,
                t_center_max_offset = t_center_max_offset,
            )
        elif config['dataset_name'] == 'frequency':
            n_classes = 2
            dataset = datasets.GaussPulseDatasetFrequency(
                sigma     = config['sigma_ref'],
                n_points  = config['n_points'],
                noise_std = torch.tensor(config['noise_std']),
                n_samples = config['n_samples'], 
                f_center_max_offset = f_center_max_offset,
                t_center_max_offset = t_center_max_offset,
            )
        elif config['dataset_name'] == 'time_frequency':
            n_classes = 3

            dataset = datasets.GaussPulseDatasetTimeFrequency(
                sigma     = config['sigma_ref'],
                n_points  = config['n_points'],
                noise_std = torch.tensor(config['noise_std']),
                n_samples = config['n_samples'], 
                f_center_max_offset = f_center_max_offset,
                t_center_max_offset = t_center_max_offset,
            )
        else:
            raise ValueError("dataset not defined: ", config['dataset_name'])

    trainset, validset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    # model
    device = config['device']

    if config['model_name'] == 'linear_net':
        net = models.LinearNet(
            n_classes  = n_classes,
            init_lambd = config['init_lambd'],
            device     = device,
            size       = size,
            hop_length = config['hop_length'],
            optimized  = config['optimized'],
        )
    elif config['model_name'] == 'mlp_net':
        net = models.MlpNet( # TODO: have I ran all results with linear model instead of MLP?
            n_classes  = n_classes,
            init_lambd = config['init_lambd'],
            device     = device,
            size       = size,
            hop_length = config['hop_length'],
            optimized  = config['optimized'],
        )
    elif config['model_name'] == 'conv_net':
        net = models.ConvNet(
            n_classes  = n_classes,
            init_lambd = config['init_lambd'],
            device     = device,
            size       = size,
            hop_length = config['hop_length'],
            optimized  = config['optimized'],
        )
    elif config['model_name'] == 'mel_mlp_net':
        net = models.MelMlpNet(
            n_classes   = n_classes,
            init_lambd  = torch.tensor(config['init_lambd']),
            device      = device,
            n_mels      = config['n_mels'],
            sample_rate = config['resample_rate'],
            n_points    = config['n_points'],
            hop_length  = config['hop_length'], #int(config['resample_rate'] * config['hop_time_s']),
            optimized   = config['optimized'], #True,
            energy_normalize = config['energy_normalize'],
        )
    elif config['model_name'] == 'mel_conv_net':
        net = models.MelConvNet(
            n_classes   = n_classes,
            init_lambd  = torch.tensor(config['init_lambd']),
            device      = device,
            n_mels      = config['n_mels'],
            sample_rate = config['resample_rate'],
            n_points    = config['n_points'],
            hop_length  = config['hop_length'], #int(config['resample_rate'] * config['hop_time_s']),
            optimized   = config['optimized'], #True,
            energy_normalize = config['energy_normalize'],
        )

    else:
        raise ValueError("model name not found: ", config['model_name'])

    net.to(device)

    net.spectrogram_layer.requires_grad_(config['trainable'])

    parameters = []
    for idx, (name, param) in enumerate(net.named_parameters()):

        if name == "spectrogram_layer.lambd":
            parameters += [{
                'params' : [param],
                'lr' : config['lr_tf'],
            }]
        else:
            parameters += [{
                'params' : [param],
                'lr' : config['lr_model'],
            }]

    if config['optimizer_name'] == 'sgd':
        optimizer = torch.optim.SGD(parameters)
    elif config['optimizer_name'] == 'adam':
        optimizer = torch.optim.Adam(parameters)
    else:
        raise ValueError("optimizer not found: ", config['optimizer_name'])

    loss_fn = torch.nn.CrossEntropyLoss()

    net, history = train.train_model(
        net=net,
        optimizer=optimizer,
        loss_fn=loss_fn,
        trainloader=trainloader,
        validloader=validloader,
        patience=config['patience'],
        max_epochs=config['max_epochs'],
        verbose=0,
        device=device,
    )

def main():

    parser = argparse.ArgumentParser(description='Hyperparameter search.')
    parser.add_argument('-s','--num_samples', help='The number of hyperparameter samples.', required=True, type=int)
    parser.add_argument('-e','--max_epochs', help='The maximum number of epochs.', required=True, type=int)
    parser.add_argument('-n','--name', help='The name of the hyperparamter search experiment..', required=True, type=str)
    args = parser.parse_args()

    # hyperparamter search space
    #search_space = search_spaces.development_space(args.max_epochs)
    #search_space = search_spaces.development_space_esc50(args.max_epochs)
    search_space = search_spaces.development_space_audio_mnist(args.max_epochs)

    # results terminal reporter
    reporter = CLIReporter(
        metric_columns=[
            "loss",
            "accuracy",
            "lambd_est",
            "training_iteration",
            "best_valid_acc",
        ],
        parameter_columns = [
            'init_lambd',
            'trainable',
            'model_name',
        ],
        max_column_length = 10
    )

    trainable_with_resources = tune.with_resources(run_experiment, {"cpu" : 2.0, "gpu": 0.25})

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space = search_space,
        run_config  = air.RunConfig(
            progress_reporter = reporter,
            name              = args.name,
            local_dir         = '/mnt/storage_1/john/ray_results/',
        ),
	tune_config = tune.TuneConfig(
	    num_samples = args.num_samples,
	),
    )

    result = tuner.fit()

if __name__ == '__main__':
    main()
