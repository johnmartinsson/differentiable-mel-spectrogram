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

def run_experiment(config):
    # data
    sigma_ref = torch.tensor(3.0)
    dataset = datasets.GaussPulseDatasetTime(
        sigma     = sigma_ref,
        n_points  = config['n_points'],
        noise_std = torch.tensor(config['noise_std']),
        n_samples = config['n_samples'], 
    )

    trainset, validset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    validloader = torch.utils.data.DataLoader(validset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    init_sigma = sigma_ref * config['sigma_scale']
    hop_length = 1

    # model
    device = config['device']

    net = models.LinearNet(
        n_classes=2,
        init_sigma=init_sigma,
        device=device,
        size=(config['n_points']+1, config['n_points']+1),
        hop_length=hop_length,
        optimized=False,
    )
    net.to(device)

    net.spectrogram_layer.requires_grad_(config['trainable'])

    parameters = []
    for idx, (name, param) in enumerate(net.named_parameters()):

        if name == "spectrogram_layer.sigma":
            parameters += [{
                'params' : [param],
                'lr' : config['lr_tf'],
            }]
        else:
            parameters += [{
                'params' : [param],
                'lr' : config['lr_model'],
            }]

    optimizer = torch.optim.SGD(parameters)
    loss_fn   = torch.nn.CrossEntropyLoss()

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
    search_space = {
        # training
        'lr_model' : 1e-3,
        'lr_tf' : 1, #tune.choice([1e-3, 1e-2, 1e-1, 1, 10]),
        'batch_size' : 64,
        'epochs' : 500,
        'trainable' : tune.choice([False, True]),
	'max_epochs' : args.max_epochs,
	'patience' : 5,
        'device' : 'cuda:0',
        
        # dataset
        'n_points' : 128,
        'noise_std' : tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'sigma_scale' : tune.choice([0.5, 1.0, 3]),
        'n_samples' : tune.choice([100, 500, 1000]),
    }

    # results terminal reporter
    reporter = CLIReporter(
        metric_columns=[
            "loss",
            "accuracy",
            "sigma_est",
            "training_iteration"
        ],
        parameter_columns = [
            'lr_tf',
            'n_samples',
            'sigma_scale',
            'noise_std',
            'trainable'
        ],
        max_column_length = 10
    )

    trainable_with_resources = tune.with_resources(run_experiment, {"cpu" : 1, "gpu": 0.1})

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
