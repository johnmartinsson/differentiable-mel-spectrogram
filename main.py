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

def run_experiment(config, data_dir):
    # load dataset
    trainset, validset, _ = utils.get_dataset_by_config(config, data_dir)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    # load model
    net = utils.get_model_by_config(config)
    net.to(config['device'])

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
        device=config['device'],
    )

def main():

    parser = argparse.ArgumentParser(description='Hyperparameter search.')
    parser.add_argument('--num_samples', help='The number of hyperparameter samples.', required=True, type=int)
    parser.add_argument('--max_epochs', help='The maximum number of epochs.', required=True, type=int)
    parser.add_argument('--name', help='The name of the hyperparamter search experiment.', required=True, type=str)
    parser.add_argument('--ray_root_dir', help='The name of the directory to save the ray search results.', required=True, type=str)
    parser.add_argument('--data_dir', help='The absolute path to the audio-mnist directory.', required=True, type=str)
    args = parser.parse_args()

    # hyperparamter search space
    if "audio_mnist" in args.name:
        search_space = search_spaces.audio_mnist(args.max_epochs)
    elif "time_frequency" in args.name:
        search_space = search_spaces.time_frequency(args.max_epochs)
    else:
        raise ValueError("search space not found ...")


    # results terminal reporter
    reporter = CLIReporter(
        metric_columns=[
            "loss",
            "lambd_est",
            "training_iteration",
            "best_lambd_est",
            "best_valid_acc",
        ],
        parameter_columns = [
            'init_lambd',
            'trainable',
            'model_name',
            'center_offset',
        ],
        max_column_length = 10
    )

    run_experiment_fn = partial(run_experiment, data_dir=args.data_dir)

    trainable_with_resources = tune.with_resources(run_experiment_fn, {"cpu" : 2.0, "gpu": 0.33})

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space = search_space,
        run_config  = air.RunConfig(
            verbose=1,
            progress_reporter = reporter,
            name              = args.name,
            local_dir         = args.ray_root_dir, 
        ),
	tune_config = tune.TuneConfig(
	    num_samples = args.num_samples,
	),
    )

    result = tuner.fit()

if __name__ == '__main__':
    main()
