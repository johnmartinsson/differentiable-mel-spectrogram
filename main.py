import torch

import datasets
import models
import train
import utils

def main():

    config = {
        # training
        'lr' : 1e-3,
        'batch_size' : 64,
        'epochs' : 500,
        'trainable' : True,
        'device' : 'cuda:0',
        
        # dataset
        'n_points' : 128,
        'noise_std' : 0.5,
        'sigma_scale' : 3,
        'n_samples' : 500,
    }

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

    # initialize window size to roughly 25 ms
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
                'lr' : 1,
            }]
        else:
            parameters += [{
                'params' : [param],
                'lr' : 1e-3,
            }]

    optimizer = torch.optim.SGD(parameters)
    loss_fn   = torch.nn.CrossEntropyLoss()

    net, history = train.train_model(
        net=net,
        optimizer=optimizer,
        loss_fn=loss_fn,
        trainloader=trainloader,
        validloader=validloader,
        patience=5,
        max_epochs=100,
        verbose=0,
        device=device,
    )

    print(history)

if __name__ == '__main__':
    main()
