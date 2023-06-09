import torch
import numpy as np
import os
import tqdm

import models
import datasets

def get_config_by_row(row):
    config = {}
    r = row[1]
    for k in r.keys():
        if 'config' in k:
            config[k.split('/')[1]] = r[k]
    return config

def get_dataset_by_config(config, data_dir):
    if config['dataset_name'] == 'audio_mnist':
        dataset = datasets.AudioMNISTDataset(
            source_dir = data_dir,
        )
    elif config['dataset_name'] == 'esc50':
        dataset = datasets.ESC50Dataset(
            source_dir = data_dir,
            resample_rate = config['resample_rate'],
        )
    else:
        # random offset 1/5 of tf-image in each direction
        if config['center_offset']:
            f_center_max_offset = 0.1
            t_center_max_offset = config['n_points']/5
        else:
            f_center_max_offset = 0.0
            t_center_max_offset = 0.0


        if config['dataset_name'] == 'time':
            dataset = datasets.GaussPulseDatasetTime(
                sigma     = torch.tensor(config['sigma_ref']),
                n_points  = config['n_points'],
                noise_std = torch.tensor(config['noise_std']),
                n_samples = config['n_samples'], 
                f_center_max_offset = f_center_max_offset,
                t_center_max_offset = t_center_max_offset,
            )
        elif config['dataset_name'] == 'frequency':
            dataset = datasets.GaussPulseDatasetFrequency(
                sigma     = torch.tensor(config['sigma_ref']),
                n_points  = config['n_points'],
                noise_std = torch.tensor(config['noise_std']),
                n_samples = config['n_samples'], 
                f_center_max_offset = f_center_max_offset,
                t_center_max_offset = t_center_max_offset,
            )
        elif config['dataset_name'] == 'time_frequency':
            dataset = datasets.GaussPulseDatasetTimeFrequency(
                sigma     = torch.tensor(config['sigma_ref']),
                n_points  = config['n_points'],
                noise_std = torch.tensor(config['noise_std']),
                n_samples = config['n_samples'], 
                f_center_max_offset = f_center_max_offset,
                t_center_max_offset = t_center_max_offset,
            )
        else:
            raise ValueError("dataset not defined: ", config['dataset_name'])

    
    gen = torch.Generator()
    gen.manual_seed(0)
    trainset, validset, testset = torch.utils.data.random_split(
        dataset, [0.7, 0.1, 0.2],
        generator=gen
    )

    return trainset, validset, testset

def get_model_by_config(config):
    if config['dataset_name'] == 'time_frequency':
        n_classes = 3
    elif config['dataset_name'] == 'audio_mnist':
        n_classes = 10
    elif config['dataset_name'] == 'esc50':
        n_classes = 50
    else:
        raise ValueError('dataset_name: {} not supported.'.format(config['dataset_name']))

    if config['model_name'] == 'linear_net':
        net = models.LinearNet(
            n_classes  = n_classes,
            init_lambd = torch.tensor(config['init_lambd']),
            device     = config['device'],
            size       = (config['n_points']+1, config['n_points']+1),
            hop_length = config['hop_length'],
            optimized  = config['optimized'],
            normalize_window = config['normalize_window'],
        )
    elif config['model_name'] == 'non_linear_net':
        net = models.NonLinearNet(
            n_classes  = n_classes,
            init_lambd = torch.tensor(config['init_lambd']),
            device     = config['device'],
            size       = (config['n_points']+1, config['n_points']+1),
            hop_length = config['hop_length'],
            optimized  = config['optimized'],
            normalize_window = config['normalize_window'],
        )
    elif config['model_name'] == 'mlp_net':
        net = models.MlpNet(
            n_classes  = n_classes,
            init_lambd = torch.tensor(config['init_lambd']),
            device     = config['device'],
            size       = (config['n_points']+1, config['n_points']+1),
            hop_length = config['hop_length'],
            optimized  = config['optimized'],
            normalize_window = config['normalize_window'],
        )
    elif config['model_name'] == 'conv_net':
        net = models.ConvNet(
            n_classes  = n_classes,
            init_lambd = torch.tensor(config['init_lambd']),
            device     = config['device'],
            size       = (config['n_points']+1, config['n_points']+1),
            hop_length = config['hop_length'],
            optimized  = config['optimized'],
            normalize_window = config['normalize_window'],
        )
    elif config['model_name'] == 'mel_linear_net':
        net = models.MelLinearNet(
            n_classes   = n_classes,
            init_lambd  = torch.tensor(config['init_lambd']),
            device      = config['device'],
            n_mels      = config['n_mels'],
            sample_rate = config['resample_rate'],
            n_points    = config['n_points'],
            hop_length  = config['hop_length'],
            optimized   = config['optimized'],
            energy_normalize = config['energy_normalize'],
            normalize_window = config['normalize_window'],
        )
    elif config['model_name'] == 'mel_mlp_net':
        net = models.MelMlpNet(
            n_classes   = n_classes,
            init_lambd  = torch.tensor(config['init_lambd']),
            device      = config['device'],
            n_mels      = config['n_mels'],
            sample_rate = config['resample_rate'],
            n_points    = config['n_points'],
            hop_length  = config['hop_length'],
            optimized   = config['optimized'],
            energy_normalize = config['energy_normalize'],
            normalize_window = config['normalize_window'],
        )
    elif config['model_name'] == 'mel_conv_net':
        net = models.MelConvNet(
            n_classes   = n_classes,
            init_lambd  = torch.tensor(config['init_lambd']),
            device      = config['device'],
            n_mels      = config['n_mels'],
            sample_rate = config['resample_rate'],
            n_points    = config['n_points'],
            hop_length  = config['hop_length'], #int(config['resample_rate'] * config['hop_time_s']),
            optimized   = config['optimized'], #True,
            energy_normalize = config['energy_normalize'],
            normalize_window = config['normalize_window'],
        )
    elif config['model_name'] == 'panns_cnn6':
        net = models.MelPANNsNet(
            n_classes   = n_classes,
            init_lambd  = torch.tensor(config['init_lambd']),
            device      = config['device'],
            n_mels      = config['n_mels'],
            sample_rate = config['resample_rate'],
            n_points    = config['n_points'],
            hop_length  = config['hop_length'],
            optimized   = config['optimized'],
            energy_normalize = config['energy_normalize'],
            normalize_window = config['normalize_window'],
        )
    else:
        raise ValueError("model name not found: ", config['model_name'])

    return net

def get_predictions_by_row(row, data_dir, device='cpu', split='valid'):
    config = get_config_by_row(row)
    config['device'] = device
    logdir = row[1]['logdir']
    model_chechpoint_path = os.path.join(logdir, 'checkpoint_000000', 'best_model')
    net = get_model_by_config(config)
    (model_state, optimizer_state) = torch.load(model_chechpoint_path)
    net.load_state_dict(model_state)
    net.to(device)

    # load dataset
    trainset,validset,testset = get_dataset_by_config(config, data_dir)

    if split == 'valid':
        dataset = validset
    elif split == 'train':
        dataset = trainset
    elif split == 'test':
        dataset = testset
    else:
        raise ValueError("data split: {} is not supported".format(split))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    net.eval()
    all_predictions = []
    all_labels = []
    for data in tqdm.tqdm(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs, _ = net(inputs)
        predictions = torch.argmax(outputs, axis=1)

        all_labels.append(labels.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_predictions)

def plot_spectrogram(s, ax, decorate_axes=True):
    ax.imshow(np.flip(s, axis=0), aspect='auto')
    
    # decorate axes
    if decorate_axes:
        ax.set_xlabel('time')
        ax.set_ylabel('normalized frequency')
    
    (fbins, tbins) = s.shape
    yticks = [t for t in np.linspace(0, fbins-1, 5)]
    yticklabels = [str(l) for l in np.linspace(0.5, 0, 5)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)


# kept in case I need it again
def sample_d_tloc_d_floc_ellipse(sigma):
    
    d_freq = 1/(torch.pi * sigma)
    d_time = 2 * sigma
    
    angle = torch.rand(1) * torch.pi * 2
    d_tloc = torch.sin(angle) * d_time
    d_floc = torch.cos(angle) * d_freq
    
    return d_tloc, d_floc

def sample_d_tloc_d_floc_between_ellipses(sigma, scale):
    r1 = 1/(torch.pi * sigma)
    r2 = r1 * scale #1/(torch.pi * scale * sigma)
    d_freq = (r1 - r2) * torch.rand(1) + r2
    
    r1 = 2 * sigma
    r2 = 2 * scale * sigma
    
    d_time = (r1 - r2) * torch.rand(1) + r2
    
    angle = torch.rand(1) * torch.pi * 2
    d_tloc = torch.sin(angle) * d_time
    d_floc = torch.cos(angle) * d_freq
    
    return d_tloc, d_floc

def sample_around_optimal_ellipse(t_loc, f_loc, sigma, scale=2):
    d_tloc, d_floc = sample_d_tloc_d_floc_between_ellipses(sigma, scale)
    return t_loc + d_tloc, f_loc + d_floc

def sample_on_optimal_ellipse(t_loc, f_loc, sigma):
    d_tloc, d_floc = sample_d_tloc_d_floc_ellipse(sigma)
    return t_loc + d_tloc, f_loc + d_floc

def sample_on_optimal_ellipse(t_loc, f_loc, sigma):
    d_tloc, d_floc = sample_d_tloc_d_floc_ellipse(sigma)
    return t_loc + d_tloc, f_loc + d_floc


