import os
import librosa
import tqdm
import torch
import numpy as np

import time_frequency as tf

def fmconst(n_points, fnorm=0.25):
    ts = torch.arange(n_points)
    # TODO: should we really have random phase on these
    random_phase = torch.rand(1) * (2*torch.pi)
    #random_phase = 0
    
    y = torch.sin(2.0 * torch.pi * fnorm * ts + random_phase)
    #y = torch.exp(1j * (2.0 * torch.pi * fnorm * ts + random_phase))
    y = y / torch.max(torch.real(y))
    return torch.real(y)

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

def gauss_pulse(t_loc, f_loc, sigma, n_points):
    gauss_window = tf.gauss_whole(sigma, t_loc, n_points)
    fm_signal = fmconst(n_points, f_loc)
    gp = gauss_window * fm_signal
    
    # TODO: should these have amplitude 1
    #gp = gp / torch.max(gp)
    return gp - torch.mean(gp)

def torch_random_uniform(limits):
    r1, r2 = limits
    x = (r1 - r2) * torch.rand(1) + r2
    return x

class GaussPulseDataset(torch.utils.data.Dataset):
    def __init__(self, sigma, n_points, noise_std, n_samples=10000):
        
        self.xs = torch.empty((n_samples, n_points), dtype=torch.float64)
        self.ys = torch.empty((n_samples), dtype=torch.long)
        self.locs = torch.zeros((n_samples, 4), dtype=torch.float64)

        d_freq = 1/(torch.pi * sigma)
        print("freq res: ", d_freq)
        # TODO: the factor of two should be removed for theoretical limit
        d_time = sigma * 2
        print("time res: ", d_time)
        
        f_lims = [d_freq*2, 0.5-d_freq*2]
        print("freq lims: ", f_lims)
        t_lims = [d_time*2, n_points-d_time*2]
        print("time lims: ", t_lims)

        # generate samples
        for idx in range(n_samples):
            t_loc = torch.tensor(np.random.uniform(low=t_lims[0], high=t_lims[1]), dtype=torch.float64)
            f_loc = torch.tensor(np.random.uniform(low=f_lims[0], high=f_lims[1]), dtype=torch.float64)
            y = np.random.choice([0, 1, 2])
            #if y == 0:
            #    sigma_scale = torch_random_uniform([0.8, 1.2])
            #else:
            sigma_scale = 1
            x = gauss_pulse(t_loc, f_loc, sigma*sigma_scale, n_points)

            scale = torch_random_uniform([0.2, 1])

            self.locs[idx, 0] = t_loc
            self.locs[idx, 1] = f_loc

            if y > 0:
                t_loc_o = t_loc
                f_loc_o = f_loc
                if y == 1:
                    t_loc_o = torch.tensor(np.random.choice([t_loc+d_time, t_loc-d_time]), dtype=torch.float64)
                if y == 2:
                    f_loc_o = torch.tensor(np.random.choice([f_loc+d_freq, f_loc-d_freq]), dtype=torch.float64)
                x_o = gauss_pulse(t_loc_o, f_loc_o, sigma, n_points)
                x = x + x_o

                self.locs[idx, 2] = t_loc_o
                self.locs[idx, 3] = f_loc_o

            # variability
            noise = noise_std * torch.rand(n_points)
            x = (x * scale) + noise
            
            self.ys[idx] = torch.tensor(y, dtype=torch.long)
            self.xs[idx] = x - torch.mean(x)
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

class GaussPulseDatasetTime(torch.utils.data.Dataset):
    def __init__(self, sigma, n_points, noise_std, n_samples=10000):
        
        self.xs = torch.empty((n_samples, n_points), dtype=torch.float64)
        self.ys = torch.empty((n_samples), dtype=torch.long)
        self.locs = torch.zeros((n_samples, 4), dtype=torch.float64)

        s = 2
        t_max = sigma*s #n_points/2 - sigma*4
        t_center = torch.tensor(n_points/2, dtype=torch.float)

        sigma_scale_max = ((s * 2) + 3) / 6 #n_points/(6*sigma)

        # generate samples
        for idx in range(n_samples):
            t_offset = torch.rand(1) * t_max
            y = np.random.choice([0, 1])

            f_loc = 0.25

            if y == 0:
                t_loc = t_center

                a = 0.5
                sigma_scale = (torch.rand(1)*(1-a) + a) * sigma_scale_max
                
                x = gauss_pulse(t_loc, f_loc, sigma*sigma_scale, n_points)

                self.locs[idx, 0] = t_loc
                self.locs[idx, 1] = f_loc
            else:
                t_loc_1 = t_center - t_offset
                t_loc_2 = t_center + t_offset

                x1 = gauss_pulse(t_loc_1, f_loc, sigma, n_points)
                x2 = gauss_pulse(t_loc_2, f_loc, sigma, n_points)
                x = x1 + x2

                self.locs[idx, 0] = t_loc_1
                self.locs[idx, 1] = f_loc
                self.locs[idx, 2] = t_loc_2
                self.locs[idx, 3] = f_loc

            # variability
            noise = noise_std * torch.rand(n_points)
            amplitude_scale = torch_random_uniform([0.5, 1])
            x = (x * amplitude_scale) + noise
            
            self.ys[idx] = torch.tensor(y, dtype=torch.long)
            self.xs[idx] = x - torch.mean(x)
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


def parse_row(row):
    filename = row[0]
    fold = int(row[1])
    target = int(row[2])
    category = row[3]
    
    return filename, fold, target, category

def parse_csv(csv_file):
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    meta = []
    for line in lines[1:]:
        row = line.rstrip().split(',')
        filename, fold, target, category = parse_row(row)
        meta.append((filename, fold, target, category))
    return meta

def load_meta_data(source_dir):
    csv_file = os.path.join(source_dir, 'meta', 'esc50.csv')
    meta_data = parse_csv(csv_file)
    return meta_data

class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, source_dir, resample_rate=8000):
        meta_data = load_meta_data(source_dir)
        
        self.xs = []
        self.ys = []
        self.categories = []
        self.folds = []
        self.sample_rate = None
        
        sample_rates = []
        for (filename, fold, target, category) in tqdm.tqdm(meta_data):
            # load audio
            audio_file = os.path.join(source_dir, 'audio', filename)
            audio, sr = librosa.load(audio_file, sr=resample_rate, res_type='kaiser_fast')
            sample_rates.append(sr)
            self.xs.append(audio)
            self.ys.append(target)
            self.categories.append(category)
            self.folds.append(fold)
            
        # assert all files have the same sample rates
        assert len(list(set(sample_rates))) == 1
        
        self.sample_rate = sample_rates[0]
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
