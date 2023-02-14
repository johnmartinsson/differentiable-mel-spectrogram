import os
import librosa
import tqdm
import torch
import numpy as np

import time_frequency as tf

def fmconst(n_points, fnorm=0.25):
    ts = torch.arange(n_points)
    random_phase = torch.rand(1) * (2*torch.pi)
    
    y = torch.sin(2.0 * torch.pi * fnorm * ts + random_phase)
    y = y / torch.max(y)
    return y

def gauss_pulse(t_loc, f_loc, sigma, n_points):
    gauss_window = tf.gauss_whole(sigma, t_loc, n_points)
    fm_signal = fmconst(n_points, f_loc)
    gp = gauss_window * fm_signal
    
    return gp - torch.mean(gp)

def torch_random_uniform(limits):
    r1, r2 = limits
    x = (r1 - r2) * torch.rand(1) + r2
    return x

class GaussPulseDatasetTimeFrequency(torch.utils.data.Dataset):
    def __init__(self, sigma, n_points, noise_std, n_samples=10000):
        
        self.xs = torch.empty((n_samples, n_points), dtype=torch.float64)
        self.ys = torch.empty((n_samples), dtype=torch.long)
        self.locs = torch.zeros((n_samples, 4), dtype=torch.float64)

	# maximum time-offset for pulses from center
        s = 2
        t_max = sigma*s

        # TODO: what should this value be?
        f_max = 0.1 #1/(np.pi*sigma) * s #0.1 #sigma*s


        # TODO: are these reasonable values?
	# maximum duration scaling for pulses on center
        sigma_scale_max = ((s + 3)*2) / 6
	# minimum duration scaling for pulses on center
        sigma_scale_min = 1 / sigma_scale_max 

        print("sigma_scale_min :", sigma_scale_min)
        print("sigma_scale_max :", sigma_scale_max)

        # TODO: center offset
        t_center = torch.tensor(n_points/2, dtype=torch.float)
        f_center = 0.25

        # generate samples
        for idx in range(n_samples):
            f_offset = torch.rand(1) * f_max
            t_offset = torch.rand(1) * t_max

            y = np.random.choice([0, 1, 2])

            if y == 0:
                t_loc = t_center
                f_loc = f_center

                # spread randomly along frequency or time axis
                r = np.random.choice([True, False])
                if r:
                    a = 1.0
                    # uniform [1.0, sigma_scale_max]
                    sigma_scale = a + torch.rand(1) * (sigma_scale_max-a)
                else:
                    b = 1.0
                    # uniform [sigma_scale_min, 1.0]
                    sigma_scale = sigma_scale_min + torch.rand(1) * (b-sigma_scale_min)
                    
                x = gauss_pulse(t_loc, f_loc, sigma*sigma_scale, n_points)

                self.locs[idx, 0] = t_loc
                self.locs[idx, 1] = f_loc
                self.locs[idx, 2] = int(r)
                self.locs[idx, 3] = sigma_scale

            elif y == 1:
                f_loc = f_center
                t_loc_1 = t_center - t_offset
                t_loc_2 = t_center + t_offset

                x1 = gauss_pulse(t_loc_1, f_loc, sigma, n_points)
                x2 = gauss_pulse(t_loc_2, f_loc, sigma, n_points)
                x = x1 + x2

                self.locs[idx, 0] = t_loc_1
                self.locs[idx, 1] = f_loc
                self.locs[idx, 2] = t_loc_2
                self.locs[idx, 3] = f_loc
            else:
                t_loc = t_center
                f_loc_1 = f_center - f_offset
                f_loc_2 = f_center + f_offset

                x1 = gauss_pulse(t_loc, f_loc_1, sigma, n_points)
                x2 = gauss_pulse(t_loc, f_loc_2, sigma, n_points)
                x = x1 + x2

                self.locs[idx, 0] = t_loc
                self.locs[idx, 1] = f_loc_1
                self.locs[idx, 2] = t_loc
                self.locs[idx, 3] = f_loc_2

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

class GaussPulseDatasetFrequency(torch.utils.data.Dataset):
    def __init__(self, sigma, n_points, noise_std, n_samples=10000):
        
        self.xs = torch.empty((n_samples, n_points), dtype=torch.float64)
        self.ys = torch.empty((n_samples), dtype=torch.long)
        self.locs = torch.zeros((n_samples, 4), dtype=torch.float64)

	# maximum frequency-offset for pulses from center
        s = 2

        # TODO: what should this value be?
        f_max = 0.1 #1/(np.pi*sigma) * s #0.1 #sigma*s

	# minimum duration scaling for pulses on center
        # TODO: what should this value be?
        sigma_scale_min = 1 / (((s + 3)*2) / 6)
        print("sigma_scale_min :", sigma_scale_min)

        t_center = torch.tensor(n_points/2, dtype=torch.float)
        f_center = 0.25

        # generate samples
        for idx in range(n_samples):
            f_offset = torch.rand(1) * f_max
            y = np.random.choice([0, 1])

            if y == 0:
                t_loc = t_center
                f_loc = f_center

                b = 1.0
                sigma_scale = sigma_scale_min + torch.rand(1) * (b-sigma_scale_min)
                
                x = gauss_pulse(t_loc, f_loc, sigma*sigma_scale, n_points)

                self.locs[idx, 0] = t_loc
                self.locs[idx, 1] = f_loc
            else:
                t_loc = t_center
                f_loc_1 = f_center - f_offset
                f_loc_2 = f_center + f_offset

                x1 = gauss_pulse(t_loc, f_loc_1, sigma, n_points)
                x2 = gauss_pulse(t_loc, f_loc_2, sigma, n_points)
                x = x1 + x2

                self.locs[idx, 0] = t_loc
                self.locs[idx, 1] = f_loc_1
                self.locs[idx, 2] = t_loc
                self.locs[idx, 3] = f_loc_2

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

class GaussPulseDatasetTime(torch.utils.data.Dataset):
    def __init__(self, sigma, n_points, noise_std, n_samples=10000):
        
        self.xs = torch.empty((n_samples, n_points), dtype=torch.float64)
        self.ys = torch.empty((n_samples), dtype=torch.long)
        self.locs = torch.zeros((n_samples, 4), dtype=torch.float64)

	# maximum time-offset for pulses from center
        s = 2
        t_max = sigma*s

	# maximum duration scaling for pulses on center
	# this maxes the one-pulse signal vary with the same
	# spread as the two-pulse signal
        #sigma_scale_max = ((s * 2) + 3) / 6
        sigma_scale_max = ((s + 3)*2) / 6
        print("sigma_scale_max :", sigma_scale_max)

        t_center = torch.tensor(n_points/2, dtype=torch.float)

        # generate samples
        for idx in range(n_samples):
            t_offset = torch.rand(1) * t_max
            y = np.random.choice([0, 1])

            f_loc = 0.25

            if y == 0:
                t_loc = t_center

                a = 1.0
                sigma_scale = a + torch.rand(1) * (sigma_scale_max-a)
                
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
