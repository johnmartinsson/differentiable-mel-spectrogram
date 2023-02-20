import os
import librosa
import tqdm
import torch
import numpy as np
import glob

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
    def __init__(self, sigma, n_points, noise_std, n_samples=10000, f_center_max_offset=0, t_center_max_offset=0, demo=False):
        
        self.xs = torch.empty((n_samples, n_points), dtype=torch.float64)
        self.ys = torch.empty((n_samples), dtype=torch.long)
        self.locs = torch.zeros((n_samples, 4), dtype=torch.float64)

	# maximum time-offset for pulses from center
        #s = 2
        # TODO: n_points / 5 ?
        image_displacement = 5
        t_max = n_points / image_displacement #sigma*s

        # TODO: what should this value be?
        f_max = 0.5 / image_displacement #0.1 #1/(np.pi*sigma) * s #0.1 #sigma*s


        # TODO: are these reasonable values?
	# maximum duration scaling for pulses on center
        #sigma_scale_max = ((s + 3)*2) / 6
        sigma_scale_max = (2*t_max)/(6*sigma) + 1
	# minimum duration scaling for pulses on center
        sigma_scale_min = 1 / sigma_scale_max 

        print("sigma_scale_min :", sigma_scale_min)
        print("sigma_scale_max :", sigma_scale_max)

        # generate samples
        for idx in range(n_samples):
            if demo:
                f_center_offset = 0
                t_center_offset = 0
            else:
                f_center_offset = torch_random_uniform([-f_center_max_offset, f_center_max_offset])
                t_center_offset = torch_random_uniform([-t_center_max_offset, t_center_max_offset])

            t_center = t_center_offset + torch.tensor(n_points/2, dtype=torch.float)
            f_center = f_center_offset + 0.25

            if demo:
                f_offset = 0.4 * f_max
                t_offset = 0.4 * t_max
            else:
                f_offset = torch.rand(1) * f_max
                t_offset = torch.rand(1) * t_max

            y = np.random.choice([0, 1, 2])

            if y == 0:
                # spread randomly along frequency or time axis
                r = np.random.choice([True, False])
                if r:
                    sigma_scale = torch_random_uniform([1.0, sigma_scale_max])
                else:
                    sigma_scale = torch_random_uniform([sigma_scale_min, 1.0])

                if demo:
                    sigma_scale = 1.0
                    
                x = gauss_pulse(t_center, f_center, sigma*sigma_scale, n_points)

                # used to sanity check
                self.locs[idx, 0] = t_center
                self.locs[idx, 1] = f_center
                self.locs[idx, 2] = int(r)
                self.locs[idx, 3] = sigma_scale

            elif y == 1:
                f_loc = f_center
                t_loc_1 = t_center - t_offset
                t_loc_2 = t_center + t_offset

                x1 = gauss_pulse(t_loc_1, f_loc, sigma, n_points)
                x2 = gauss_pulse(t_loc_2, f_loc, sigma, n_points)
                x = x1 + x2

                # used to sanity check
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

                # used to sanity check
                self.locs[idx, 0] = t_loc
                self.locs[idx, 1] = f_loc_1
                self.locs[idx, 2] = t_loc
                self.locs[idx, 3] = f_loc_2

            # variability
            noise = noise_std * torch.rand(n_points)

            if demo:
                amplitude_scale = 1.0
            else:
                amplitude_scale = torch_random_uniform([0.5, 1])
            x = (x * amplitude_scale) + noise
            
            self.ys[idx] = torch.tensor(y, dtype=torch.long)
            self.xs[idx] = x - torch.mean(x)
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

class GaussPulseDatasetFrequency(torch.utils.data.Dataset):
    def __init__(self, sigma, n_points, noise_std, n_samples=10000, f_center_max_offset=0, t_center_max_offset=0):
        
        self.xs = torch.empty((n_samples, n_points), dtype=torch.float64)
        self.ys = torch.empty((n_samples), dtype=torch.long)
        self.locs = torch.zeros((n_samples, 4), dtype=torch.float64)

	# maximum time-offset for pulses from center
        # TODO: n_points / 5 ?
        image_displacement = 5
        t_max = n_points / image_displacement #sigma*s

        # TODO: what should this value be?
        f_max = 0.5 / image_displacement #0.1 #1/(np.pi*sigma) * s #0.1 #sigma*s

        # TODO: are these reasonable values?
	# maximum duration scaling for pulses on center
        sigma_scale_max = (2*t_max)/(6*sigma) + 1
	# minimum duration scaling for pulses on center
        sigma_scale_min = 1 / sigma_scale_max 

        # generate samples
        for idx in range(n_samples):
            f_center_offset = torch_random_uniform([-f_center_max_offset, f_center_max_offset])
            t_center_offset = torch_random_uniform([-t_center_max_offset, t_center_max_offset])

            t_center = t_center_offset + torch.tensor(n_points/2, dtype=torch.float)
            f_center = f_center_offset + 0.25

            y = np.random.choice([0, 1])

            if y == 0:
                sigma_scale = torch_random_uniform([sigma_scale_min, 1.0])
                
                x = gauss_pulse(t_center, f_center, sigma*sigma_scale, n_points)

                self.locs[idx, 0] = t_center
                self.locs[idx, 1] = f_center
            else:
                f_offset = torch.rand(1) * f_max
                f_loc_1 = f_center - f_offset
                f_loc_2 = f_center + f_offset

                x1 = gauss_pulse(t_center, f_loc_1, sigma, n_points)
                x2 = gauss_pulse(t_center, f_loc_2, sigma, n_points)
                x = x1 + x2

                self.locs[idx, 0] = t_center
                self.locs[idx, 1] = f_loc_1
                self.locs[idx, 2] = t_center
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
    def __init__(self, sigma, n_points, noise_std, n_samples=10000, f_center_max_offset=0, t_center_max_offset=0):
        
        self.xs = torch.empty((n_samples, n_points), dtype=torch.float64)
        self.ys = torch.empty((n_samples), dtype=torch.long)
        self.locs = torch.zeros((n_samples, 4), dtype=torch.float64)

	# maximum time-offset for pulses from center
        # TODO: n_points / 5 ?
        image_displacement = 5
        t_max = n_points / image_displacement #sigma*s

        # TODO: what should this value be?
        f_max = 0.5 / image_displacement #0.1 #1/(np.pi*sigma) * s #0.1 #sigma*s

        # TODO: are these reasonable values?
	# maximum duration scaling for pulses on center
        sigma_scale_max = (2*t_max)/(6*sigma) + 1
	# minimum duration scaling for pulses on center
        sigma_scale_min = 1 / sigma_scale_max 

        # generate samples
        for idx in range(n_samples):
            f_center_offset = torch_random_uniform([-f_center_max_offset, f_center_max_offset])
            t_center_offset = torch_random_uniform([-t_center_max_offset, t_center_max_offset])

            t_center = t_center_offset + torch.tensor(n_points/2, dtype=torch.float)
            f_center = f_center_offset + 0.25

            y = np.random.choice([0, 1])

            if y == 0:
                sigma_scale = torch_random_uniform([1.0, sigma_scale_max])
                
                x = gauss_pulse(t_center, f_center, sigma*sigma_scale, n_points)

                self.locs[idx, 0] = t_center
                self.locs[idx, 1] = f_center
            else:
                t_offset = torch.rand(1) * t_max
                t_loc_1 = t_center - t_offset
                t_loc_2 = t_center + t_offset

                x1 = gauss_pulse(t_loc_1, f_center, sigma, n_points)
                x2 = gauss_pulse(t_loc_2, f_center, sigma, n_points)
                x = x1 + x2

                self.locs[idx, 0] = t_loc_1
                self.locs[idx, 1] = f_center
                self.locs[idx, 2] = t_loc_2
                self.locs[idx, 3] = f_center

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
        #self.categories = []
        #self.folds = []
        self.sample_rate = None

        
        xs_path = os.path.join(source_dir, "{}_xs.npy".format(resample_rate))
        ys_path = os.path.join(source_dir, "{}_ys.npy".format(resample_rate))

        if os.path.exists(xs_path) and os.path.exists(ys_path):
            self.xs = np.load(xs_path)
            self.ys = np.load(ys_path)
            self.sample_rate = resample_rate
        else:
            sample_rates = []
            for (filename, fold, target, category) in tqdm.tqdm(meta_data):
                # load audio
                audio_file = os.path.join(source_dir, 'audio', filename)
                audio, sr = librosa.load(audio_file, sr=resample_rate, res_type='kaiser_fast')
                sample_rates.append(sr)
                self.xs.append(audio)
                self.ys.append(target)

            self.xs = np.array(self.xs)
            self.ys = np.array(self.ys)

            np.save(xs_path, self.xs)
            np.save(ys_path, self.ys)
                
            # assert all files have the same sample rates
            assert len(list(set(sample_rates))) == 1
            # assert proper resampling
            assert sample_rates[0] == resample_rate
            
            self.sample_rate = resample_rate
        
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

class AudioMNISTDataset(torch.utils.data.Dataset):
    # VERSION: https://doi.org/10.5281/zenodo.1342401
    def __init__(self, source_dir):
        # load data
        wav_paths = glob.glob(os.path.join(source_dir, 'recordings', '*.wav'))

        self.xs = []
        self.ys = []

        sample_rates = []
        for wav_path in wav_paths:
            audio, sr = librosa.load(wav_path, sr=None) # already 8000 Hz
            sample_rates.append(sr)
            target = int(os.path.basename(wav_path).split('_')[0])

            if len(audio) >= 1500 and len(audio) <= 5500:
                x = audio.copy()
                x.resize(5500) # pad end with zeros up to max length
                self.xs.append(x)
                self.ys.append(target)

        assert len(list(set(self.ys))) == 10 # assert 10 classes

        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)

        # assert all files have the same sample rates
        assert len(list(set(sample_rates))) == 1
        # assert proper sample rate
        assert sample_rates[0] == 8000

        self.sample_rate = 8000

    def __len__(self):
        return len(self.xs)
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
