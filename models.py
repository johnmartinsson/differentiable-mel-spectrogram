import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import time_frequency as tf

class SpectrogramLayer(nn.Module):
    def __init__(self, init_lambd, device='cpu', optimized=False, size=(512, 1024), hop_length=1):
        super(SpectrogramLayer, self).__init__()
        
        self.hop_length = hop_length
        self.lambd      = nn.Parameter(init_lambd)
        self.device     = device
        self.size       = size #(512, 1024)
        self.optimized  = optimized
        
    def forward(self, x):
    
        (batch_size, n_points) = x.shape
        if self.optimized:
            spectrograms = torch.empty((batch_size, 1, self.size[0], self.size[1]), dtype=torch.float32).to(self.device)
        else:
            # TODO: redundancy in spectrogram, change to n_points // 2 + 1 again?
            spectrograms = torch.empty((batch_size, 1, n_points + 1, n_points // self.hop_length + 1), dtype=torch.float32).to(self.device)
        
        for idx in range(batch_size):
            spectrogram = tf.differentiable_spectrogram(x[idx]-torch.mean(x[idx]), torch.abs(self.lambd), optimized=self.optimized, device=self.device, hop_length=self.hop_length)

            if self.optimized:
                spectrogram = F.interpolate(torch.unsqueeze(torch.unsqueeze(spectrogram, axis=0), axis=0), size=(self.size[0], self.size[1]))
                spectrogram = spectrogram[0,0]

            spectrograms[idx,:,:,:] = torch.unsqueeze(spectrogram, axis=0)
        
        return spectrograms

class MelSpectrogramLayer(nn.Module):
    def __init__(self, init_lambd, n_mels, n_points, sample_rate, f_min=0, f_max=None, hop_length=1, device='cpu', optimized=False):
        super(MelSpectrogramLayer, self).__init__()
        
        self.hop_length = hop_length
        self.lambd      = nn.Parameter(init_lambd)
        self.device     = device
        self.optimized  = optimized

        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        self.n_freq = n_mels
        self.n_time = n_points // hop_length + 1

      
    def forward(self, x):
    
        (batch_size, n_points) = x.shape
        mel_spectrograms = torch.empty((batch_size, 1, self.n_freq, self.n_time), dtype=torch.float32).to(self.device)
        for idx in range(batch_size):
            spectrogram = tf.differentiable_spectrogram(x[idx]-torch.mean(x[idx]), torch.abs(self.lambd), device=self.device, optimized=self.optimized, hop_length=self.hop_length)

            (n_freq, _) = spectrogram.shape

            mel_fb = torchaudio.functional.melscale_fbanks(
                n_freqs = n_freq,
                f_min = self.f_min,
                f_max = self.f_max, 
                n_mels = self.n_mels,
                sample_rate = self.sample_rate,
            )

            mel_fb = mel_fb.to(self.device)
     
            mel_spectrogram = torch.matmul(spectrogram.transpose(-1, -2), mel_fb).transpose(-1, -2)
            mel_spectrograms[idx,:,:,:] = torch.unsqueeze(mel_spectrogram, axis=0)

        return mel_spectrograms

class MelLinearNet(nn.Module):
    def __init__(self, n_classes, init_lambd, device, n_mels, sample_rate, n_points, hop_length=1, optimized=False):
        super(MelLinearNet, self).__init__()
        self.spectrogram_layer = MelSpectrogramLayer(init_lambd, n_mels=n_mels, n_points=n_points, sample_rate=sample_rate, hop_length=hop_length, device=device, optimized=optimized)
        self.device = device
        self.size = (n_mels, n_points // hop_length + 1)
        
        self.fc = nn.Linear(self.size[0] * self.size[1], n_classes)

    def forward(self, x):
        # compute spectrograms
        s = self.spectrogram_layer(x)
        # normalization of s? PCEN?
        #s = torch.log(s + 1)

        # dropout on spectrogram
        x = F.dropout(s.view(-1, self.size[0] * self.size[1]), p=0.4)
        #x = s.view(-1, self.size[0] * self.size[1])
        x = self.fc(x)
        return x, s

class MelMlpNet(nn.Module):
    def __init__(self, n_classes, init_lambd, device, n_mels, sample_rate, n_points, hop_length=1, optimized=False):
        super(MelMlpNet, self).__init__()
        self.spectrogram_layer = MelSpectrogramLayer(init_lambd, n_mels=n_mels, n_points=n_points, sample_rate=sample_rate, hop_length=hop_length, device=device, optimized=optimized)
        self.device = device
        self.size = (n_mels, n_points // hop_length + 1)
        
        self.fc1 = nn.Linear(self.size[0] * self.size[1], 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        # compute spectrograms
        s = self.spectrogram_layer(x)
        # normalization of s? PCEN?
        #s = torch.log(s + 1)

        x = self.fc1(s.view(-1, self.size[0] * self.size[1]))
        x = F.relu(x)
        x = F.dropout(x, p=0.2)
        x = self.fc2(x)
        return x, s
 

class LinearNet(nn.Module):
    def __init__(self, n_classes, init_lambd, device, optimized=False, size=(512, 1024), hop_length=1):
        super(LinearNet, self).__init__()
        self.spectrogram_layer = SpectrogramLayer(init_lambd, device=device, optimized=optimized, size=size, hop_length=hop_length)
        self.device = device
        self.size = size
        
        self.fc = nn.Linear(size[0] * size[1], n_classes)

    def forward(self, x):
        # compute spectrograms
        s = self.spectrogram_layer(x)
        #x = F.dropout(s.view(-1, self.size[0] * self.size[1]), p=0.2)
        x = s.view(-1, self.size[0] * self.size[1])
        x = self.fc(x)
        return x, s

class MlpNet(nn.Module):
    def __init__(self, n_classes, init_lambd, device, optimized=False, size=(512, 1024), hop_length=1):
        super(MlpNet, self).__init__()
        self.spectrogram_layer = SpectrogramLayer(init_lambd, device=device, optimized=optimized, size=size, hop_length=hop_length)
        self.device = device
        self.size = size
        
        self.fc1 = nn.Linear(size[0] * size[1], 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        # compute spectrograms
        s = self.spectrogram_layer(x)
        x = self.fc1(s.view(-1, self.size[0] * self.size[1]))
        x = F.relu(x)
        x = F.dropout(x, p=0.2)
        x = self.fc2(x)
        return x, s

class ConvNet(nn.Module):
    def __init__(self, n_classes, init_lambd, device, optimized=False, size=(512, 1024), hop_length=1):
        super(ConvNet, self).__init__()
        self.spectrogram_layer = SpectrogramLayer(init_lambd, device=device, optimized=optimized, size=size, hop_length=hop_length)
        
        self.device = device
        self.size = size

        self.hidden_state = 32
        
        self.conv1 = nn.Conv2d(1, self.hidden_state, 5, padding='same')
        self.fc1 = nn.Linear(self.hidden_state * (size[0] // 2) * (size[1] // 2), self.hidden_state)
        self.fc2 = nn.Linear(self.hidden_state, n_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # compute spectrograms
        s = self.spectrogram_layer(x)

        # TODO: residual layers?
        x = self.conv1(s)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, self.hidden_state * (self.size[0] // 2) * (self.size[1] // 2))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x, s
