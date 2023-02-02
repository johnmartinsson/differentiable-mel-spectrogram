import torch
import torch.nn as nn
import torch.nn.functional as F

def gauss_whole(sigma, tc, signal_length, norm='amplitude', device='cpu'):    
    ts = torch.arange(0, signal_length).float()
    
    ts = ts.to(device)
    tc = tc.to(device) #torch.tensor(tc, dtype=torch.float64).to(device)
    sigma = sigma.to(device) #torch.tensor(sigma, dtype=torch.float64).to(device)
    
    window = torch.exp(-0.5 * torch.pow((ts-tc) / (sigma + 1e-15), 2))
    
    if norm == 'density':
        window_norm = window / torch.sum(window)
    elif norm == 'amplitude':
        window_norm = window / torch.max(window)
    
    return window_norm

def stft(x, windows):
    dim = (len(x) // 2 + 1, len(windows))
    s = torch.empty(dim, dtype=torch.complex64)
    for idx, window in enumerate(windows):
        x_w = x * window
        fft = torch.fft.rfft(x_w)
        s[:,idx] = fft
    
    return s

def spectrogram(x, sigma, overlap=0.5):
    signal_length = len(x)
    windows = get_gauss_windows(signal_length, sigma, overlap)
    s = stft(x, windows)
    s = torch.pow(torch.abs(s), 2)
    return s

def spectrogram_whole(x, sigma, device='cpu'):
    signal_length = len(x)
    windows = get_gauss_windows_whole(signal_length, sigma, device=device)
    windows = [w.to(device) for w in windows]
    s = stft(x, windows)
    s = torch.pow(torch.abs(s), 2)
    return s

def get_gauss_windows(signal_length, sigma, overlap):
    hop_size = (sigma*6*(1-overlap)).int()
    windows = [
        gauss_whole(sigma=sigma, tc=(i+1)*hop_size, signal_length=signal_length)
        for i in range(signal_length // hop_size - 1)
    ]
    
    return windows

def get_gauss_windows_whole(signal_length, sigma, device='cpu'):
    half_window_length = 0 #(sigma*3).int()
    windows = [
        gauss_whole(sigma=sigma, tc=torch.tensor(i).float(), signal_length=signal_length, device=device)
        for i in range(half_window_length, signal_length-half_window_length)
    ]
    
    return windows

def stft(x, windows):
    dim = (len(x) // 2 + 1, len(windows))
    s = torch.empty(dim, dtype=torch.complex64)
    for idx, window in enumerate(windows):
        x_w = x * window
        fft = torch.fft.rfft(x_w)
        s[:,idx] = fft
    
    return s

def spectrogram(x, sigma, overlap=0.5):
    signal_length = len(x)
    windows = get_gauss_windows(signal_length, sigma, overlap)
    s = stft(x, windows)
    s = torch.pow(torch.abs(s), 2)
    return s

def spectrogram_whole(x, sigma, device='cpu'):
    signal_length = len(x)
    windows = get_gauss_windows_whole(signal_length, sigma, device=device)
    windows = [w.to(device) for w in windows]
    s = stft(x, windows)
    s = torch.pow(torch.abs(s), 2)
    return s

def get_gauss_windows(signal_length, sigma, overlap):
    hop_size = (sigma*6*(1-overlap)).int()
    windows = [
        gauss_whole(sigma=sigma, tc=(i+1)*hop_size, signal_length=signal_length)
        for i in range(signal_length // hop_size - 1)
    ]
    
    return windows

def get_gauss_windows_whole(signal_length, sigma, device='cpu'):
    half_window_length = 0 #(sigma*3).int()
    windows = [
        gauss_whole(sigma=sigma, tc=torch.tensor(i).float(), signal_length=signal_length, device=device)
        for i in range(half_window_length, signal_length-half_window_length)
    ]
    
    return windows

def differentiable_gaussian_window(sigma, window_length, device='cpu'):
    #f_window_length = n_stds * sigma
    #window_length = torch.ceil(f_window_length).int()
    
    m = torch.arange(0, window_length).float().to(device)
    
    window = torch.exp(-0.5 * torch.pow((m - window_length / 2) / (sigma + 1e-15), 2))
    window_norm = window / torch.sum(torch.pow(window, 2)) #torch.max(window) # TODO: energy or max?
    
    return window

def shift_bit_length(x):
    x = int(x)
    return 1<<(x-1).bit_length()

def next_power_of_2(x):
    return shift_bit_length(x)

def differentiable_spectrogram(x, sigma, optimized=False, device='cpu', hop_length=1):
    n_stds = 10

    # optimization potentially makes gradients weaker, but faster
    if optimized:
        window_length = next_power_of_2((sigma * n_stds).detach().cpu().numpy())
    else:
        window_length = len(x)
    
    window = differentiable_gaussian_window(sigma, window_length=window_length, device=device).to(device)
    n_fft = len(window)
    
    # TODO: quadratic TF-image, n_fft=len(window)*2 could be removed to remove redundancy
    s = torch.stft(x, n_fft=len(window)*2, hop_length=hop_length, win_length=len(window), window=window, return_complex=True, pad_mode='constant')
    
    s = torch.pow(torch.abs(s), 2)

    return s


