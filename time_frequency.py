import torch
import torch.nn as nn
import torch.nn.functional as F

def gauss_whole(sigma, tc, signal_length, norm='amplitude', device='cpu'):    
    ts = torch.arange(0, signal_length).float()
    
    ts = ts.to(device)
    tc = tc.to(device)
    sigma = sigma.to(device)
    
    window = torch.exp(-0.5 * torch.pow((ts-tc) / (sigma + 1e-15), 2))
    
    if norm == 'energy':
        window_norm = window / torch.sum(torch.pow(window, 2))
    elif norm == 'amplitude':
        window_norm = window / torch.max(window)
    
    return window_norm

def differentiable_gaussian_window(lambd, window_length, device='cpu', norm=True):
    m = torch.arange(0, window_length).float().to(device)
    
    window = torch.exp(-0.5 * torch.pow((m - window_length / 2) / (lambd + 1e-15), 2))
    window_norm = window / torch.sqrt(torch.sum(torch.pow(window, 2)))

    if norm:
        return window_norm
    else:
        return window

def differentiable_spectrogram(x, lambd, optimized=False, device='cpu', hop_length=1, return_window=False, norm=False, n_stds=6):

    # optimization potentially makes gradients weaker, but faster
    if optimized:
        # TODO: not sure if this optimization works as intended,
        # never used in the experiments. Will be become important
        # for longer signals.
        window_length = next_power_of_2((lambd * n_stds).detach().cpu().numpy())
    else:
        window_length = len(x)
    
    window = differentiable_gaussian_window(lambd, window_length=window_length, device=device, norm=norm).to(device)
    n_fft = len(window)
    
    # quadratic TF-image without redundancy
    if optimized:
        s = torch.stft(x, n_fft=len(window), hop_length=hop_length, win_length=len(window), window=window, return_complex=True, pad_mode='constant')
    else:
    # quadratic TF-image with redundancy
        s = torch.stft(x, n_fft=len(window)*2, hop_length=hop_length, win_length=len(window), window=window, return_complex=True, pad_mode='constant')
    
    s = torch.pow(torch.abs(s), 2)

    if not return_window:
        return s
    else:
        return s, window

def shift_bit_length(x):
    x = int(x)
    return 1<<(x-1).bit_length()

def next_power_of_2(x):
    return shift_bit_length(x)


# NOTE: initial time-frequency implementation from scratch,
# a bit slower than the pytorch implementation, kept here
# if found useful in the future, or simply for educational
# purposes.

#def stft(x, windows):
#    dim = (len(x) // 2 + 1, len(windows))
#    s = torch.empty(dim, dtype=torch.complex64)
#    for idx, window in enumerate(windows):
#        x_w = x * window
#        fft = torch.fft.rfft(x_w)
#        s[:,idx] = fft
#    
#    return s
#
#def spectrogram(x, lambd, overlap=0.5):
#    signal_length = len(x)
#    windows = get_gauss_windows(signal_length, lambd, overlap)
#    s = stft(x, windows)
#    s = torch.pow(torch.abs(s), 2)
#    return s
#
#def spectrogram_whole(x, lambd, device='cpu'):
#    signal_length = len(x)
#    windows = get_gauss_windows_whole(signal_length, lambd, device=device)
#    windows = [w.to(device) for w in windows]
#    s = stft(x, windows)
#    s = torch.pow(torch.abs(s), 2)
#    return s
#
#def get_gauss_windows(signal_length, lambd, overlap):
#    hop_size = (lambd*6*(1-overlap)).int()
#    windows = [
#        gauss_whole(sigma=lambd, tc=(i+1)*hop_size, signal_length=signal_length)
#        for i in range(signal_length // hop_size - 1)
#    ]
#    
#    return windows
#
#def get_gauss_windows_whole(signal_length, lambd, device='cpu'):
#    half_window_length = 0 #(lambd*3).int()
#    windows = [
#        gauss_whole(sigma=lambd, tc=torch.tensor(i).float(), signal_length=signal_length, device=device)
#        for i in range(half_window_length, signal_length-half_window_length)
#    ]
#    
#    return windows

