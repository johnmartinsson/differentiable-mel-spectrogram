import numpy as np

def plot_spectrogram(s, ax):
    ax.imshow(np.flip(s, axis=0), aspect='auto')
    
    # decorate axes
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


