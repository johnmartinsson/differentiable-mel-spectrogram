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
