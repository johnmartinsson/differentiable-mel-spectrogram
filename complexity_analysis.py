# Mel spectrogram

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['text.usetex'] = True
mpl.rc('font', family = 'serif')

#init_mis = [0.01, 0.3]
init_mis = [0.02, 0.3]
mi_labels = [r'$l_{\lambda_{init}} = 20$ ms', r'$l_{\lambda_{init}} = 300$ ms']
#mi_labels = [r'$l_{\lambda_{init}} = 10$ ms', r'$l_{\lambda_{init}} = 300$ ms']

fig, ax = plt.subplots(1, 2, figsize=(5, 2.5)) #plt.figure(figsize=(4/2, 3/2))
for idx_mi in range(2):
    C1s = [0.0001, 0.9999]
    #c1_labels = [r'$C_2 \gg C_1$',r'$C_1 \gg C_2$']
    c1_labels = ['Cost dominated by NN', 'Cost dominated by FFT']

    for idx_C1 in range(2):
        C1 = C1s[idx_C1]
        C2 = 1-C1

        K_min, K_max = 1, 60
        Ks = np.arange(K_min, K_max) # number of models
        Fs = 8000 # sample rate
        n  = Fs * 5 # signal length
        
        M  = 128 # mel bins
        c  = 0.010 * Fs # hop length
        lr = 0.001 # pseudo-learning rate

        init_mi = init_mis[idx_mi] # initial window length
        #print("init_mi: ", init_mi)
        opt_mi = 0.035 # optimal window length
        
        B  = int(np.abs(init_mi - opt_mi)/lr) # number of forward passes
        #print(B)

        cost_quote = np.zeros(len(Ks))

        for idx_K, K in enumerate(Ks):
            base_mi = np.linspace(c*2, 0.3 * Fs, K) # TODO: discuss this
            ours_mi = np.linspace(init_mi * Fs, opt_mi * Fs, B)

            cost_base_tf = B * C1 * np.sum(n * np.log(base_mi)) # TODO: w/o DA, or w. enough space B dissapears
            #print("base_mi: ", base_mi)
            #print("ours_mi: ", ours_mi)
            #cost_base_tf = B * C1 * np.sum(n/base_mi * np.log(base_mi)) # TODO: w/o DA, or w. enough space B dissapears
            cost_base_nn = B * C2 * np.sum(2 * M * n / base_mi)
            cost_base    = cost_base_tf + cost_base_nn

            cost_ours_tf = C1 * n/c * np.sum(ours_mi * np.log(ours_mi))
            #cost_ours_tf = C1 * n/c * np.sum(np.log(ours_mi))
            cost_ours_nn = B * C2 * M * n / c
            cost_ours    = cost_ours_tf + cost_ours_nn

            cost_quote[idx_K] = cost_ours / cost_base

        ax[idx_C1].plot(Ks, cost_quote, label=mi_labels[idx_mi])
        #if idx_C1 == 0:
        #    ax[idx_C1].plot(Ks, cost_quote, label=mi_labels[idx_mi])
            #print(idx_mi)
        #else:
        #    ax[idx_C1].plot(Ks, cost_quote)

        ax[idx_C1].set_title(c1_labels[idx_C1])
        ax[idx_C1].set_xlabel('D')
        ax[idx_C1].set_ylim([0, 2.0])

plt.tight_layout()
ax[1].hlines(1, color='purple', xmin=K_min, xmax=K_max, label='reference', linestyle='dashed')
ax[0].hlines(1, color='purple', xmin=K_min, xmax=K_max, label='reference', linestyle='dashed')
ax[0].set_ylabel(r'$C_{DMEL} / C_{baseline}$')    
ax[0].legend()
#ax[1].set_yticks([])
ax[1].legend()
plt.savefig('time_complexity.pdf', bbox_inches='tight')