from ray import tune

def esc50(max_epochs):
    resample_rate = 8000
    search_space = {
        # model
        'model_name' : 'panns_cnn6', #tune.grid_search(['mel_conv_net', 'mel_linear_net']),
        'n_mels' : 64,
        'hop_length' :int(resample_rate * 0.025),
        'energy_normalize' : True,
        'optimized' : True,
        'normalize_window' : True,

        # training
        'optimizer_name' : 'adam',
        'lr_model' : 1e-3,
        'lr_tf' : 5e-3,
        'batch_size' : 64,
        'trainable' : False, #tune.grid_search([True, False]),
        'max_epochs' : max_epochs,
        'patience' : 200,
        'device' : 'cuda:0',
        
        # dataset
        'resample_rate' : resample_rate,
        #'init_lambd' : tune.grid_search([(resample_rate*x)/6 for x in [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]]),
        'init_lambd' : tune.grid_search([(resample_rate*x)/6 for x in [0.05]]),
        'dataset_name' : 'esc50', 
        'n_points' : resample_rate * 5, # hard coded zero-padding
    }

    return search_space

def audio_mnist(max_epochs):
    resample_rate = 8000
    search_space = {
        # model
        'model_name' : tune.grid_search(['mel_conv_net', 'mel_linear_net']),
        'n_mels' : 128,
        'hop_length' :int(resample_rate * 0.010),
        'energy_normalize' : True,
        'optimized' : False,
        'normalize_window' : False,

        # training
        'optimizer_name' : 'sgd',
        'lr_model' : 1e-3,
        'lr_tf' : 1e2,
        'batch_size' : 64,
        'trainable' : tune.grid_search([True, False]),
        'max_epochs' : max_epochs,
        'patience' : 10,
        'device' : 'cuda:0',
        
        # dataset
        'resample_rate' : resample_rate,
        'init_lambd' : tune.grid_search([(resample_rate*x)/6 for x in [0.01, 0.025, 0.05, 0.1, 0.2, 0.4]]),
        'dataset_name' : 'audio_mnist', 
        'n_points' : 5500, # hard coded zero-padding
    }

    return search_space

def time_frequency(max_epochs):
    sigma_ref = 6.38

    search_space = {
        # model
        'model_name' : tune.grid_search(['conv_net', 'linear_net']),
        'hop_length' : 1,
        'optimized'  : False,
        'normalize_window' : False,

        # training
        'optimizer_name' : 'sgd',
        'lr_model'       : 1e-3, 
        'lr_tf'          : 1e-1,
        'batch_size'     : 128,
        'trainable'      : tune.grid_search([True, False]),
        'max_epochs'     : max_epochs,
        'patience'       : 10,
        'device'         : 'cuda:0',
        
        # dataset
        'n_points'      : 128,
        'noise_std'     : 0.5,
        'init_lambd'    : tune.grid_search([x * sigma_ref for x in [0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6]]),
        'n_samples'     : 5000, 
        'sigma_ref'     : sigma_ref,
        'dataset_name'  : 'time_frequency', 
        'center_offset' : False, 
    }

    return search_space
