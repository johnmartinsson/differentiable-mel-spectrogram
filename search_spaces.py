from ray import tune

def development_space_audio_mnist(max_epochs):
    resample_rate = 8000
    search_space = {
        # model
        'model_name' : 'mel_mlp_net', #tune.grid_search(['mel_mlp_net', 'mel_linear_net']),
        'n_mels' : 128,
        'hop_length' :int(resample_rate * 0.010),
        'energy_normalize' : True,
        'optimized' : False,

        # training
        'optimizer_name' : 'adam',
        'lr_model' : 1e-3,
        'lr_tf' : 1e-2,
        'batch_size' : 64,
        'trainable' : True,
        'max_epochs' : max_epochs,
        'patience' : 5,
        'device' : 'cuda:0',
        
        # dataset
        'resample_rate' : resample_rate,
        'init_lambd' : tune.grid_search([(resample_rate*x)/6 for x in [0.005, 0.01, 0.025, 0.1, 0.2]]),
        #'init_lambd' : tune.grid_search([(resample_rate*x)/6 for x in [0.005, 0.025, 0.8]]),
        'dataset_name' : 'audio_mnist', 
        'n_points' : 5500, # hard coded zero-padding
    }

    return search_space

def audio_mnist_lambda_search_all_not_trainable(max_epochs):
    resample_rate = 8000
    search_space = {
        # model
        'model_name' : tune.grid_search(['mel_mlp_net', 'mel_linear_net']),
        'n_mels' : 128,
        'hop_length' :int(resample_rate * 0.010),
        'energy_normalize' : True,
        'optimized' : False,

        # training
        'optimizer_name' : 'adam',
        'lr_model' : 1e-3,
        'lr_tf' : 1000,
        'batch_size' : 64,
        'trainable' : False,
        'max_epochs' : max_epochs,
        'patience' : 5,
        'device' : 'cuda:0',
        
        # dataset
        'resample_rate' : resample_rate,
        'init_lambd' : tune.grid_search([(resample_rate*x)/6 for x in [0.005, 0.01, 0.025, 0.050, 0.1, 0.2, 0.4, 0.8]]),
        #'init_lambd' : tune.grid_search([(resample_rate*x)/6 for x in [0.005, 0.025, 0.8]]),
        'dataset_name' : 'audio_mnist', 
        'n_points' : 5500, # hard coded zero-padding
    }

    return search_space



def development_space_esc50(max_epochs):
    search_space = {
        # model
        'model_name' : 'mel_mlp_net',
        'n_mels' : 126, # square image with default hopsize of 40ms
        'hop_length' : int(8000 * 0.040),
        'energy_normalize' : True, #tune.choice([True, False]),
        'optimized' : True,

        # training
        'optimizer_name' : 'sgd',
        'lr_model' : tune.choice([0.01, 0.1, 1]), #1e-2, 1e-3]),
        'lr_tf' : tune.choice([0.1, 1.0, 10]),
        'batch_size' : 64,
        'epochs' : 500,
        'trainable' : True, #tune.choice([True, False]),
        'max_epochs' : max_epochs,
        'patience' : 5,
        'device' : 'cuda:0',
        
        # dataset
        'resample_rate' : 8000,
        'init_lambd' : tune.sample_from(lambda spec: (spec.config.resample_rate*0.1) / 6), # ~25 ms
        'dataset_name' : 'esc50', 
        'n_points' : 8000 * 5,
    }

    return search_space

def development_space(max_epochs):
    sigma_ref = 6.38

    search_space = {
        # model
        'model_name' : 'linear_net', #tune.choice(['linear_net', 'mlp_net', 'conv_net']),
        'hop_length' : 1,
        'optimized'  : False,

        # training
        'optimizer_name' : 'adam',
        'lr_model'       : 1e-3, 
        'lr_tf'          : 1e-1,
        'batch_size'     : 64,
        'trainable'      : tune.grid_search([True, False]),
        'max_epochs'     : max_epochs,
        'patience'       : 50,
        'device'         : 'cuda:0',
        
        # dataset
        'n_points'      : 128,
        'noise_std'     : 1.0,
        #'init_lambd'    : tune.grid_search([x * sigma_ref for x in [0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6]]),
        'init_lambd'    : tune.grid_search([x * sigma_ref for x in [0.25, 1.0, 3.0]]),
        'n_samples'     : 5000, 
        'sigma_ref'     : sigma_ref,
        'dataset_name'  : 'time_frequency', 
        'center_offset' : False,
    }

    return search_space

def time_frequency_sigma_search_all_not_trainable(max_epochs):
    sigma_ref = 6.38

    search_space = {
        # model
        'model_name' : tune.choice(['linear_net', 'mlp_net', 'conv_net']),
        'hop_length' : 1,
        'optimized'  : False,

        # training
        'optimizer_name' : 'adam',
        'lr_model'       : 1e-3, 
        'lr_tf'          : 1e-1,
        'batch_size'     : 64,
        'trainable'      : False,
        'max_epochs'     : max_epochs,
        'patience'       : max_epochs,
        'device'         : 'cuda:0',
        
        # dataset
        'n_points'      : 128,
        'noise_std'     : 0.1,
        'init_lambd'    : tune.choice([x * sigma_ref for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]]),
        'n_samples'     : 500, 
        'sigma_ref'     : sigma_ref,
        'dataset_name'  : 'time_frequency', 
        'center_offset' : False,
    }

    return search_space


