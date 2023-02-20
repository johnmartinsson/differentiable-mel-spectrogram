from ray import tune

def development_space_audio_mnist(max_epochs):
    resample_rate = 8000
    search_space = {
        # model
        'model_name' : tune.choice(['mel_conv_net', 'mel_mlp_net']),
        'n_mels' : 128,
        'hop_length' :int(resample_rate * 0.010),
        'energy_normalize' : True, #tune.choice([True, False]),
        'optimized' : False, #tune.choice([True, False]),

        # training
        'optimizer_name' : 'sgd',
        'lr_model' : 1e-3, #0.001, #tune.choice([1e-3, 1e-2, 1e-1]), #1e-2, 1e-3]),
        'lr_tf' : 1000, #tune.choice([1.0, 10, 100]),
        'batch_size' : 64, #tune.choice([32, 64, 128]),
        'epochs' : 500,
        'trainable' : tune.choice([True, False]),
        'max_epochs' : max_epochs,
        'patience' : 5,
        'device' : 'cuda:0',
        
        # dataset
        'resample_rate' : resample_rate,
        'init_lambd' : tune.uniform(13, 700), #tune.choice([(8000*x)/6 for x in [0.01, 0.025, 0.050, 0.1, 0.2, 0.4, 0.8]]), #tune.sample_from(lambda spec: (spec.config.resample_rate*0.1) / 6), # ~25 ms
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
        'trainable' : tune.choice([True, False]),
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
        'model_name' : 'mel_mlp_net', #tune.choice(['linear_net', 'mlp_net', 'conv_net']),
        'hop_length' : 1,
        'optimized' : False,

        # training
        'optimizer_name' : 'sgd',
        'lr_model' : 1e-4, #tune.choice([1e-4, 1e-3]), #tune.sample_from(lambda spec: 1e-4 if spec.config.model_name == 'conv_net' else 1e-3),
        'lr_tf' : 1.0, #tune.choice([1e-4, 1e-3, 1e-2]),
        'batch_size' : 64,
        'epochs' : 500,
        'trainable' : True, #False, #tune.choice([True, False]),
        'max_epochs' : max_epochs,
        'patience' : 5,
        'device' : 'cuda:0',
        
        # dataset
        'n_points' : 128,
        'noise_std' : 0.1, #0.5, #tune.choice([0.1, 0.5, 1.0]),
        #'sigma_scale' : 0.125, #tune.choice([0.125, 0.25, 0.5, 1.0, 2.0, 4.0]),
        'init_lambd' : 0.125 * sigma_ref,
        'n_samples' : 2000, 
        'sigma_ref' : sigma_ref,
        'dataset_name' : 'time_frequency', 
        'center_offset' : False, #True, 

        # TODO: extra debugging
        'n_mels' : 129,
        'resample_rate' : 8000,
        'energy_normalize' : True
    }

    return search_space
