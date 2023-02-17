from ray import tune

def development_space(max_epochs):
    search_space = {
        # model
        'model_name' : 'mlp_net', #tune.choice(['linear_net', 'mlp_net', 'conv_net']),

        # training
        'optimizer_name' : 'sgd',
        'lr_model' : 1e-4, #tune.choice([1e-4, 1e-3]), #tune.sample_from(lambda spec: 1e-4 if spec.config.model_name == 'conv_net' else 1e-3),
        'lr_tf' : tune.choice([0.1, 1.0, 10]),
        'batch_size' : 64,
        'epochs' : 500,
        'trainable' : True, #False, #tune.choice([True, False]),
        'max_epochs' : max_epochs,
        'patience' : 5,
        'device' : 'cuda:0',
        
        # dataset
        'n_points' : 128,
        'noise_std' : 0.5, #tune.choice([0.1, 0.5, 1.0]),
        'sigma_scale' : tune.choice([0.125, 0.25, 0.5, 1.0, 2.0, 4.0]),
        'n_samples' : 2000, 
        'sigma_ref' : 6.38,
        'dataset_name' : 'time_frequency', 
        'center_offset' : True, 
    }

    return search_space
