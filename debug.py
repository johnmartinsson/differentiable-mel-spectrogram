import utils

def debug_esc50_training(config, data_dir):
    trainset, validset, _ = utils.get_dataset_by_config(config, data_dir)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    net = utils.get_model_by_config(config)
    net.to(config['device'])
    net.spectrogram_layer.requires_grad_(config['trainable'])

    if 'panns' in config['model_name']:
        one_hot = True
        loss_fn = torch.nn.functional.binary_cross_entropy
    else:
        one_hot = False
        loss_fn = torch.nn.CrossEntropyLoss()


    if one_hot:
        # TODO: this won't work in general
        labels = torch.nn.functional.one_hot(labels, 50).float()

    for idx_batch, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        continue

    logits, s = net(inputs)
    loss = loss_fn(logits, labels)
    print("batch_loss = ", loss.item())


    # debug logits

    # debug spectrogram
