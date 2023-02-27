import torch
import numpy as np
import utils

device = 'cuda:0'
lambd_scale = 4.0
sigma_ref = 6.38
config = {
    # model
    'model_name' : 'linear_net',
    'hop_length' : 1,
    'optimized'  : False,

    # training
    'optimizer_name' : 'adam',
    'lr_model' : 1e-2,
    'lr_tf' : 1e-3,
    'batch_size' : 512,
    'trainable' : True,
    'max_epochs' : 400,
    'patience' : 50,
    'device' : device,

    # dataset
    'n_points' : 128,
    'noise_std' : 0.1,
    'init_lambd' : lambd_scale * sigma_ref,
    'n_samples' : 50000,
    'sigma_ref' : sigma_ref,
    'dataset_name' : 'time_frequency',
    'center_offset' : False,
}

print("initialize model ... ")

net = utils.get_model_by_config(config)
net = net.to(device)

net.spectrogram_layer.requires_grad_(config['trainable'])
print(net)

print("initialize dataset ... ")
trainset, validset, _ = utils.get_dataset_by_config(config)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
validloader = torch.utils.data.DataLoader(validset, batch_size=config['batch_size'], shuffle=True, num_workers=8)

loss_fn = torch.nn.CrossEntropyLoss()

if config['optimizer_name'] == 'sgd':
    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr_model'])
elif config['optimizer_name'] == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr_model'])
else:
    raise ValueError("optimizer not found: ", config['optimizer_name'])


init_lambd = net.spectrogram_layer.lambd.item()

for idx_epoch in range(config['max_epochs']):

    net.train()
    for idx_batch, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logits, s = net(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()

        lambd = net.spectrogram_layer.lambd
        
        if not lambd.grad is None:
            lambd_gradient = lambd.grad.cpu().detach().numpy()
        else:
            lambd_gradient = 0

        optimizer.step()
        
        if idx_batch % 10 == 0:
            print("epoch = {}, idx_batch = {}, init_lambd = {:.3f}, lambd = {:.3f}, gradient = {:.4f}, loss = {:.2f}".format(
                idx_epoch,
                idx_batch,
                init_lambd,
                lambd.item(),
                lambd_gradient,
                loss.item()
            ))

    # validation
    running_loss = 0.0
    count = 0
    running_acc = 0.0

    net.eval()
    for data in validloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs, spectrograms = net(inputs)
        loss = loss_fn(outputs, labels)

        predictions = torch.argmax(outputs, axis=1)

        accuracy = torch.mean((predictions == labels).float())
        running_acc += accuracy.item()

        running_loss += loss.item()
        count += 1

    valid_loss = running_loss / count
    valid_acc = running_acc / count
    print("-------------------------------------------------------------------------")
    print("VALIDATION")
    print("epoch = {}, validation loss = {}, validation accuracy = {}".format(idx_epoch, valid_loss, valid_acc))
    print("-------------------------------------------------------------------------")
