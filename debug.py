import torch
import utils

resample_rate = 8000
config = {
    # model
    'model_name' : 'mel_linear_net',
    'n_mels' : 128,
    'hop_length' :int(resample_rate * 0.01),
    'energy_normalize' : True,
    'optimized' : True,
    'normalize_window' : False,
    'augment' : False,
    'trainable' : True,

    # training
    'pretrained' : False,
    'checkpoint_path' : '/home/john/gits/differentiable-time-frequency-transforms/weights/Cnn6_mAP=0.343.pth',
    'optimizer_name' : 'adam',
    'lr_model' : 1e-3,
    'lr_tf' : 1e-1,
    'batch_size' : 16,
    'max_epochs' : 1,
    'patience' : 10000,
    'device' : 'cuda:0',
    
    # dataset
    'resample_rate' : resample_rate,
    'init_lambd' : resample_rate*0.025/6, 
    'dataset_name' : 'esc50', 
    'n_points' : resample_rate * 5,
}

trainset, validset, _ = utils.get_dataset_by_config(config, data_dir='./data/esc50')
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
    inputs, labels = inputs.to(config['device']), labels.to(config['device'])
    continue

logits, s = net(inputs)
loss = loss_fn(logits, labels)
print("batch_loss = ", loss.item())


# debug logits
print("logits: ", logits[0].detach().cpu().numpy())
print("softmax: ", torch.nn.functional.softmax(logits).detach().cpu().numpy()[0])
print("label: ", labels[0].detach().cpu().numpy())
print("spectrogram: ", s[0].detach().cpu().numpy())

# debug spectrogram

