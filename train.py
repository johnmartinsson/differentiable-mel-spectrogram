import torch
import numpy as np

def train_model(net, optimizer, loss_fn, trainloader, validloader, patience, max_epochs, verbose=1, device='cuda:0'):
    history = {
        "best_valid_acc" : 0,
        "best_valid_loss" : np.inf,
        "init_sigma" : net.spectrogram_layer.sigma.item(),
        "converged" : False,
    }
    best_valid_acc = 0
    best_valid_loss = np.inf
    patience_count = 0
    for epoch in range(max_epochs):

        net.train()

        running_loss = 0.0
        count = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits, _ = net(inputs)
            loss = loss_fn(logits, labels)# + aux_loss
            loss.backward()
            optimizer.step()

            if verbose >= 2:
                if i % 10 == 0:
                    print("max values: ", torch.max(logits, dim=1).values.cpu().detach().numpy())
                    print("batch loss = {}".format(loss.item()))
                    print("est. sigma = ", net.spectrogram_layer.sigma.item())

            running_loss += loss.item()
            count += 1
            
        train_loss = running_loss / count

        if verbose >= 1:
            print("epoch {}, train loss = {}".format(epoch, running_loss / count))
            print("est. sigma = ", net.spectrogram_layer.sigma.item())

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
        if valid_loss < best_valid_loss: # < best_valid_loss:
            best_valid_acc = valid_acc
            best_valid_loss = valid_loss
            patience_count = 0
            if verbose >= 1:
                print("new best valid acc  = {}, patience_count = {}".format(best_valid_acc, patience_count))
        else:
            patience_count += 1

        if verbose >= 1:
            print("epoch {}, valid loss = {}".format(epoch, valid_loss))
            print("epoch {}, valid acc  = {}".format(epoch, valid_acc))
            
            # plot spectrogram
            plt.imshow(np.flip(spectrograms[0,0,:,:].cpu().detach().numpy(), axis=0), aspect='auto')
            plt.title("label = {}".format(labels.cpu().detach().numpy()[0]))
            plt.show()
            
        running_loss = 0.0
        running_acc  = 0.0
        count = 0

        if patience_count >= patience:
            print("no more patience, break training loop ...")
            history["converged"] = True
            break
            
    # save history
    history["best_valid_acc"] = best_valid_acc
    history["best_valid_loss"] = best_valid_loss
    history["est_sigma"] = net.spectrogram_layer.sigma.item()
    
    return net, history
