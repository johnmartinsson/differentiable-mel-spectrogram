import os
from ray import tune

import torch
import time
import numpy as np

def train_model(net, optimizer, loss_fn, trainloader, validloader, scheduler, patience, max_epochs, verbose=1, device='cuda:0', one_hot=False, n_classes=50):
    history = {
        "best_valid_acc" : 0,
        "best_valid_loss" : np.inf,
        "init_lambd" : net.spectrogram_layer.lambd.item(),
        "converged" : False,
    }
    best_valid_acc = 0
    best_valid_loss = np.inf
    patience_count = 0
    for epoch in range(max_epochs):

        net.train()

        running_loss = 0.0
        running_energy = 0.0
        count = 0
        for i, data in enumerate(trainloader):
            t_tot1 = time.time()
            inputs, labels = data

            if one_hot:
                # TODO: this won't work in general
                labels = torch.nn.functional.one_hot(labels, n_classes).float()

            t1 = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            t2 = time.time()
            #print("time 1 = {}, batch = {}".format(t2-t1, i))


            optimizer.zero_grad()

            t1 = time.time()
            logits, s = net(inputs)
            t2 = time.time()
            #print("time 2 = {}, batch = {}".format(t2-t1, i))

            loss = loss_fn(logits, labels)# + aux_loss
            loss.backward()

            optimizer.step()

            if verbose >= 2:
                if i % 10 == 0:
                    print("max values: ", torch.max(logits, dim=1).values.cpu().detach().numpy())
                    print("batch loss = {}".format(loss.item()))
                    print("est. lambd = ", net.spectrogram_layer.lambd.item())

            running_loss += loss.item()
            running_energy += np.sum(s.cpu().detach().numpy())
            count += 1

            t_tot2 = time.time()
            #print("time total = {}, batch = {}".format(t_tot2-t_tot1, i))

        # step scheduler
        scheduler.step()
            
        train_loss = running_loss / count
        train_energy = running_energy / count

        if verbose >= 1:
            print("epoch {}, train loss = {}".format(epoch, running_loss / count))
            print("est. lambd = ", net.spectrogram_layer.lambd.item())

        running_loss = 0.0
        count = 0
        running_acc = 0.0
        
        net.eval()
        for data in validloader:
            inputs, labels = data

            if one_hot:
                # TODO: this won't work in general
                labels = torch.nn.functional.one_hot(labels, n_classes).float()

            inputs, labels = inputs.to(device), labels.to(device)

            outputs, spectrograms = net(inputs)
            loss = loss_fn(outputs, labels)

            predictions = torch.argmax(outputs, axis=1)

            if one_hot:
                labels = torch.argmax(labels, axis=1)

            accuracy = torch.mean((predictions == labels).float())
            running_acc += accuracy.item()

            running_loss += loss.item()
            count += 1

        valid_loss = running_loss / count
        valid_acc = running_acc / count


        # save epoch model
        #with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #    path = os.path.join(checkpoint_dir, "checkpoint")
        #    torch.save((net.state_dict(), optimizer.state_dict()), path)


        if valid_loss < best_valid_loss: # < best_valid_loss:

            # save best model
            with tune.checkpoint_dir(0) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "best_model")
                torch.save((net.state_dict(), optimizer.state_dict()), path)

            best_valid_acc = valid_acc
            best_valid_loss = valid_loss
            best_lambd_est = net.spectrogram_layer.lambd.item()
            patience_count = 0
            if verbose >= 1:
                print("new best valid acc  = {}, patience_count = {}".format(best_valid_acc, patience_count))
        else:
            patience_count += 1

        # report results
        tune.report(loss=train_loss, lambd_est=net.spectrogram_layer.lambd.item(), valid_loss=valid_loss, valid_acc=valid_acc, best_valid_acc=best_valid_acc, best_valid_loss=best_valid_loss, energy=train_energy, best_lambd_est=best_lambd_est)

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
    history["est_lambd"] = net.spectrogram_layer.lambd.item()
    
    return net, history
