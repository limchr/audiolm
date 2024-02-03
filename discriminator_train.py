import torch
import torch.nn.functional as F
import torchaudio

import audiolm_pytorch.data as data
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass

import audiolm_pytorch.gpt as gpt

from audiolm_pytorch import EncodecWrapper

import random
import numpy as np
import gc


import matplotlib.pyplot as plt
import matplotlib


from audiolm_pytorch.data import get_audio_dataset
from experiment_config import ds_folders, ds_buffer, ckpt_vae, ckpt_transformer, ckpt_discriminator
from audiolm_pytorch.discriminator import Discriminator


device = 'cuda'
num_passes = 100 # num passes through the dataset

learning_rate = 3e-4 # max learning rate
weight_decay = 0.05
beta1 = 0.9
beta2 = 0.95
batch_size = 1024

loss_fn = F.binary_cross_entropy

input_crop = 64
channels = [128,256,512,256]
linears = [256, 128, 64, 5]
dropout = 0.15

seed = 1234


torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)



if __name__ == '__main__':

    # get the audio dataset
    dsb, ds_train, ds_val, dl_train, dl_val = get_audio_dataset(audiofile_paths= ds_folders,
                                                                    dump_path= ds_buffer,
                                                                    build_dump_from_scratch=False,
                                                                    only_labeled_samples=True,
                                                                    test_size=0.2,
                                                                    equalize_class_distribution=True,
                                                                    equalize_train_data_loader_distribution=True,
                                                                    batch_size=batch_size,
                                                                    seed=seed)

    # calculate loss of model for a given dataset (executed during training)
    @torch.no_grad()
    def det_loss_testing(ds, model):
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False) # get new dataloader because we want random sampling here!
        model.eval()
        losses = []
        accuracies = []
        for dx, _, dy, _ in dl:
            dx = dx.to(device)
            dy = dy.to(device)
            logits = model.forward(dx)
            loss = loss_fn(logits, dy)
            losses.append(loss.cpu().detach().item())    
            accuracy = (torch.max(dy,dim=1)[1] == torch.max(logits,dim=1)[1]).float().mean().item()
            accuracies.append(accuracy)
        model.train()
        return np.mean(losses), np.mean(accuracies)


    def train(is_parameter_search):
        global learning_rate
        
        model = Discriminator(channels, linears, input_crop, dropout)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
        
        # do not change
        best_val_loss = 1e9
        best_val_loss_iter = 0
        best_train_loss = 1e9
        best_train_loss_iter = 0

        train_losses = []
        val_losses = []

        model.train()
        for i in range(num_passes):
            for dx, _, dy, _ in dl_train: # training is unsupervised so we don't need the labels (only shifted x)
                dx = dx.to(device)
                dy = dy.to(device)
                
                logits = model.forward(dx)
                
                loss = loss_fn(logits, dy)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if i == int(num_passes*0.8) or i == int(num_passes*0.6) or i == int(num_passes*0.5):
                learning_rate = learning_rate * 0.5
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
                if not is_parameter_search:
                    print('changed learning rate to %.3e at pass %d' % (learning_rate, i))

            # plot training stats
            if i % 3 == 0:
                train_loss, train_acc = det_loss_testing(ds_train, model)
                val_loss, val_acc = det_loss_testing(ds_val, model)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                if not is_parameter_search:
                    print('%d/%d\ttrain acc: %.5f\tval acc: %.5f\ttrain loss: %.5f\tval loss: %.5f \tbest train loss: %.5f \tbest val loss: %.5f (epoch %i)' % (i, num_passes, train_acc, val_acc, train_loss, val_loss, best_train_loss, best_val_loss, best_val_loss_iter))
                if i > 0 and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_iter = i
                    if not is_parameter_search:
                        print('saving model to %s with val loss %.5f' % (ckpt_transformer, best_val_loss))
                        torch.save(model, ckpt_discriminator)
                if i > 0 and train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_train_loss_iter = i

                    
                if not is_parameter_search:    
                    # save losses plot
                    plt.close(0)
                    plt.figure(0)
                    plt.plot(train_losses, label='train')
                    plt.plot(val_losses, label='val')
                    plt.legend()
                    plt.savefig('results/discriminator_losses.png')
                    # plt.show()
                    
                # early stopping
                if i > best_val_loss_iter + 50:
                    print('early stopping at pass %d' % i)
                    break
                
        del model
        gc.collect()

        return best_val_loss, best_val_loss_iter, best_train_loss, best_train_loss_iter
    
    
    # doing a random search
    
    train(is_parameter_search=False)