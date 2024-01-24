
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import math


import audiolm_pytorch.data as data
from dataclasses import dataclass
from audiolm_pytorch.condition_cnn import ConditionCNN
from audiolm_pytorch.data import get_class_weighted_sampler, get_audio_dataset

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from experiment_config import ds_folders, ds_buffer


seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# 24.01
# pass: 4998       train loss: 0.1671      val loss: 0.4199        best val loss: 0.4148   at pass: 4986

device = 'cuda'


batch_size = 512
num_passes = 5000
lr = 6e-4 # learning rate
wd = 0.1 # weight decay
betas = (0.9, 0.95) # adam betas
dr = 0.65 # dropout rate

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



pre_layers = [32*128, 8*128, 128, 64]
post_layers = [2, 4, dsb.dataset.dataset.classes.__len__()]


#
# data initialization
#




# basically for limiting the 'area' of the bottleneck
def loss_fn_bn(z):
    # https://www.wolframalpha.com/input?i=plot+%28x*0.4%29%5E16+from+x+%3D+%5B-3%2C3%5D+from+y+%3D+%5B-3%2C3%5D
    l = torch.linalg.vector_norm(z,ord=2,dim=1)
    loss = (l*0.35)**32
    return loss.mean()

loss_fn_classification = nn.CrossEntropyLoss()
loss_fn_bottleneck_regularization = loss_fn_bn

def loss_fn(x_hat, y, bn):
    return loss_fn_classification(x_hat, y) + loss_fn_bottleneck_regularization(bn)



# 
# model initialization
# 

cnn = ConditionCNN(pre_layers=pre_layers,post_layers=post_layers, dropout_rate=dr).to(device)

#
# utility functions
#

def get_optimizer():
    global cnn
    global lr
    global wd
    global betas
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=lr, weight_decay=wd, betas=betas)
    return optimizer
optimizer = get_optimizer()

@torch.no_grad()
def det_loss(cnn,ds):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False) # get new dataloader because we want random sampling here!
    losses = []
    cnn.eval()
    for d in dl:
        x = d[0]
        x = x[:,:32:]
        x_hat,z = cnn.forward(x)
        loss = loss_fn(x_hat, d[2],z)
        losses.append(loss.item())
    cnn.train()
    losses = np.array(losses).mean()
    return losses

def print_param_stats(model):
    for pn, p in model.named_parameters(): 
        if hasattr(p,'grad') and p.grad is not None:
            print('param: %s \t mean: %.6f \t std: %.6f \t\t grad mean: %.6f \t grad std: %.6f' % (pn, p.mean().item(), p.std().item(), p.grad.mean().item(), p.grad.std().item()))
        else:
            print('param: %s \t mean: %.6f \t std: %.6f' % (pn, p.mean().item(), p.std().item()))




#
# training loop
#

cnn.train()

train_losses = []
val_losses = []
best_val_loss = (-1, 1000000.0) # (epoch, loss)

for i in range(num_passes):
    if i == int(num_passes * 0.7) or i == int(num_passes * 0.85):
        # change learning rate
        lr = lr * 0.2
        optimizer = get_optimizer()
        print('############################# new lr: %.8f' % lr)
        
        # print parameter stats
        print_param_stats(cnn)


    for d in dl_train:
        sax = d[0]
        # sax[:] = 0.0 # sanity check
        sax = sax[:,:32,:]
        x_hat, bn = cnn.forward(sax)
        loss = loss_fn(x_hat, d[2], bn)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    if i % 2 == 0:
        train_loss = det_loss(cnn,ds_train)
        val_loss = det_loss(cnn,ds_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss[1]:
            best_val_loss = (i, val_loss)
            # save model
            torch.save(cnn, 'results/cnn_model.pt')

        print('pass: %d \t train loss: %.4f \t val loss: %.4f \t best val loss: %.4f \t at pass: %d' 
            % (i, train_loss, val_loss, best_val_loss[1], best_val_loss[0]))
        
        
        # save losses plot
        # train_losses = np.array(train_losses)
        # val_losses = np.array(val_losses)   
        plt.close(0)
        plt.figure(0)
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend()
        plt.savefig('results/cnnlosses.png')
        plt.show()

                    

# save model
torch.save(cnn, 'results/cnn_model_last.pt')




#
# testing plots
#

cnn.eval()



# save conditional plot for training set

xp = []
yp = []

for d in ds_train:
    xp.append(d[0])
    yp.append(d[2])

x = torch.stack(xp)    
y = torch.stack(yp)

plt.figure()
plt.ion()

for i in range(0,dsb.dataset.dataset.classes.__len__()):
    cl = [True if cc[i] == 1.0 else False for cc in y ]
    samples_of_class = x[cl]
    if len(samples_of_class) > 0:
        cl_name = list(dsb.dataset.dataset.class_groups.keys())[i]
        samples_of_class = samples_of_class[:,:32,:]
        x_bottleneck  = cnn.forward(samples_of_class,bn_only=True)    
        bn = x_bottleneck.cpu().detach().numpy()
        plt.scatter(bn[:,0], bn[:,1], label=cl_name)


plt.legend()
plt.show()
plt.savefig('results/cnn_cond_plot_train.png')


# save conditional plot for validation set

xp = []
yp = []

for d in ds_val:
    xp.append(d[0])
    yp.append(d[2])

x = torch.stack(xp)    
y = torch.stack(yp)

plt.figure()
plt.ion()

for i in range(0,dsb.dataset.dataset.classes.__len__()):
    cl = [True if cc[i] == 1.0 else False for cc in y ]
    samples_of_class = x[cl]
    if len(samples_of_class) > 0:
        cl_name = list(dsb.dataset.dataset.class_groups.keys())[i]
        samples_of_class = samples_of_class[:,:32,:]
        x_bottleneck  = cnn.forward(samples_of_class,bn_only=True)    
        bn = x_bottleneck.cpu().detach().numpy()
        plt.scatter(bn[:,0], bn[:,1], label=cl_name)


plt.legend()
plt.show()
plt.savefig('results/cnn_cond_plot_val.png')





pass