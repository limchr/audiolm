
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F

import math


import audiolm_pytorch.data as data
from dataclasses import dataclass
from audiolm_pytorch.condition_cnn import ConditionCNN

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# things to look at:

# other regulation, model complexities
# conditioning z to be between -1 and 1 (penelizing large vectors as second optimization term)
# model exporting
# try convolutional layers for pre stage
# model importing for transformer



device = 'cuda'
batch_size = 256
num_passes = 2500

lr = 12e-4 # learning rate
wd = 0.1 # weight decay
betas = (0.9, 0.95) # adam betas

loss_fn = nn.CrossEntropyLoss()

dr = 0.7 # dropout rate


pre_layers = [32*128, 8*128, 2*128, 64]
post_layers = [2, 5]


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

def det_loss(cnn,dl):
    losses = []
    cnn.eval()
    for d in dl:
        x = d[0]
        x = x[:,:32:]
        x_hat = cnn.forward(x)
        loss = loss_fn(x_hat, d[2])
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
# data initialization
#

ds = data.EncodecSoundDataset(folder='/home/chris/data/audio_samples/ds_extracted', seed=seed)
dsb = data.BufferedDataset(ds, '/home/chris/data/buffered_ds_extracted.pkl', False)
train_size = math.floor(0.7 * len(dsb))
val_size = math.floor(0.3 * len(dsb))
test_size = len(dsb) - train_size - val_size

ds_train, ds_test, ds_val = random_split(dsb, [train_size, test_size, val_size])


# some sanity debug checks
# ds_train = torch.utils.data.Subset(ds_train, range(0,10))
# ds_val = ds_train

from torch.utils.data import WeightedRandomSampler
yp = []
for d in ds_train:
    yp.append(d[2])
y = torch.stack(yp).cpu().numpy()

cys = [len(np.where(y[:,yy] > 0.)[0]) for yy in range(y.shape[1])]
totalx = np.sum(cys)
cys = cys / totalx
cys = 1/cys

cw = np.zeros(len(y))
for i in range(len(y)):
    if y[i].sum() > 0:
        cw[i] = cys[y[i].argmax()] # not quite correct because there could be multiple classes

sampler = WeightedRandomSampler(cw, len(cw), replacement=True)

dl = DataLoader(ds_train, batch_size=batch_size, sampler=sampler) # shuffle=True
dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True)



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


    for d in dl:
        sax = d[0]
        # sax[:] = 0.0 # sanity check
        sax = sax[:,:32,:]
        x_hat = cnn.forward(sax)
        loss = loss_fn(x_hat, d[2])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    if i % 10 == 0:
        train_loss = det_loss(cnn,dl)
        val_loss = det_loss(cnn,dl_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss[1]:
            best_val_loss = (i, val_loss)
            # save model
            torch.save(cnn, 'results/cnn_model.pt')

        print('pass: %d \t train loss: %.4f \t val loss: %.4f \t best val loss: %.4f \t at pass: %d' 
            % (i, train_loss, val_loss, best_val_loss[1], best_val_loss[0]))
            

# save model
torch.save(cnn, 'results/cnn_model_last.pt')




#
# testing plots
#

cnn.eval()

# save losses plot

train_losses = np.array(train_losses)
val_losses = np.array(val_losses)   
plt.figure()
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.savefig('results/cnnlosses.png')
plt.show()


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

for i in range(0,5):
    cl = [True if cc[i] == 1.0 else False for cc in y ]
    samples_of_class = x[cl]
    if len(samples_of_class) > 0:
        cl_name = list(ds.class_groups.keys())[i]
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

for i in range(0,5):
    cl = [True if cc[i] == 1.0 else False for cc in y ]
    samples_of_class = x[cl]
    if len(samples_of_class) > 0:
        cl_name = list(ds.class_groups.keys())[i]
        samples_of_class = samples_of_class[:,:32,:]
        x_bottleneck  = cnn.forward(samples_of_class,bn_only=True)    
        bn = x_bottleneck.cpu().detach().numpy()
        plt.scatter(bn[:,0], bn[:,1], label=cl_name)


plt.legend()
plt.show()
plt.savefig('results/cnn_cond_plot_val.png')





pass