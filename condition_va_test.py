
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F

import math


import audiolm_pytorch.data as data
from dataclasses import dataclass
from audiolm_pytorch.condition_va import ConditionVA

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib


device = 'cuda'
seed = 1234
batch_size = 256
num_passes = 100_000

lr = 10e-4

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


layers = [128, 64]
hidden_dim = 32
latent_dim = 2

va = ConditionVA(hidden_dim, latent_dim, layers).to(device)

ds = data.EncodecSoundDataset(folder='/home/chris/data/audio_samples/ds_extracted', seed=seed)
dsb = data.BufferedDataset(ds, '/home/chris/data/buffered_ds_extracted.pkl', False)
train_size = math.floor(0.9 * len(dsb))
val_size = math.floor(0.1 * len(dsb))
test_size = len(dsb) - train_size - val_size


ds_train, ds_test, ds_val = random_split(dsb, [train_size, test_size, val_size])


# some sanity debug checks
# ds_train = torch.utils.data.Subset(ds_train, range(0,10))
# ds_val = ds_train



dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True)



BCE_loss = nn.BCELoss()



optimizer = torch.optim.AdamW(va.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
loss_fn = nn.MSELoss(reduction='sum')

def loss_fn2(x, x_hat, mean, var, beta = 1.):
    normalization = x.shape[0]*x.shape[1]*x.shape[2]

    reproduction_loss = loss_fn(x_hat, x) / normalization
    kl_loss = ( beta * -0.5 * torch.sum(1 + var - mean**2 - var.exp()) ) / normalization

    li = reproduction_loss.item()
    if li > 100:
        print('#################### %f' % li)
    return reproduction_loss, kl_loss


def det_loss(va,dl):
    losses = []
    va.eval()
    for d in dl:
        x = d[0]
        x = x[:,:30,:]
        x_hat, mean, var = va.forward(x)
        rec_loss, kld_loss = loss_fn2(x, x_hat, mean, var)
        losses.append([rec_loss.item(), kld_loss.item()])
    va.train()
    losses = np.array(losses).mean(axis=0)
    return losses

def print_param_stats(model):
    for pn, p in va.named_parameters(): 
        if hasattr(p,'grad') and p.grad is not None:
            print('param: %s \t mean: %.6f \t std: %.6f \t\t grad mean: %.6f \t grad std: %.6f' % (pn, p.mean().item(), p.std().item(), p.grad.mean().item(), p.grad.std().item()))
        else:
            print('param: %s \t mean: %.6f \t std: %.6f' % (pn, p.mean().item(), p.std().item()))


va.train()

train_losses = []
val_losses = []

for i in range(num_passes):
    if i == int(num_passes * 0.8) or i == int(num_passes * 0.9):
        # change learning rate
        lr = lr * 0.2
        optimizer = torch.optim.AdamW(va.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
        print('############################# new lr: %.8f' % lr)
        
        # print parameter stats
        print_param_stats(va)


    for d in dl:
        sax = d[0]
        # sax[:] = 0.0 # sanity check
        sax = sax[:,:30,:]
        sax[:,29,123:] = d[2] # quick and dirty condition
        x_hat, mean, var = va.forward(sax)
        rec_loss, kld_loss = loss_fn2(sax, x_hat, mean, var)
        loss = rec_loss + kld_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    if i % 10 == 0:
        train_loss_rec, train_loss_kld = det_loss(va,dl)
        val_loss_rec, val_loss_kld = det_loss(va,dl_val)
        train_losses.append([train_loss_rec, train_loss_kld])
        val_losses.append([val_loss_rec, val_loss_kld])

        print('pass: %d \t train loss: %.4f \t train rec loss: %.4f \t train kld loss: %.4f \t val loss: %.4f \t val rec loss: %.4f \t val kld loss: %.4f' 
            % (i,train_loss_rec+train_loss_kld, train_loss_rec, train_loss_kld, val_loss_rec+val_loss_kld, val_loss_rec, val_loss_kld))
            

va.eval()

# save losses plot
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)   
plt.figure()
plt.plot(train_losses.sum(axis=1), label='train')
plt.plot(train_losses[:,0], label='train rec')
plt.plot(train_losses[:,1], label='train kld')
plt.plot(val_losses.sum(axis=1), label='val')
plt.plot(val_losses[:,0], label='val rec')
plt.plot(val_losses[:,1], label='val kld')
plt.ylim(0,0.2)
plt.legend()
plt.savefig('results/valosses.png')
plt.show()





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
        samples_of_class = samples_of_class[:,:30,:]
        outp_mean, outp_var  = va.encoder(samples_of_class)    
        px = outp_mean.cpu().detach().numpy()
        plt.scatter(px[:,0], px[:,1], label=cl_name)

plt.xlim(-1,1)
plt.ylim(-1,1)


plt.legend()
plt.show()
plt.savefig('results/vacond_plot.png')


plt.ioff()


for i in range(0,5):
    cl = [True if cc[i] == 1.0 else False for cc in y ]
    samples_of_class = x[cl]
    if len(samples_of_class) > 0:
        plt.figure()
        f, axarr = plt.subplots(1,2) 

        cl_name = list(ds.class_groups.keys())[i]
        samples_of_class = samples_of_class[:,:30,:]
        x_hat, x_mean, x_log_var = va.forward(samples_of_class)
        xorig = samples_of_class[0].cpu().detach().numpy()
        xrec = x_hat[0].cpu().detach().numpy()
        axarr[0].imshow(xorig)
        axarr[1].imshow(xrec)
        
        axarr[0].set_title('original')
        axarr[1].set_title('reconstructed')
        # for ax in axarr:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        plt.suptitle(cl_name)
        plt.show()
        plt.savefig('results/va_img_reconstr_%s.png' % cl_name)



inpw = va.encoder.fc1.weight.mean(axis=0).cpu().detach().numpy()
plt.plot(inpw)
plt.show()

pass