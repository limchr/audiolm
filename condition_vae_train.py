#
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# for enabling deterministic behaviour of this script
#

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import math


import audiolm_pytorch.data as data
from dataclasses import dataclass
from audiolm_pytorch.condition_vae import ConditionVAE, ConditionConvVAE
from audiolm_pytorch.data import get_class_weighted_sampler, get_audio_dataset

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from experiment_config import ds_folders, ds_buffer, ckpt_vae


# we want absolute deterministic behaviour here for reproducibility a good looking 2d embedding
seed = 1237
data_seed = 1234

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True)

device = 'cuda'

num_passes = 150
batch_size = 512
lr = 3e-4 # learning rate
wd = 0.1 # weight decay
betas = (0.9, 0.95) # adam betas
dr = 0.3 # dropout rate

# crop the time dimension to this length (150 -> input_crop)
input_crop = 150

# simple vae without convolutional layers
# layers = [128, 64, 32, 16]

# vae with convolutional layers
channels = [128,128,64,32]
linears = [128, 64, 32, 2]


# get the audio dataset
# ds_folders = ['/home/chris/data/audio_samples/ds_min/'] # for debugging
# ds_buffer = '/home/chris/data/audio_samples/ds_min.pkl' # for debugging
dsb, ds_train, ds_val, dl_train, dl_val = get_audio_dataset(audiofile_paths= ds_folders,
                                                                dump_path= ds_buffer,
                                                                build_dump_from_scratch=False,
                                                                only_labeled_samples=True,
                                                                test_size=0.1,
                                                                equalize_class_distribution=True,
                                                                equalize_train_data_loader_distribution=True,
                                                                batch_size=batch_size,
                                                                seed=data_seed)
# check for train test split random correctness
print(ds_train[123][3], ds_train[245][3], ds_val[456][3], ds_val[125][3])



#
# just for visualization during training
#
xp = []
yp = []
for d in ds_train:
    xp.append(d[0])
    yp.append(d[3])
dsx = torch.stack(xp)    
dsy = torch.stack(yp)


vae = ConditionConvVAE(channels, linears, input_crop).to(device)


optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=wd, betas=betas)
loss_fn = nn.MSELoss(reduction='sum')

def loss_fn2(x, x_hat, mean, var, beta = 1.):
    normalization = x.shape[0]*x.shape[1]*x.shape[2]
    reproduction_loss = None
    if False:
        # weighting function for increasing the weight of the beginning of the sample
        # idea would be envolope loss (adsr)
        weight = torch.zeros_like(x)
        for i in range(x.shape[1]):
            weight[:,i,:] = 1-(i/x.shape[1])**2
        weight = 1/weight.mean() * weight
        reproduction_loss = loss_fn(x_hat*weight, x*weight) / normalization
    else:
        reproduction_loss = loss_fn(x_hat, x) / normalization

    kl_loss = ( beta * -0.5 * torch.sum(1 + var - mean**2 - var.exp()) ) / normalization
    l = torch.linalg.vector_norm(mean,ord=2,dim=1)


    scalar = torch.FloatTensor([0.0]).to(device)
    kl_loss += torch.max((l-1.0),scalar.expand_as(l)).mean() * 0.1
    # kl_loss += torch.mean(torch.abs(l-0.8)) * 0.001
    # loss = (l*0.35)**32

    li = reproduction_loss.item()
    if li > 100:
        print('#################### %f' % li)
    return reproduction_loss, kl_loss


def det_loss(va,ds):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False) # same class distribution as dataset
    losses = []
    va.eval()
    for dx, _, _, _ in dl:
        dx = dx.to(device)
        x, x_hat, mean, var = va.forward(dx)
        rec_loss, kld_loss = loss_fn2(x, x_hat, mean, var)
        losses.append([rec_loss.item(), kld_loss.item()])
    va.train()
    losses = np.array(losses).mean(axis=0)
    return losses

def print_param_stats(model):
    for pn, p in model.named_parameters(): 
        if hasattr(p,'grad') and p.grad is not None:
            print('param: %s \t mean: %.6f \t std: %.6f \t\t grad mean: %.6f \t grad std: %.6f' % (pn, p.mean().item(), p.std().item(), p.grad.mean().item(), p.grad.std().item()))
        else:
            print('param: %s \t mean: %.6f \t std: %.6f' % (pn, p.mean().item(), p.std().item()))


vae.train()

train_losses = []
val_losses = []

for i in range(num_passes):

    for dx, _, _, _ in dl_train:
        dx = dx.to(device)
        x, x_hat, mean, var = vae.forward(dx)
        rec_loss, kld_loss = loss_fn2(x, x_hat, mean, var)
        loss = rec_loss + kld_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if i == int(num_passes * 0.6) or i == int(num_passes * 0.8):
        # change learning rate
        lr = lr * 0.2
        optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=wd, betas=betas)
        print('############################# new lr: %.8f' % lr)
        
        # print parameter stats
        print_param_stats(vae)


    if i % 10 == 0:
        train_loss_rec, train_loss_kld = det_loss(vae,ds_train)
        val_loss_rec, val_loss_kld = det_loss(vae,ds_val)
        train_losses.append([train_loss_rec, train_loss_kld])
        val_losses.append([val_loss_rec, val_loss_kld])

        print('pass: %d \t train loss: %.4f \t train rec loss: %.4f \t train kld loss: %.4f \t val loss: %.4f \t val rec loss: %.4f \t val kld loss: %.4f' 
            % (i,train_loss_rec+train_loss_kld, train_loss_rec, train_loss_kld, val_loss_rec+val_loss_kld, val_loss_rec, val_loss_kld))
        
        # print('saving model of epoch %d' % i)
        torch.save(vae, ckpt_vae)



        vae.eval()


        # save losses plot
        tp = np.array(train_losses)
        vp = np.array(val_losses)   
        plt.close(0)
        plt.figure(0)
        plt.plot(tp.sum(axis=1), label='train')
        plt.plot(tp[:,0], label='train rec')
        plt.plot(tp[:,1], label='train kld')
        plt.plot(vp.sum(axis=1), label='val')
        plt.plot(vp[:,0], label='val rec')
        plt.plot(vp[:,1], label='val kld')
        # plt.ylim(0,0.2)
        plt.legend()
        plt.savefig('results/vae_losses.png')




        plt.close(0)
        plt.figure(0)

        classes = ds_train.ds.classes
        for i in range(len(classes)):
            samples_of_class = dsx[dsy==i]
            if len(samples_of_class) > 0:
                outp_mean, outp_var  = vae.forward(samples_of_class.to(device),encoder_only=True)    
                px = outp_mean.cpu().detach().numpy()
                plt.scatter(px[:,0], px[:,1], label=classes[i], s=0.1, alpha=0.7)

        plt.xlim(-1,1)
        plt.ylim(-1,1)

        plt.legend()
        plt.savefig('results/vae_latent_distribution.png')



        for i in range(len(classes)):
            samples_of_class = dsx[dsy==i]
            if len(samples_of_class) > 0:
                plt.close(0)
                plt.figure(0)
                f, axarr = plt.subplots(1,2) 

                cl_name = classes[i]
                x, x_hat, x_mean, x_log_var = vae.forward(samples_of_class.to(device))
                xorig = x[0].cpu().detach().numpy()
                xrec = x_hat[0].cpu().detach().numpy()
                axarr[0].imshow(xorig)
                axarr[1].imshow(xrec)
                
                axarr[0].set_title('original')
                axarr[1].set_title('reconstructed')
                # for ax in axarr:
                #     ax.set_xticks([])
                #     ax.set_yticks([])
                plt.suptitle(cl_name)
                plt.savefig('results/vae_reconstruction_%s.png' % cl_name)
                

        vae.train()



pass