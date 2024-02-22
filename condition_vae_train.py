#
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# for enabling deterministic behaviour of this script
#
# python3 condition_vae_train.py; python3 transformer_train_torch.py ; python3 transformer_generate.py ; python3 evaluation_table.py

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
import gc


# we want absolute deterministic behaviour here for reproducibility a good looking 2d embedding
seed = 123132
data_seed = 1234

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True)

device = 'cuda'

num_passes = 10000
batch_size = 1024
lr = 6e-4 # learning rate
wd = 0.05 # weight decay
betas = (0.9, 0.95) # adam betas

weighted_reproduction = True

# crop the time dimension to this length (150 -> input_crop)
input_crop = 150

# simple vae without convolutional layers
# layers = [128, 64, 32, 16]

# vae with convolutional layers
channels = [128,256,128,64]
linears = [256, 128, 64, 32, 2]


# get the audio dataset
# ds_folders = ['/home/chris/data/audio_samples/single_packs/kb6'] # for debugging
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
print(ds_train[54][3], ds_train[23][3], ds_val[87][3], ds_val[43][3])



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

def loss_fn2(x, x_hat, mean, var, iter):
    
    b, t, f = x.shape
    
    #
    # reproduction loss
    #
    reproduction_loss = None
    rep_norm = b*t*f
    if weighted_reproduction:
        # weighting function for increasing the weight of the beginning of the sample
        # idea would be envolope loss (adsr)
        weight = torch.zeros_like(x)
        for i in range(x.shape[1]):
            weight[:,i,:] = 1-(i/x.shape[1])**2
        weight = 1/weight.mean() * weight
        reproduction_loss = loss_fn(x_hat*weight, x*weight) / rep_norm
    else:
        reproduction_loss = loss_fn(x_hat, x) / rep_norm

    #
    # regularization loss terms
    #
    
    # neighboring loss
    min_dist = 2 / math.sqrt(batch_size) # distance between samples that is desired
    eps = 1e-3
    
    dists = torch.cdist(mean,mean, p=2)
    dists = torch.where(dists < min_dist, dists, torch.ones_like(dists)*1000000)
    repulsion_effect = 1.0 / (dists + eps)
    
    mask = torch.eye(dists.size(0), device=device).bool()
    repulsion_effect = repulsion_effect.masked_fill_(mask, 0)
    neighbor_loss = repulsion_effect.sum() / (b * (1/eps))


    # spatial regularization loss
    l_orig = torch.linalg.vector_norm(mean,ord=2,dim=1)
    zero_tensor = torch.FloatTensor([0.0]).to(device)
    spatial_loss = torch.max((l_orig-1.0),zero_tensor.expand_as(l_orig)).mean()
    
    # # debug print outs
    # print('###')
    # print(reproduction_loss)
    # print(neighbor_loss)
    # print(spatial_loss)
    # print('')
    
    warmup_ratio = 0.1
    training_progress = (iter/num_passes)
    reg_beta = 0.0 if training_progress < warmup_ratio else training_progress   

    return reproduction_loss, reg_beta * (0.5 * neighbor_loss + spatial_loss)


def det_loss(va,ds):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False) # same class distribution as dataset
    losses = []
    for dx, _, _, _ in dl:
        dx = dx.to(device)
        x_hat, mean, var = va.forward(dx)
        rec_loss, kld_loss = loss_fn2(dx, x_hat, mean, var,1)
        losses.append([rec_loss.item(), kld_loss.item()])
    losses = np.array(losses).mean(axis=0)
    return losses

def print_param_stats(model):
    for pn, p in model.named_parameters(): 
        if hasattr(p,'grad') and p.grad is not None:
            print('param: %s \t mean: %.6f \t std: %.6f \t\t grad mean: %.6f \t grad std: %.6f' % (pn, p.mean().item(), p.std().item(), p.grad.mean().item(), p.grad.std().item()))
        else:
            print('param: %s \t mean: %.6f \t std: %.6f' % (pn, p.mean().item(), p.std().item()))




train_losses = []
val_losses = []

for i in range(num_passes):
    vae.train()
    for dx, _, _, _ in dl_train:
        optimizer.zero_grad(set_to_none=True)
        dx = dx.to(device)
        x_hat, mean, var = vae.forward(dx)
        rec_loss, kld_loss = loss_fn2(dx, x_hat, mean, var, i)
        loss = rec_loss + kld_loss
        loss.backward()
        optimizer.step()

    if i == int(num_passes * 0.7) or i == int(num_passes * 0.8) or i == int(num_passes * 0.9):
        # change learning rate
        lr = lr * 0.1
        optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=wd, betas=betas)
        print('############################# new lr: %.8f' % lr)
        
        # print parameter stats
        print_param_stats(vae)


    if i>0 and i % 10 == 0:

        vae.eval()
        
        train_loss_rec, train_loss_kld = det_loss(vae,ds_train)
        val_loss_rec, val_loss_kld = det_loss(vae,ds_val)
        train_losses.append([train_loss_rec, train_loss_kld])
        val_losses.append([val_loss_rec, val_loss_kld])

        print('pass: %d \t train loss: %.4f \t train rec loss: %.4f \t train kld loss: %.4f \t val loss: %.4f \t val rec loss: %.4f \t val kld loss: %.4f' 
            % (i,train_loss_rec+train_loss_kld, train_loss_rec, train_loss_kld, val_loss_rec+val_loss_kld, val_loss_rec, val_loss_kld))
        
        # print('saving model of epoch %d' % i)
        torch.save(vae, ckpt_vae)

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
            samples_of_class = dsx[dsy==i][:batch_size*2]
            if len(samples_of_class) > 0:
                outp_mean, outp_var  = vae.forward(samples_of_class.to(device),encoder_only=True)    
                px = outp_mean.cpu().detach().numpy()
                plt.scatter(px[:,0], px[:,1], label=classes[i], s=0.1, alpha=0.7)

        plt.xlim(-1,1)
        plt.ylim(-1,1)

        plt.legend()
        plt.savefig('results/vae_latent_distribution.png')

        if True:
            for i in range(len(classes)):
                samples_of_class = dsx[dsy==i][:batch_size]
                if len(samples_of_class) > 0:
                    plt.close(0)
                    plt.figure(0)
                    f, axarr = plt.subplots(1,2) 

                    cl_name = classes[i]
                    soc = samples_of_class.to(device)
                    x_hat, x_mean, x_log_var = vae.forward(soc)
                    xorig = soc[0].cpu().detach().numpy()
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
                    
        gc.collect()
        torch.cuda.empty_cache()



pass