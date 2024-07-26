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


import audiolm_pytorch.pitched_data as pdata
from dataclasses import dataclass











class ConditionConvVAE(nn.Module):
    def __init__(self, enc_linears, enc_channels, dec_linears, dec_channels, cls_linears, cls_channels, input_crop, kernel_size=5, dilation=1, padding=None, stride=2, output_padding=1):
        super(ConditionConvVAE, self).__init__()
        if padding is None:
            padding = kernel_size//2
                 
                 
        # calculate all dimensions here   
        conv_sizes = [input_crop]
        for _ in range(len(enc_channels)-1):
            cs = ( (conv_sizes[-1]+2*padding-dilation*(kernel_size-1)-1)/stride ) + 1
            conv_sizes.append(math.floor(cs))
            # print(cs)
        intermediate_output_size = conv_sizes[-1] * enc_channels[-1]
        deconv_sizes = [conv_sizes[-1]]
        for _ in range(len(enc_channels)-1):
            dcs = (deconv_sizes[-1]-1) * stride - 2*padding + dilation * (kernel_size-1) + output_padding + 1
            deconv_sizes.append(dcs)


        # calculate all dimensions here   
        cls_conv_sizes = [input_crop]
        for _ in range(len(cls_channels)-1):
            cs = ( (cls_conv_sizes[-1]+2*padding-dilation*(kernel_size-1)-1)/stride ) + 1
            cls_conv_sizes.append(math.floor(cs))
        cls_intermediate_output_size = cls_conv_sizes[-1] * cls_channels[-1]




        
        enc_linears = [intermediate_output_size,] + enc_linears
        dec_linears = dec_linears + [intermediate_output_size,]
        cls_linears = [cls_intermediate_output_size,] + cls_linears
        
        
        # encoder
        for i in range(len(enc_channels)-1):
            setattr(self, 'enc_conv{}'.format(i), nn.Conv1d(
                enc_channels[i], enc_channels[i+1], kernel_size=kernel_size, stride=stride, padding=padding
            ))
            setattr(self, 'enc_conv_norm{}'.format(i), nn.LayerNorm([enc_channels[i+1], conv_sizes[i+1]]))
        
        for i in range(len(enc_linears)-2):
            setattr(self, 'enc_lin{}'.format(i), nn.Linear(
                enc_linears[i], enc_linears[i+1],
            ))
            setattr(self, 'enc_lin_norm{}'.format(i), nn.LayerNorm(enc_linears[i+1]))

        self.mean = nn.Linear(enc_linears[-2], enc_linears[-1])
        self.var = nn.Linear(enc_linears[-2], enc_linears[-1])


        # classifier
        for i in range(len(cls_channels)-1):
            setattr(self, 'cls_conv{}'.format(i), nn.Conv1d(
                cls_channels[i], cls_channels[i+1], kernel_size=kernel_size, stride=stride, padding=padding
            ))
            setattr(self, 'cls_conv_norm{}'.format(i), nn.LayerNorm([cls_channels[i+1], conv_sizes[i+1]]))
        
        for i in range(len(cls_linears)-1):
            setattr(self, 'cls_lin{}'.format(i), nn.Linear(
                cls_linears[i], cls_linears[i+1],
            ))
            setattr(self, 'cls_lin_norm{}'.format(i), nn.LayerNorm(cls_linears[i+1]))






        # decoder
        # Fully connected layers
        for i in range(len(dec_linears)-1):
            setattr(self, 'dec_lin{}'.format(i), nn.Linear(dec_linears[i], dec_linears[i+1]))
            setattr(self, 'dec_lin_norm{}'.format(i), nn.LayerNorm(dec_linears[i+1]))
            
        # make deconvolutional layers
        for i in range(len(dec_channels)-1):
            setattr(self, 'dec_deconv{}'.format(i), nn.ConvTranspose1d(
                dec_channels[i], dec_channels[i+1], kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding
            ))
            setattr(self, 'dec_deconv_norm{}'.format(i), nn.LayerNorm([dec_channels[i+1], deconv_sizes[i+1]]))
    
        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.enc_linears = enc_linears
        self.dec_linears = dec_linears
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.cls_linears = cls_linears
        self.cls_channels = cls_channels
        self.conv_sizes = conv_sizes
        self.deconv_sizes = deconv_sizes
                
        self.initialize_weights()

        self.input_crop = input_crop
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.output_padding = output_padding

    def forward(self, input, notev, return_cls_out=True):
        mean, var = self.encode(input)

        # reparameterization trick
        epsilon = torch.randn_like(var).to(device)        # sampling epsilon        
        x = mean + torch.exp(var / 2) * epsilon          # reparameterization trick
        
        xhat = self.decode(x, notev)
        
        xcls = None
        if return_cls_out:
            x = xhat[:,:self.input_crop,:]
            x = x.swapaxes(1,2) # for making it batch, channels, time
            
            # cls
            for i in range(0,len(self.cls_channels)-1):
                x = getattr(self, 'cls_conv{}'.format(i))(x)
                x = getattr(self, 'cls_conv_norm{}'.format(i))(x)
                x = self.relu(x)
                x = self.dropout(x)
            
            x = x.view([x.shape[0], -1] )
            for i in range(0,len(self.cls_linears)-1):
                x = getattr(self, 'cls_lin{}'.format(i))(x)
                x = getattr(self, 'cls_lin_norm{}'.format(i))(x)
                x = self.relu(x)
                x = self.dropout(x)
            xcls = x
                
        return xhat, xcls, mean, var


    def encode(self, x):
        x = x[:,:self.input_crop,:]
        x = x.swapaxes(1,2) # for making it batch, channels, time
        
        # encoder
        for i in range(0,len(self.enc_channels)-1):
            x = getattr(self, 'enc_conv{}'.format(i))(x)
            x = getattr(self, 'enc_conv_norm{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        x = x.view([x.shape[0], -1] )
        for i in range(0,len(self.enc_linears)-2):
            x = getattr(self, 'enc_lin{}'.format(i))(x)
            x = getattr(self, 'enc_lin_norm{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)

        mean = self.mean(x)
        var = self.var(x)
       
        return mean, var

    def decode(self, x, notev):
        
        x = torch.concat((x,notev), dim=1)
          
        # decoder
        for i in range(0,len(self.dec_linears)-1):
            x = getattr(self, 'dec_lin{}'.format(i))(x)
            x = getattr(self, 'dec_lin_norm{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = x.view([x.shape[0], self.dec_channels[0], self.deconv_sizes[0]]) # batch, channels, time

        for i in range(0,len(self.dec_channels)-1):
            x = getattr(self, 'dec_deconv{}'.format(i))(x)
            x = getattr(self, 'dec_deconv_norm{}'.format(i))(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = x[:,:,:self.input_crop] # crop it to input crop for calculating loss etc
        x = x.swapaxes(1,2) # for making it batch, time, channels
        
        return x

    def initialize_weights(self):
        for layer in list(self.children()):
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
            if isinstance(layer, nn.ConvTranspose1d):
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero or another suitable value













import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from pitched_experiment_config import ds_folder, ds_buffer, ckpt_vae
import gc


# we want absolute deterministic behaviour here for reproducibility a good looking 2d embedding
seed = 123132
data_seed = 1234

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True)

device = 'cuda'

num_passes = 300 # before 10000
batch_size = 1024
lr = 6e-4 # learning rate
wd = 0.05 # weight decay
betas = (0.9, 0.95) # adam betas

weighted_reproduction = True

# crop the time dimension to this length (150 -> input_crop)
input_crop = 225

# simple vae without convolutional layers
# layers = [128, 64, 32, 16]


# get the audio dataset
# ds_folders = ['/home/chris/data/audio_samples/single_packs/kb6'] # for debugging
# ds_buffer = '/home/chris/data/audio_samples/ds_min.pkl' # for debugging
dsb, ds_train, ds_val, dl_train, dl_val = pdata.get_audio_dataset(ds_folder, 
                dump_path= ds_buffer, 
                build_dump_from_scratch=True, 
                test_size=0.1,
                equalize_class_distribution=True,
                equalize_train_data_loader_distribution=True,
                batch_size=batch_size,
                seed=data_seed,
                num_samples=None
                )


#
# just for visualization during training
#
vis_size = 200
xp = []
yp = []
for d in torch.utils.data.Subset(ds_train,list(range(0,vis_size))):
    xp.append(d[0])
    yp.append(torch.tensor(d[3]))
dsx = torch.stack(xp)    
dsy = torch.stack(yp)




# vae with convolutional layers
enc_linears = [256, 128, 64, 32, 2]
enc_channels = [128,256,128,64]
dec_linears = [90, 128, 256,]
dec_channels = [64, 128, 256, 128]
cls_linears = [256, 128, 64, 32, len(ds_train.ds.classes)]
cls_channels = [128,256,128]



# check for train test split random correctness
print(ds_train[3][1], ds_train[2][1], ds_val[1][1], ds_val[2][1])



vae = ConditionConvVAE(enc_linears, enc_channels, dec_linears, dec_channels, cls_linears, cls_channels, input_crop).to(device)

optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=wd, betas=betas)
loss_fn = nn.MSELoss(reduction='sum')
cls_criterion = nn.CrossEntropyLoss()

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
    for dx, x_transposed, cls, cls_i, cls_v, notei, notev  in dl:
        dx = dx.to(device)
        notev = notev.to(device)
        cls_v = cls_v.to(device)
        x_hat, cls_logits, mean, var = vae.forward(dx,notev,return_cls_out=True)
        cls_loss = cls_criterion(input=cls_logits, target=cls_v)

        rec_loss, kld_loss = loss_fn2(dx, x_hat, mean, var,num_passes)
        losses.append([rec_loss.item(), kld_loss.item(), cls_loss.item()])
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
    for dx, x_transposed, cls, cls_i, cls_v, notei, notev  in dl_train:
        dx = dx.to(device)
        notev = notev.to(device)
        cls_v = cls_v.to(device)
        


        # # reconstruction loss
        # optimizer.zero_grad(set_to_none=True)
        # x_hat, _, mean, var = vae.forward(dx,notev,return_cls_out=False)
        # rec_loss, kld_loss = loss_fn2(dx, x_hat, mean, var, i)
        # loss = rec_loss + kld_loss
        # loss.backward()
        # optimizer.step()
        
        # notev = torch.zeros_like(notev).to(device)

        # # classification loss
        # optimizer.zero_grad(set_to_none=True)
        # x_hat, cls_logits, mean, var = vae.forward(dx,notev,return_cls_out=True)
        # clsloss = cls_criterion(input=cls_logits, target=cls_v)
        # clsloss.backward()
        # optimizer.step()
        


        # classification loss
        optimizer.zero_grad(set_to_none=True)
        x_hat, cls_logits, mean, var = vae.forward(dx,notev,return_cls_out=True)
        rec_loss, kld_loss = loss_fn2(dx, x_hat, mean, var, i)
        clsloss = cls_criterion(input=cls_logits, target=cls_v)
        loss = rec_loss + kld_loss + clsloss
        
        loss.backward()
        optimizer.step()
        



    if i == int(num_passes * 0.7) or i == int(num_passes * 0.8) or i == int(num_passes * 0.9):
        # change learning rate
        lr = lr * 0.1
        optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=wd, betas=betas)
        print('############################# new lr: %.8f' % lr)
        
        # print parameter stats
        print_param_stats(vae)


    if i>0 and i % 2 == 0:

        vae.eval()
        
        train_loss_rec, train_loss_kld, train_loss_cls = det_loss(vae,torch.utils.data.Subset(ds_train,list(range(0,len(ds_val))))  )
        val_loss_rec, val_loss_kld, val_loss_cls = det_loss(vae,ds_val)
        train_losses.append([train_loss_rec, train_loss_kld, train_loss_cls])
        val_losses.append([val_loss_rec, val_loss_kld, val_loss_cls])

        print('pass: %d \t train: %.4f \t train rec: %.4f \t train kld: %.4f \t train cls: %.4f \t val: %.4f \t val rec: %.4f \t val kld: %.4f \t val cls: %.4f' 
            % (i,train_loss_rec+train_loss_kld+train_loss_cls, train_loss_rec, train_loss_kld, train_loss_cls, val_loss_rec+val_loss_kld+val_loss_cls, val_loss_rec, val_loss_kld, val_loss_cls))
        
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
        plt.plot(tp[:,2], label='train cls')
        plt.plot(vp.sum(axis=1), label='val')
        plt.plot(vp[:,0], label='val rec')
        plt.plot(vp[:,1], label='val kld')
        plt.plot(vp[:,2], label='val cls')
        # plt.ylim(0,0.2)
        plt.legend()
        plt.savefig('results/pitched_vae_losses.png')



        if True:

            plt.close(0)
            plt.figure(0)

            classes = ds_train.ds.classes
            for i in range(len(classes)):
                samples_of_class = dsx[dsy==i]
                if len(samples_of_class) > 0:
                    outp_mean, outp_var  = vae.encode(samples_of_class.to(device))    
                    px = outp_mean.cpu().detach().numpy()
                    plt.scatter(px[:,0], px[:,1], label=classes[i], s=0.5, alpha=0.7)

            plt.xlim(-1,1)
            plt.ylim(-1,1)

            plt.legend()
            plt.savefig('results/pitched_vae_latent_distribution.png')

        if False:
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