import torch
import torch.nn.functional as F
import torchaudio


from audiolm_pytorch.data import get_audio_dataset
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass

import audiolm_pytorch.gpt as gpt

from audiolm_pytorch import EncodecWrapper

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

from transformer_train import load_model
from experiment_config import ds_folders, ds_buffer, ckpt_vae, ckpt_transformer, ckpt_discriminator

import os
import shutil

device = 'cuda'
seed = 1234
num_generate = 150
nneighbors = 5
# visualization_area = [-0.8, 0.5, -0.6, 0.4] # area to be sampled (where training data is within the embedding space xmin, xmax, ymin, ymax)

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)



model, checkpoint = load_model(ckpt_transformer)
config = model.config

condition_model = torch.load(ckpt_vae)
condition_model.eval()

discriminator = torch.load(ckpt_discriminator)
discriminator.eval()



# get the audio dataset
dsb, ds_train, ds_val, dl_train, dl_val = get_audio_dataset(audiofile_paths= ds_folders,
                                                                dump_path= ds_buffer,
                                                                build_dump_from_scratch=False,
                                                                only_labeled_samples=True,
                                                                test_size=0.1,
                                                                equalize_class_distribution=True,
                                                                equalize_train_data_loader_distribution=True,
                                                                batch_size=512,
                                                                seed=seed)
# check for train test split random correctness
print(ds_train[123][3], ds_train[245][3], ds_val[456][3], ds_val[125][3])

# get the bottlenecks for nn model (always train set)
bottlenecks = []
numeric_classes = []
for d in ds_train:
    dx = d[0].unsqueeze(0).to(device=device)
    condition_bottleneck = condition_model(dx,True)[0][0].cpu().detach().numpy()
    bottlenecks.append(condition_bottleneck)
    numeric_classes.append(d[3].item())
bottlenecks = np.array(bottlenecks, dtype=np.float32)
# project bottlenecks to visualization space with model to visualization space function
numeric_classes = np.array(numeric_classes, dtype=np.int32)




def get_maes(ds):
    for d in ds:
        dx = d[0].unsqueeze(0).to(device=device)

        # calculate latent
        condition_bottleneck = condition_model(dx,True)[0]

        # generate with transformer
        gx = model.generate(num_generate=num_generate-1, condition=condition_bottleneck)

        # generate with vae
        vae_gx = condition_model.decoder(condition_bottleneck)
        vae_gx = vae_gx.swapaxes(1,2)

        sort_i = np.argsort(np.linalg.norm(bottlenecks - condition_bottleneck[0,:].cpu().detach().numpy(), axis=1))[:nneighbors]
        nn_gx = torch.zeros((1,150,128)).to(device)
        for i in sort_i:
            dxn = ds_train[i][0].to(device) # always train set for nn model
            nn_gx += dxn
        nn_gx /= nneighbors

        # cropping because VAE has crop, to make a fair comparison
        cropt = 64

        mae_trans = torch.mean(torch.abs(dx[0,:cropt,:] - gx[0,:cropt,:])).item()
        mae_vae = torch.mean(torch.abs(dx[0,:cropt,:] - vae_gx[0,:cropt,:])).item()
        mae_nn = torch.mean(torch.abs(dx[0,:cropt,:] - nn_gx[0,:cropt,:])).item()

        # use classifier for MAE of posterior probabilities
        
        dx_p = discriminator.forward(dx,True)
        gx_p = discriminator.forward(gx,True)
        vae_gx_p = discriminator.forward(vae_gx,True)
        nn_gx_p = discriminator.forward(nn_gx,True)   

        mae_trans_cp = torch.mean(torch.abs(dx_p - gx_p)).item()
        mae_vae_cp = torch.mean(torch.abs(dx_p - vae_gx_p)).item()
        mae_nn_cp = torch.mean(torch.abs(dx_p - nn_gx_p)).item()

        return mae_trans, mae_vae, mae_nn, mae_trans_cp, mae_vae_cp, mae_nn_cp


print('Calculating MAEs for train set (transformer, vae, nn)')
maes_train = get_maes(ds_train)
print('Calculating MAEs for val set (transformer, vae, nn)')
maes_val = get_maes(ds_val)

print(' & Transformer & VAE & NN \\\\')
print('training set & & & \\')
print('embedding MAE & %.4f & %.4f & %.4f \\\\' % (maes_train[0], maes_train[1], maes_train[2]))
print('classifier MAE & %.4f & %.4f & %.4f \\\\' % (maes_train[3], maes_train[4], maes_train[5]))
print('test set & & & \\')
print('embedding MAE & %.4f & %.4f & %.4f \\\\' % (maes_val[0], maes_val[1], maes_val[2]))
print('classifier MAE & %.4f & %.4f & %.4f \\\\' % (maes_val[3], maes_val[4], maes_val[5]))

