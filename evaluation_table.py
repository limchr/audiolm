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
batch_size = 512
# visualization_area = [-0.8, 0.5, -0.6, 0.4] # area to be sampled (where training data is within the embedding space xmin, xmax, ymin, ymax)

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)



model, checkpoint = load_model(ckpt_transformer)
config = model.config
model.eval()

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
                                                                batch_size=batch_size,
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
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    maes = np.zeros((0,6), dtype=np.float32)
    mses = np.zeros((0,6), dtype=np.float32)

    for d in dl:
        dx = d[0].to(device=device)

        # calculate latent
        condition_bottleneck = condition_model(dx,True)[0]

        # generate with transformer
        gx = model.generate(num_generate=num_generate-1, condition=condition_bottleneck)

        # generate with vae
        vae_gx = condition_model.decoder(condition_bottleneck)
        vae_gx = vae_gx.swapaxes(1,2)

        nn_gx = torch.zeros((dx.shape[0],150,128)).to(device)

        for bi in range(dx.shape[0]):
            sort_i = np.argsort(np.linalg.norm(bottlenecks - condition_bottleneck[bi,:].cpu().detach().numpy(), axis=1))[:nneighbors]
            averagenn = torch.zeros((1,150,128)).to(device)
            for i in sort_i:
                dxn = ds_train[i][0].to(device) # always train set for nn model
                averagenn += dxn
            averagenn /= nneighbors
            nn_gx[bi,:,:] = averagenn

        # plot with matplotlib all four images dx, gx vae_gx and nn_gx squared below each other
        plt.figure(figsize=(10,15))
        # remove all axes
        plt.axis('off')
        # remove all ticks
        
        plt.subplot(4,1,1)
        plt.imshow(dx[0,:,:].cpu().detach().numpy().T, aspect='auto', origin='lower')
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.title('original')
        
        plt.subplot(4,1,2)
        plt.imshow(gx[0,:,:].cpu().detach().numpy().T, aspect='auto', origin='lower')
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.title('transformer generated')

        plt.subplot(4,1,3)
        plt.imshow(vae_gx[0,:,:].cpu().detach().numpy().T, aspect='auto', origin='lower')
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.title('VAE generated')        

        plt.subplot(4,1,4)
        plt.imshow(nn_gx[0,:,:].cpu().detach().numpy().T, aspect='auto', origin='lower')
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.title('NN generated')
        
        plt.tight_layout()
        plt.savefig('results/embedding_comparison.png')
    















        # cropping because VAE has crop, to make a fair comparison
        cropt = 150
        
        dx_p = discriminator.forward(dx,True)
        gx_p = discriminator.forward(gx,True)
        vae_gx_p = discriminator.forward(vae_gx,True)
        nn_gx_p = discriminator.forward(nn_gx,True)   


        mae_trans = torch.mean(torch.abs(dx[:,:cropt,:] - gx[:,:cropt,:])).item()
        mae_vae = torch.mean(torch.abs(dx[:,:cropt,:] - vae_gx[:,:cropt,:])).item()
        mae_nn = torch.mean(torch.abs(dx[:,:cropt,:] - nn_gx[:,:cropt,:])).item()
        mae_trans_cp = torch.mean(torch.abs(dx_p - gx_p)).item()
        mae_vae_cp = torch.mean(torch.abs(dx_p - vae_gx_p)).item()
        mae_nn_cp = torch.mean(torch.abs(dx_p - nn_gx_p)).item()
        # concatenate to maes
        maes = np.concatenate((maes, np.array([[mae_trans, mae_vae, mae_nn, mae_trans_cp, mae_vae_cp, mae_nn_cp]], dtype=np.float32)), axis=0)
        
        mse_trans = torch.mean(torch.square(dx[:,:cropt,:] - gx[:,:cropt,:])).item()
        mse_vae = torch.mean(torch.square(dx[:,:cropt,:] - vae_gx[:,:cropt,:])).item()
        mse_nn = torch.mean(torch.square(dx[:,:cropt,:] - nn_gx[:,:cropt,:])).item()
        mse_trans_cp = torch.mean(torch.square(dx_p - gx_p)).item()
        mse_vae_cp = torch.mean(torch.square(dx_p - vae_gx_p)).item()
        mse_nn_cp = torch.mean(torch.square(dx_p - nn_gx_p)).item()
        # concatenate to mses
        mses = np.concatenate((mses, np.array([[mse_trans, mse_vae, mse_nn, mse_trans_cp, mse_vae_cp, mse_nn_cp]], dtype=np.float32)), axis=0)

    return np.mean(maes, axis=0), np.mean(mses, axis=0)


print('Calculating MAEs for train set (transformer, vae, nn)')
maes_train, mses_train = get_maes(ds_train)
print('Calculating MAEs for val set (transformer, vae, nn)')
maes_val, mses_val = get_maes(ds_val)

print(' & Transformer & VAE & NN \\\\')
print('\\hline')
print('\\hline')
print('\\textbf{training set} & & & \\\\')
print('embedding MAE & %.4f & %.4f & %.4f \\\\' % (maes_train[0], maes_train[1], maes_train[2]))
print('classifier MAE & %.4f & %.4f & %.4f \\\\' % (maes_train[3], maes_train[4], maes_train[5]))
print('\\hline')
print('\\textbf{test set} & & & \\\\')
print('embedding MAE & %.4f & %.4f & %.4f \\\\' % (maes_val[0], maes_val[1], maes_val[2]))
print('classifier MAE & %.4f & %.4f & %.4f \\\\' % (maes_val[3], maes_val[4], maes_val[5]))
print('\hline')
print('\hline')

print(' & Transformer & VAE & NN \\\\')
print('training set & & & \\')
print('embedding MSE & %.4f & %.4f & %.4f \\\\' % (mses_train[0], mses_train[1], mses_train[2]))
print('classifier MSE & %.4f & %.4f & %.4f \\\\' % (mses_train[3], mses_train[4], mses_train[5]))
print('test set & & & \\')
print('embedding MSE & %.4f & %.4f & %.4f \\\\' % (mses_val[0], mses_val[1], mses_val[2]))
print('classifier MSE & %.4f & %.4f & %.4f \\\\' % (mses_val[3], mses_val[4], mses_val[5]))

