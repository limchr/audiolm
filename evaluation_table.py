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

from transformer_train_torch import GesamTransformer
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

ckpt = ckpt_transformer



model = torch.load(ckpt)[0]
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

    d_trans = np.zeros(shape=(0,150,128), dtype=np.float32)
    d_vae = np.zeros(shape=(0,150,128), dtype=np.float32)
    d_knn = np.zeros(shape=(0,150,128), dtype=np.float32)
    cd_trans = np.zeros(shape=(0,5), dtype=np.float32)
    cd_vae = np.zeros(shape=(0,5), dtype=np.float32)
    cd_knn = np.zeros(shape=(0,5), dtype=np.float32)

    for d in dl:
        dx = d[0].to(device=device)

        # calculate latent
        condition_bottleneck = condition_model(dx,True)[0]

        # generate with transformer
        gx = model.generate(num_generate=num_generate, condition=condition_bottleneck)

        # generate with vae
        vae_gx = condition_model.decode(condition_bottleneck)

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
        if False:
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
        



        dx_p = discriminator.forward(dx,True)
        gx_p = discriminator.forward(gx,True)
        vae_gx_p = discriminator.forward(vae_gx,True)
        nn_gx_p = discriminator.forward(nn_gx,True)   

        d_trans = np.concatenate((d_trans, (dx-gx).cpu().detach().numpy()), axis=0)
        d_vae = np.concatenate((d_vae, (dx-vae_gx).cpu().detach().numpy()), axis=0)
        d_knn = np.concatenate((d_knn, (dx-nn_gx).cpu().detach().numpy()), axis=0)
        cd_trans = np.concatenate((cd_trans, (dx_p-gx_p).cpu().detach().numpy()), axis=0)
        cd_vae = np.concatenate((cd_vae, (dx_p-vae_gx_p).cpu().detach().numpy()), axis=0)
        cd_knn = np.concatenate((cd_knn, (dx_p-nn_gx_p).cpu().detach().numpy()), axis=0)

    return d_trans, d_vae, d_knn, cd_trans, cd_vae, cd_knn


def mae(input):
    return np.mean(np.abs(input))

def mse(input):
    return np.mean(np.square(input))



def get_score(differences, metric, crop=None):
    if crop is None:
        crop = differences.shape[1]

    return metric(differences[:,:crop])


datasets = [ds_val, ds_train]
dataset_names = ['test set', 'training set']


print(' & Transformer & VAE-Dec & NN-Map \\\\')
print('\\hline')
print('\\hline')


for ds,dsname in zip(datasets, dataset_names):
    d_trans, d_vae, d_knn, cd_trans, cd_vae, cd_knn = get_maes(ds)
    print('\\textbf{'+dsname+'} & & & \\\\')

    print('embedding MAE & %.4f & %.4f & %.4f \\\\' % (get_score(d_trans, mae, None), get_score(d_vae, mae, None), get_score(d_knn, mae, None)))
    print('embedding MAE50 & %.4f & %.4f & %.4f \\\\' % (get_score(d_trans, mae, 75), get_score(d_vae, mae, 75), get_score(d_knn, mae, 75)))
    print('embedding MAE20 & %.4f & %.4f & %.4f \\\\' % (get_score(d_trans, mae, 30), get_score(d_vae, mae, 30), get_score(d_knn, mae, 30)))

    print('classifier MAE & %.4f & %.4f & %.4f \\\\' % (get_score(cd_trans, mae, None), get_score(cd_vae, mae, None), get_score(cd_knn, mae, None)))
    print('\hline')
    print('\hline')

