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

from experiment_config import ds_folders, ds_buffer, ckpt_vae, ckpt_transformer, ckpt_transformer_latest, ckpt_discriminator, ns

seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


import os
import shutil

outdir = 'results/samples'

import pickle as pkl



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






generated_map = None
with open('results/map_data.pkl', 'rb') as f:
    generated_map = pkl.load(f)

classes = ds_train.ds.classes


plt.close()
# Define the figure and subplot grid
fig, axs = plt.subplots(2, 3, figsize=(10, 10))

model_labels = ['Transformer', 'VAE-Dec', 'kNN-Map']
models = ['transformer', 'vae', 'nn']


# Loop through each subplot
for i in range(3): # models
    for j in range(2): # 0 for predicted classes, 1 for margin
        # Create an image for the subplot
        img = np.zeros((ns, ns), dtype=np.int32 if j == 0 else np.float32)
        for xx in range(ns):
            for yy in range(ns):
                img[xx, yy] = generated_map[xx][yy]['classifications'][models[i]][j]


        if j==1:
            # Plot the image on the current subplot
            im = axs[j, i].imshow(img, interpolation='none', vmin=0, vmax=1.0, cmap='Grays')
        if j==0:
            im = axs[j, i].imshow(img, interpolation='none', cmap="tab10")

            # Get the colors of the values, according to the colormap used by imshow
            colors = [im.cmap(im.norm(value)) for value in range(len(classes))]

            # Create a patch (proxy artist) for every color
            patches = [matplotlib.patches.Patch(color=colors[k], label=classes[k]) for k in range(len(classes))]

            # Put those patched as legend-handles into the legend
            # axs[j, i].legend(handles=patches, borderaxespad=0., loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(classes)//2)

            axs.flat[i].set_title(model_labels[i])




        axs[j, i].grid(True, which='both', axis='both', linestyle='--', color='k', linewidth=0.1)
        axs[j, i].set_xticks(np.arange(0, ns, 1) + 0.5)
        axs[j, i].set_yticks(np.arange(0, ns, 1) + 0.5)
        axs[j, i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                              labelbottom=False, labelleft=False)

# Create a single legend outside of the subplots
plt.legend(handles=patches, loc='lower center', bbox_to_anchor=(-0.5, 1.0), ncol=len(classes))
plt.gcf().set_size_inches(20, 10)


# Show the plot
plt.tight_layout()
plt.savefig('results/classification_map.png')
# plt.show()


