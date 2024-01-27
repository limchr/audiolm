import torch
import torch.nn.functional as F
import torchaudio

import audiolm_pytorch.data as data
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass

import audiolm_pytorch.gpt as gpt

from audiolm_pytorch import EncodecWrapper

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from transformer_train import load_model, AudioConfig

device = 'cuda'
seed = 1234

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

model, checkpoint = load_model('results/ckpt.pt')

conds = checkpoint['conditions']

config = model.config

ds = data.EncodecSoundDataset(folder='/home/chris/data/audio_samples/ds_extracted', seed=seed)

# generate samples
model.eval()
num_generate = 150
emb_sample_classes = []
for clsi, cls in enumerate(conds):
    gx = torch.zeros((1,1,config.vocab_size)).to(device)
    # gx[:,:,:] = dx[clsi%dx.shape[0],0,:]

    sample_class = torch.zeros((1,conds.__len__())).to(device=device)
    sample_class[0,clsi] = 1.0

    # emb_sample_class = model.transformer.class_cond_layer[0](sample_class)
    # emb_sample_classes.append(emb_sample_class)


    for i in range(num_generate):
        ng = model.forward(gx, None, sample_class)[0]
        gx = torch.cat((gx, ng), dim=1)

    wav = ds.decode_sample(gx)
    ds.save_audio(wav, f'results/generate_{cls}.wav')



# d = torch.cat(emb_sample_classes).to('cpu').detach().numpy()

# # plot condition bottleneck
# fig, ax = plt.subplots()
# cmap = matplotlib.cm.get_cmap('Spectral')

# for i in range(len(conds)):
#     ax.scatter(d[i,0],d[i,1], c=cmap(i/20.), cmap='tab20', label=conds[i])
#     ax.annotate(conds[i], (d[i,0],d[i,1]))
# plt.legend()
# plt.savefig('results/condition_bottleneck.png')
# plt.show()


# # plot attention weights
# plt.figure()
# plt.imshow(model.transformer.h[0].attn.c_attn.weight.cpu().detach().numpy())
# plt.savefig('results/att_layer.png')
# plt.show()
