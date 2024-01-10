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

from encodec_transformer import load_model, AudioConfig



device = 'cuda'
seed = 1234

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

model, checkpoint = load_model('results/ckpt.pt')

ds = data.EncodecSoundDataset(folder='/home/chris/data/audio_samples/ds_extracted', seed=seed)
dsb = data.BufferedDataset(ds, '/home/chris/data/buffered_ds_extracted.pkl', False)

x = []
y = []

for d in dsb:
    x.append(d[0])
    y.append(d[2])

x = torch.stack(x)    
y = torch.stack(y)

plt.ion()
plt.figure()



for i in range(0,5):
    cl = [True if cc[i] == 1.0 else False for cc in y ]
    cl_name = list(ds.class_groups.keys())[i]
    outp = model.transformer.cond_bn2(x[cl])    
    px = outp.cpu().detach().numpy()
    plt.scatter(px[:,0], px[:,1], label=cl_name)

plt.legend()
plt.show()
plt.savefig('results/cond_plot.png')



pass