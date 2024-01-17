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
config = model.config

ss = np.linspace(0,1,10)

ds = data.EncodecSoundDataset(folder='/home/chris/data/audio_samples/ds_extracted', seed=seed)

num_generate = 150

for xi,x in enumerate(ss):
    for yi,y in enumerate(ss):
        gx = torch.zeros((1,1,config.vocab_size), dtype=torch.float32).to(device)
    # gx[:,:,:] = dx[clsi%dx.shape[0],0,:]
        for i in range(num_generate):
            ng = model.forward(gx, None, torch.tensor([[x,y]], dtype=torch.float32).to(device=device))[0]
            gx = torch.cat((gx, ng), dim=1)

        wav = ds.decode_sample(gx)
        ds.save_audio(wav, f'results/generate_%d_%d.wav' % (xi,yi))


pass