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
ns = 30 # number of samples for x and y (total samples = ns*ns)
num_generate = 150

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

model, checkpoint = load_model('results/best_so_far.pt')
config = model.config

ss = np.linspace(-1,1,30)


ds = data.EncodecSoundDataset(folder='/home/chris/data/audio_samples/ds_extracted', seed=seed)

condition_cnn = torch.load('results/cnn_best_so_far.pt')
condition_cnn.eval()


bottlenecks = []
numeric_classes = []

for d in ds:
    dx = d[0].unsqueeze(0)
    condition_bottleneck = condition_cnn(dx[:,:32,:],True)[0].cpu().detach().numpy()

    bottlenecks.append(condition_bottleneck)
    numeric_classes.append(d[3])

    
bottlenecks = np.array(bottlenecks, dtype=np.float32)
numeric_classes = np.array(numeric_classes, dtype=np.int32)

np.savetxt('results/z.csv', bottlenecks, delimiter=',', newline='\n', fmt='%.6f')
np.savetxt('results/y.csv', numeric_classes, delimiter=',', newline='\n', fmt='%d')




for xi,x in enumerate(ss):
    for yi,y in enumerate(ss):
        gx = torch.zeros((1,1,config.vocab_size), dtype=torch.float32).to(device)
    # gx[:,:,:] = dx[clsi%dx.shape[0],0,:]
        for i in range(num_generate):
            ng = model.forward(gx, None, torch.tensor([[x,y]], dtype=torch.float32).to(device=device))[0]
            gx = torch.cat((gx, ng), dim=1)
        wav = ds.decode_sample(gx)
        ds.save_audio(wav, f'results/samples/generated_%05d_%05d.wav' % (xi,yi))


pass